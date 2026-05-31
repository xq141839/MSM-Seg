# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch

from tqdm import tqdm
import time
from sam2_train.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2_train.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames, load_video_frames_from_data


class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    def forward(
        self,
        imgs_tensor,
        prompt,
    ):
        video_height = self.image_size
        video_width = self.image_size
        images = imgs_tensor
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # =====================================================================
        # [显存优化] 官方特征转移 CPU (可根据需要开启，当前保持 False 避免速度损耗)
        # =====================================================================
        inference_state["offload_video_to_cpu"] = False

        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        
        inference_state["device"] = imgs_tensor.device
        inference_state["storage_device"] = imgs_tensor.device

        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  
            "non_cond_frame_outputs": {},  
        }
        inference_state["constants"] = {}
        inference_state["modality_memory"] = {}  
        inference_state["slice_memory"] = {}
        inference_state["self_prompt"] = {} 
        inference_state["modality_mask"] = {}
        
        bs = inference_state['num_frames']      
        output_dict = inference_state["output_dict"]
        outputs_mask = []
        outputs_mask_low = []

        # =====================================================================
        # 批量并行提取 ViT 图像特征
        # =====================================================================
        chunk_size = 16  
        all_vision_feats = []
        all_vision_pos_embeds = []
        global_feat_sizes = None
        
        for start_idx in range(0, bs, chunk_size):
            end_idx = min(start_idx + chunk_size, bs)
            img_chunk = inference_state["images"][start_idx:end_idx]
            
            backbone_out = self.forward_image(img_chunk)
            vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
            
            if global_feat_sizes is None:
                global_feat_sizes = feat_sizes
                all_vision_feats = [[] for _ in range(len(vision_feats))]
                all_vision_pos_embeds = [[] for _ in range(len(vision_pos_embeds))]
                
            for lvl in range(len(vision_feats)):
                all_vision_feats[lvl].append(vision_feats[lvl])
                all_vision_pos_embeds[lvl].append(vision_pos_embeds[lvl])
                
        all_vision_feats = [torch.cat(feats, dim=1) for feats in all_vision_feats]
        all_vision_pos_embeds = [torch.cat(pos, dim=1) for pos in all_vision_pos_embeds]

        for frame_idx in range(bs):
            current_vision_feats = [feat[:, frame_idx:frame_idx+1, :] for feat in all_vision_feats]
            current_vision_pos_embeds = [pos[:, frame_idx:frame_idx+1, :] for pos in all_vision_pos_embeds]

            # =====================================================================
            # [修改] box/点提示只用于引导 self_prompt(MCP-Encoder)，与"条件帧/记忆"解耦。
            # 否则若给每个含肿瘤的切片都加提示，会让几乎每帧都变成零记忆初始帧，把双记忆旁路掉。
            # =====================================================================
            frame_prompt = None
            if prompt is not None:
                try:
                    if prompt[frame_idx] is not None:
                        frame_prompt = prompt[frame_idx]
                except (IndexError, KeyError, TypeError):
                    frame_prompt = None

            # 条件帧/记忆只看是否第 0 帧，不再受 prompt 影响
            is_init_cond_frame = (frame_idx == 0)
            storage_key = "cond_frame_outputs" if is_init_cond_frame else "non_cond_frame_outputs"

            current_out, pred_masks = self._run_single_frame_inference(
                inference_state = inference_state,
                current_vision_feats = current_vision_feats,
                current_vision_pos_embeds = current_vision_pos_embeds,
                feat_sizes=global_feat_sizes,
                output_dict=output_dict,
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=frame_prompt,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
            inference_state["modality_mask"][frame_idx] = pred_masks
            self_prompt = current_out['self_prompt']
            
            outputs_mask.append(pred_masks.squeeze(0))
            outputs_mask_low.append(self_prompt.squeeze(0))
            
            # ===================================================
            # [显存优化]：动态释放过期的历史记忆
            # 最多回溯 7 个 slice (约 28 帧)，我们保留最近 32 帧即可，更老的直接删掉
            # ===================================================
            expire_idx = frame_idx - 32
            if expire_idx >= 0:
                if expire_idx in output_dict["non_cond_frame_outputs"]:
                    del output_dict["non_cond_frame_outputs"][expire_idx]
                if expire_idx in inference_state["modality_mask"]:
                    del inference_state["modality_mask"][expire_idx]
                if getattr(inference_state, "self_prompt", None) and expire_idx in inference_state["self_prompt"]:
                    del inference_state["self_prompt"][expire_idx]
                    
            # 定期清理 PyTorch 显存碎片，保持显存水位健康
            if frame_idx % 40 == 39:
                torch.cuda.empty_cache()
            
        stacked_masks = torch.stack(outputs_mask, dim=0)        # [N,1,H,W] 各(切片,模态)预测 Ŷ_{t,m}
        stacked_low = torch.stack(outputs_mask_low, dim=0)      # [N,1,H,W] 自提示 guidance

        # =====================================================================
        # [新增] 论文 Eq.9 模态自适应融合：把每个切片的 M 个模态预测融合成最终 Ŷ_t
        # 帧按切片优先交错排列(frame = t*M + m)，故 view(L,M,H,W) 即可把每切片的 M 个模态归到一组
        # =====================================================================
        N = stacked_masks.shape[0]
        M = getattr(self, 'num_modality', 4)
        C = stacked_masks.shape[1]                          # 类别数 (WT/TC/ET = 3)
        H, W = stacked_masks.shape[-2], stacked_masks.shape[-1]
        if M > 0 and N % M == 0:
            L = N // M
            # [L,M,C,H,W]：每切片 M 个模态、每模态 C 个类别预测 Ŷ_{t,m}
            mod_logits = stacked_masks.view(L, M, C, H, W)
            # 把类别折进 batch，对每个类别独立地在 M 个模态上做自适应融合(Eq.9)
            x = mod_logits.permute(0, 2, 1, 3, 4).reshape(L * C, M, H, W)  # [L*C, M, H, W]
            fused_x, _ = self.modality_fusion(x)            # [L*C, 1, H, W]
            fused = fused_x.view(L, C, H, W)                # [L, C, H, W] = Ŷ_t (3 个区域)
        else:
            # 兜底：帧数非 M 整数倍时退回取最后一个模态
            fused = stacked_masks[(M - 1)::M] if M > 0 else stacked_masks

        return stacked_masks, stacked_low, fused

    def _run_single_frame_inference(
        self, inference_state, current_vision_feats, current_vision_pos_embeds,
        feat_sizes, output_dict, frame_idx, is_init_cond_frame,
        point_inputs, mask_inputs, reverse, run_mem_encoder, prev_sam_mask_logits=None,
    ):
        current_out = self.track_step(
            inference_state=inference_state,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        pred_masks_gpu = current_out["pred_masks_high_res"]
        pred_masks = pred_masks_gpu 
        
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        self_prompt = current_out.get("self_prompt", None)
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": None,
            "self_prompt": self_prompt,
        }

        if inference_state['self_prompt'].get(frame_idx,None) is None:
            inference_state['self_prompt'][frame_idx] = self_prompt
            
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts):
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
        )
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc