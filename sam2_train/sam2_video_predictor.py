# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from sympy import im
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
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj


    # @torch.inference_mode()
    def forward(
        self,
        imgs_tensor,
        prompt,
        normalize_coords=True,
    ):
        
        video_height = self.image_size
        video_width = self.image_size
        # images = load_video_frames_from_data(
        #     imgs_tensor=imgs_tensor,
        #     offload_video_to_cpu=False,
        #     async_loading_frames=False,
        # )
        images = imgs_tensor
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = False

        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        inference_state["storage_device"] = torch.device("cuda")

        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        inference_state["constants"] = {}

        inference_state["modality_memory"] = {}  # {frame_idx: <prompt>}
        inference_state["slice_memory"] = {}
        inference_state["self_prompt"] = {} # {frame_idx: <self_prompt>}
        inference_state["modality_mask"] = {}
        
        bs = inference_state['num_frames']      

        output_dict = inference_state["output_dict"]
        outputs_mask = []
        outputs_mask_low = []

        points = prompt.reshape(-1, 2, 2)
        labels = torch.tensor([2, 3], dtype=torch.int)

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        for frame_idx in range(bs):
            backbone_out = self.forward_image(inference_state["images"][frame_idx].unsqueeze(0))
            current_vision_feats, current_vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
            storage_key = "cond_frame_outputs"

            # 检查points[frame_idx]是否有nan值
            if torch.isnan(points[frame_idx]).any():
                point_inputs = None
            else:
                point_inputs = concat_points(None, points[frame_idx].unsqueeze(0), labels)

            current_out, pred_masks = self._run_single_frame_inference(
                inference_state = inference_state,
                current_vision_feats = current_vision_feats,
                current_vision_pos_embeds = current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                frame_idx=frame_idx,
                is_init_cond_frame=False if frame_idx > 0 else True,
                point_inputs=point_inputs if prompt is not None else None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
            inference_state["modality_mask"][frame_idx] = pred_masks
            self_prompt = current_out['self_prompt']
            outputs_mask.append(pred_masks.squeeze(0))
            outputs_mask_low.append(self_prompt.squeeze(0))
        time_end = time.time()
        return torch.stack(outputs_mask, dim=0), torch.stack(outputs_mask_low, dim=0)


    def _run_single_frame_inference(
        self,
        inference_state,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        frame_idx,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # print(is_init_cond_frame)
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

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]

        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks_high_res"]

        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        # make a compact version of this frame's output to reduce the state size
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

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )

        
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc
