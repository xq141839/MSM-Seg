# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from torch.nn.init import trunc_normal_

from sam2_train.modeling.sam.mask_decoder import MaskDecoder
from sam2_train.modeling.sam.prompt_encoder import PromptEncoder
from sam2_train.modeling.sam.transformer import TwoWayTransformer
from sam2_train.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from sam2_train.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2_train.modeling.sam2_self_prompt import PyramidPromptUNet
from sam2_train.modeling.split_attention import SplitAttention
# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -4.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        num_modality=4,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        # 新增：模态memory attention
        self.modality_memory_attention = MemoryAttention(
            d_model=memory_attention.d_model,
            pos_enc_at_input=memory_attention.pos_enc_at_input,
            layer=MemoryAttentionLayer(
                activation=memory_attention.layers[0].activation_str,
                cross_attention=memory_attention.layers[0].cross_attn_image,
                d_model=memory_attention.d_model,
                dim_feedforward=memory_attention.layers[0].dim_feedforward,
                dropout=memory_attention.layers[0].dropout_value,
                pos_enc_at_attn=memory_attention.layers[0].pos_enc_at_attn,
                pos_enc_at_cross_attn_keys=memory_attention.layers[0].pos_enc_at_cross_attn_keys,
                pos_enc_at_cross_attn_queries=memory_attention.layers[0].pos_enc_at_cross_attn_queries,
                self_attention=memory_attention.layers[0].self_attn,
            ),
            num_layers=memory_attention.num_layers,
            batch_first=memory_attention.batch_first,
        )
        

        # 新增：Split Attention 用于增强特征多样性
        self.split_attention = SplitAttention(channels=memory_attention.d_model)
        self.hidden_dim = memory_attention.d_model
        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = 64
        # print(self.mem_dim)
        # if hasattr(self.memory_encoder, "out_proj") and hasattr(
        #     self.memory_encoder.out_proj, "weight"
        # ):
        #     # if there is compression of memories along channel dim
        #     self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        # print(self.mem_dim)
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj

        self._build_sam_heads()
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.num_modality = num_modality
        self.self_prompt = PyramidPromptUNet()
        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            # print(
            #     "Image encoder compilation is enabled. First forward pass will be slow."
            # )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        channels = 256
        radix = 4
        #权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, radix, 1),  # 2个split的权重
            nn.Softmax(dim=1)
        )
 
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference."
            "See notebooks/video_predictor_example.ipynb for an example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        #print(self.sam_image_embedding_size)
        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=4,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        frame_idx,
        inference_state,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = 1#backbone_features.size(0)
        device = backbone_features.device
        # print(backbone_features.size(2), self.sam_image_embedding_size)
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            # print(sam_point_coords.size(), sam_point_labels.size())
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            # print(mask_inputs.shape[:2], (B, 1))
#             print(mask_inputs.shape[-2:], self.sam_prompt_encoder.mask_input_size)
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        
        low_res_multimasks = self.sam_mask_decoder(
            mode='mask',
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            frame_idx=frame_idx,
            inference_state=inference_state,
            high_res_features=high_res_features,
        )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        
        modal_index = frame_idx % 4
        if modal_index==3:
            t2 = high_res_multimasks
            t1c = inference_state['modality_mask'][frame_idx-1]
            t1 = inference_state['modality_mask'][frame_idx-2]
            flair = inference_state['modality_mask'][frame_idx-3]

            modal_weights = self.weight_gen(backbone_features)  
            flair_weight = modal_weights[:, 0:1, :, :].squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            t1_weight = modal_weights[:, 1:2, :, :].squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            t1c_weight = modal_weights[:, 2:3, :, :].squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            t2_weight = modal_weights[:, 3:4, :, :].squeeze(0).squeeze(0).squeeze(0).squeeze(0)

            high_res_multimasks = t2 * t2_weight + t1c * t1c_weight + t1 * t1_weight + flair * flair_weight
            # high_res_multimasks = t2 + t1c + t1 + flair

        low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
        return (low_res_masks, high_res_masks)


    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
    
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        # print(not is_init_cond_frame)
        # print(frame_idx)
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"]
                #print(feats.shape)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))#[[4096, 1, 64]]
                #print(to_cat_memory[-1].shape)
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                # print(maskmem_enc.shape, self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1].shape)
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (abs(frame_idx - t), out["obj_ptr"])
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention

                num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid emtpy memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        # print(to_cat_memory[0].shape, to_cat_memory[1].shape)
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        
        # 使用Split Attention增强特征多样性
        current_feat_tensor = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)  # [B, C, H, W]
        
        # 构建模态间memory的参数
        modal_memory_kwargs = self._build_modal_memory_kwargs(
            frame_idx, is_init_cond_frame, output_dict, num_frames, B, current_vision_pos_embeds
        )
        
        # 构建帧间memory的参数
        temporal_memory_kwargs = {
            'curr_pos': current_vision_pos_embeds,
            'memory': memory,
            'memory_pos': memory_pos_embed,
            'num_obj_ptr_tokens': num_obj_ptr_tokens,
        }
        
        # 使用Split Attention处理特征
        enhanced_features = self.split_attention(
            current_feat_tensor,
            temporal_memory_module=self.memory_attention,
            modal_memory_module=self.modality_memory_attention,
            temporal_kwargs=temporal_memory_kwargs,
            modal_kwargs=modal_memory_kwargs
        )
        return enhanced_features

    def _prepare_modality_memory_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        pix_feat_with_frame_mem=None,
    ):
        """
        对pix_feat_with_frame_mem再进行一次模态memory attention。
        融合最近num_modality帧（每帧一个模态），tpos编码为各自模态编号，条件帧tpos=0。
        """
        current_modality = frame_idx % self.num_modality
        # 只有一个模态或当前帧为初始条件帧时，直接返回
        if self.num_modality == 1 or is_init_cond_frame:
            return pix_feat_with_frame_mem
        #print("当前模态编号:", current_modality)
        # Step 1: 从最近的前几个模态中收集memory
        to_cat_memory, to_cat_memory_pos_embed = [], []
        for i in range(1,current_modality+1): # 1, 2, ..., current_modality
            idx = frame_idx - i # 当前要读取的帧编号,从当前帧开始往前读取，直到模态id为0
            if idx < 0:
                continue
            m = idx % self.num_modality  # 该帧的模态编号
            out = None
            tpos = None
            # 优先从cond_frame_outputs取，若取到则tpos=0
            if idx in output_dict["cond_frame_outputs"]:
                out = output_dict["cond_frame_outputs"][idx]
                tpos = 0
            elif idx in output_dict["non_cond_frame_outputs"]:
                out = output_dict["non_cond_frame_outputs"][idx]
                tpos = i  # 非条件帧，tpos为i，也就是说越近的帧tpos越小
            if out is None:
                continue
            feats = out["maskmem_features"]
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))  # [HW, B, C]
            # 空间位置编码
            maskmem_enc = out["maskmem_pos_enc"][-1]
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            # tpos编码
            if tpos == 0:
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[-1]
            else:
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_modality - tpos - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        if len(to_cat_memory) == 0:
            return pix_feat_with_frame_mem

        # Step 2: 拼接memory和pos，送入modality memory attention
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_modality_mem = self.modality_memory_attention(
            curr=[pix_feat_with_frame_mem],
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=0,
        )

        return pix_feat_with_modality_mem
        
    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        return maskmem_features, maskmem_pos_enc

    def track_step(
        self,
        inference_state,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
        )

        self_prompt = self.self_prompt(
            high_res_features = high_res_features,
            pix_feat_with_mem = pix_feat_with_mem,
            past_mask = None,
            point_coords = point_inputs["point_coords"] if point_inputs is not None else None,
            point_labels = point_inputs["point_labels"] if point_inputs is not None else None,
            scribbles = None,
            box = None,
            use_prompts = False,
            input_resolution = self.image_size  # 传递图像尺寸作为坐标系分辨率
        )

        # print(point_inputs)
        sam_outputs = self._forward_sam_heads(
            frame_idx=frame_idx,
            inference_state=inference_state,
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            mask_inputs=torch.sigmoid(self_prompt),
            high_res_features=high_res_features,
            multimask_output=True,
        )

        (
            low_res_masks,
            high_res_masks,
        ) = sam_outputs

        # print(sam_outputs)

        # print(high_res_masks.shape)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["self_prompt"] = self_prompt
        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                is_mask_from_pts=(point_inputs is not None),
            )
#             print(maskmem_features, maskmem_pos_enc)
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _build_modal_memory_kwargs(self, frame_idx, is_init_cond_frame, output_dict, num_frames, B, curr_pos):

        """构建模态间Memory Attention的参数 - 完全匹配_prepare_modality_memory_features的逻辑"""
        # print(is_init_cond_frame)
        # print(f"🔍 [模态Memory] 帧{frame_idx}: 构建模态间memory参数")
        if self.num_modality == 1 or is_init_cond_frame:
            # print(f"  → 单模态或初始条件帧，使用空memory")
            # print(self.no_mem_embed.shape, self.mem_dim)
            return {
                'curr_pos': curr_pos,
                'memory': self.no_mem_embed.expand(1, B, self.mem_dim),
                'memory_pos': self.no_mem_pos_enc.expand(1, B, self.mem_dim),
                'num_obj_ptr_tokens': 0,
            }
        
        current_modality = frame_idx #% self.num_modality
        to_cat_memory = []
        to_cat_memory_pos_embed = []
        found_frames = []
        
        # print(f"  → 当前模态: {current_modality}")
        # print(f"  → 逻辑帧布局: 每{self.num_modality}帧为一组，当前帧{frame_idx}在组内位置{current_modality}")
        
        # 完全按照_prepare_modality_memory_features的逻辑
        for i in range(1, current_modality + 1):  # 1, 2, ..., current_modality
            idx = frame_idx - i  # 当前要读取的帧编号,从当前帧开始往前读取，直到模态id为0
            if idx < 0:
                continue
                
            target_modality = idx % self.num_modality  # 该帧的模态编号
            out = None
            tpos = None
            
            # 优先从cond_frame_outputs取，若取到则tpos=0
            if idx in output_dict.get("cond_frame_outputs", {}):
                out = output_dict["cond_frame_outputs"][idx]
                tpos = 0
                # print(f"    找到条件帧 {idx} (模态{target_modality}) -> tpos=0")
            elif idx in output_dict.get("non_cond_frame_outputs", {}):
                out = output_dict["non_cond_frame_outputs"][idx]
                tpos = i  # 非条件帧，tpos为i，也就是说越近的帧tpos越小
                # print(f"    找到非条件帧 {idx} (模态{target_modality}) -> tpos={tpos}")
                
            if out is None:
                # print(f"    跳过帧 {idx} (模态{target_modality}) - 无输出")
                continue
                
            if "maskmem_features" not in out:
                # print(f"    跳过帧 {idx} (模态{target_modality}) - 无maskmem_features")
                continue
                
            # 获取该帧的memory特征
            feats = out["maskmem_features"]
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))  # [HW, B, C]
            
            # 空间位置编码
            maskmem_enc = out["maskmem_pos_enc"][-1]
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            
            # 时间位置编码 - 与原方法完全一致
            if tpos == 0:
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[-1]
            else:
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_modality - tpos - 1]
                
            to_cat_memory_pos_embed.append(maskmem_enc)
            found_frames.append((idx, target_modality, tpos))
        
        # print(f"  → 找到的模态帧: {found_frames}")
        # print(f"    这些帧来自同一个物理时间点的不同模态 ✅")
        # print(current_modality)
        # print(self.no_mem_embed.shape)
        # print(self.mem_dim)
        # print((1, B, self.mem_dim))
        if len(to_cat_memory) == 0:
            # print(f"  → 未找到模态memory，使用空memory")
            return {
                'curr_pos': curr_pos,
                'memory': self.no_mem_embed.expand(1, B, self.mem_dim),
                'memory_pos': self.no_mem_pos_enc.expand(1, B, self.mem_dim),
                'num_obj_ptr_tokens': 0,
            }
        
        # 拼接所有模态memory
        modal_memory = torch.cat(to_cat_memory, dim=0)
        modal_memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        
        # print(f"  → 模态memory形状: {modal_memory.shape}, pos编码形状: {modal_memory_pos_embed.shape}")
        
        return {
            'curr_pos': curr_pos,
            'memory': modal_memory,
            'memory_pos': modal_memory_pos_embed,
            'num_obj_ptr_tokens': 0,
        }
