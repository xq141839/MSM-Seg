# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2_train.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


import logging
import torch

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return

    # 1. 从磁盘加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = checkpoint.get("model", checkpoint)

    # 2. 拿到模型当前的 state_dict
    model_state = model.state_dict()

    # 3. 剔除 shape 不匹配的参数
    mismatched = []
    for name in list(sd.keys()):
        if name in model_state:
            if sd[name].shape != model_state[name].shape:
                logging.warning(
                    f"跳过参数 {name}：checkpoint shape={tuple(sd[name].shape)} ≠ model shape={tuple(model_state[name].shape)}"
                )
                mismatched.append(name)
                sd.pop(name)
        else:
            # checkpoint 里有，但模型里没有，也一并保留给 strict=False 处理
            logging.debug(f"checkpoint 中有模型里不存在的参数 {name}，后续会由 strict=False 忽略")

    # 4. 加载剩余的参数
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    # 5. 从 missing_keys 里过滤掉那些我们故意剔除的 mismatched
    missing_keys = [k for k in missing_keys if k not in mismatched]

    # # 6. 报错检查
    # if missing_keys:
    #     logging.error(f"以下参数在 checkpoint 中缺失：{missing_keys}")
    #     raise RuntimeError(f"Missing keys: {missing_keys}")
    # if unexpected_keys:
    #     logging.error(f"以下参数在模型中不存在：{unexpected_keys}")
    #     raise RuntimeError(f"Unexpected keys: {unexpected_keys}")

    logging.info("Loaded checkpoint successfully")
