""" function for training and validation in one epoch
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm
import numpy as np
import cv2
from torch.autograd import Variable
import csv
from datetime import datetime
import cfg
from conf import settings
from func_3d.utils import eval_seg, build_prompt_dict
import pandas as pd
import time
args = cfg.parse_args()


def train_sam(args, net: nn.Module, optimizer, scheduler, train_loader, epoch):
    # [修改] 与论文一致：modal/fusion 损失用 Dice + Focal（脑肿瘤前景占比极小，Focal 比普通 CE 更抗背景主导）
    mask_loss = DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True,
                              lambda_dice=1.0, lambda_focal=1.0)
    
    # [改进] 提示监督改用 Dice+Focal。原来的纯 BCE 在前景占比极小时，模型只要把 self-prompt
    # 预测成"几乎全背景"就能把 loss 压到很低（这正是 prompt-loss 一上来就卡在 ~0.12 不动的原因），
    # 导致 tumor guidance 形同虚设。Dice 项强制与真实肿瘤区域产生重叠，提示才会真正去定位肿瘤。
    prompt_loss_fn = DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True,
                                   lambda_dice=1.0, lambda_focal=1.0)

    # train mode
    net.train()

    prompt = args.prompt
    prompt_freq = args.prompt_freq
    use_box_prompt = str(prompt).lower() == 'bbox'  # 是否启用 box 提示模式训练

    # [改进] 与论文 Eq.10 一致，L_prompt 同权 (=1.0)。原来 0.5 偏低，加上纯 BCE 容易让提示塌成空白。
    self_prompt_weight = 1.0

    total_prompt_loss_list = []
    main_loss_list = []
    prompt_loss_list = []
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs_tensor = pack['image'].cuda().squeeze(0).clone().detach().float()
            
            # btats.py 中 ToTensor 已经把时间维度转换到第 0 维，所以挤压 batch 维度后直接就是 [L*4, 1, H, W]
            # 与模型输出 video_segments 的维度完美对齐，切勿再用 transpose 转置
            mask_gt = pack['label'].cuda().squeeze(0)
            
            # [DDP 关键修复]
            is_empty = len(torch.unique(mask_gt)) < 2
            loss_weight = 0.0 if is_empty else 1.0
            
            optimizer.zero_grad()

            # [新增] box 提示模式：从 GT 生成的逐帧框构造 prompt 字典；自动模式则 prompt=None
            prompt_dict = None
            if use_box_prompt and ('box' in pack):
                box_t = pack['box'].squeeze(0)  # [N,4]
                prompt_dict = build_prompt_dict(box_t, imgs_tensor.device)

            # [修复] 单一 bf16 autocast，只包裹前向 + 计算 loss；不再嵌套 fp16 的 torch.cuda.amp.autocast()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # video_segments: 各(切片,模态)预测 Ŷ_{t,m}；fused_pred: Eq.9 融合后的切片预测 Ŷ_t
                video_segments, self_prompts, fused_pred = net(imgs_tensor, prompt=prompt_dict)

                # --- 将 GT 适配到 SAM2 的原生 256 尺度 ---
                if mask_gt.shape[-1] != 256:
                    mask_gt_256 = F.interpolate(mask_gt, size=(256, 256), mode='nearest')
                else:
                    mask_gt_256 = mask_gt

                # =========================================================
                # [修改] 按论文 Eq.10 组织损失：L = L_prompt + (1/M)Σ_m L_modal^m + L_fusion
                #   · L_modal：每个模态预测各自与 GT 的 Dice+Focal，等权 (1/M)
                #   · L_fusion：Eq.9 自适应融合后的切片预测 Ŷ_t 与 GT 的 Dice+Focal
                #   原来的“固定阶梯权重 + 取第4模态当融合”被替换为真正的自适应融合
                # =========================================================
                num_mod = 4  # 与数据交错(4 模态)一致
                modal_loss = 0.0
                for i in range(num_mod):
                    modal_loss += mask_loss(video_segments[i::4], mask_gt_256[i::4])
                modal_loss = modal_loss / num_mod

                # 每个切片只有一份 GT（4 个模态共享），取模态 0 的帧即为逐切片 GT [L,1,H,W]
                gt_slice = mask_gt_256[0::4]
                fusion_loss = mask_loss(fused_pred, gt_slice)

                # Self prompt loss：self_prompts 为单通道“整肿瘤”引导(category-agnostic)，
                # 与 GT 的 WT 通道(通道0)对齐。两者均在 256 尺度、逐帧。
                prompt_loss = prompt_loss_fn(self_prompts, mask_gt_256[:, 0:1].float())

                main_loss = modal_loss + fusion_loss  # 用于日志（掩码相关总损失）
                total_prompt_loss = (self_prompt_weight * prompt_loss + modal_loss + fusion_loss) * loss_weight

            # [修复] backward / 梯度裁剪 / step / scheduler 全部放在 autocast 之外
            # bf16 autocast 不需要 GradScaler（bf16 指数范围与 fp32 一致，不会下溢）
            total_prompt_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            if not is_empty:
                total_prompt_loss_list.append(total_prompt_loss.item())
                main_loss_list.append(main_loss.item())
                prompt_loss_list.append(prompt_loss.item())
                
            pbar.update()
            
    l1 = float(np.mean(total_prompt_loss_list)) if len(total_prompt_loss_list) > 0 else 0.0
    l2 = float(np.mean(main_loss_list)) if len(main_loss_list) > 0 else 0.0
    l3 = float(np.mean(prompt_loss_list)) if len(prompt_loss_list) > 0 else 0.0
    
    return l1, l2, l3


def validation_sam(args, net: nn.Module, val_loader, epoch):
    """
    测试集验证模块：将 2D 的切片堆叠还原为 3D 尺寸计算 Case-Level 的指标 (Dice 和 HD95)
    加入 DDP 分块逻辑，验证速度起飞！
    """
    net.eval()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")

    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    with torch.no_grad():
        if rank == 0:
            loader_tqdm = tqdm(val_loader, desc=f'Val Epoch {epoch}', unit='img')
        else:
            loader_tqdm = val_loader
            
        for i, pack in enumerate(loader_tqdm):
            if i % world_size != rank:
                continue
                
            imgs_tensor = pack['image'].cuda().squeeze(0).clone().detach().float()
            
            # [修改点 4] 读取未裁剪前的 256x256 原始 Mask (shape: [1, L, 256, 256])
            original_gt_vol = pack['original_mask'].cuda().squeeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 用 Eq.9 融合后的逐切片预测 Ŷ_t 作为验证输出
                _, _, fused_pred = net(imgs_tensor, prompt=None)

            # fused_pred 已是逐切片预测 [L, 1, 256, 256]
            pred_final = (torch.sigmoid(fused_pred) > 0.5).float()

            # --- 将 256 缩放回 192 ---
            pred_final_192 = F.interpolate(pred_final, size=(192, 192), mode='nearest')
            
            # 动态计算 Padding，将 192 恢复到 256 (或原尺寸)
            target_h, target_w = original_gt_vol.shape[-2], original_gt_vol.shape[-1]
            pred_h, pred_w = pred_final_192.shape[-2], pred_final_192.shape[-1]
            pad_left = (target_w - pred_w) // 2
            pad_right = target_w - pred_w - pad_left
            pad_top = (target_h - pred_h) // 2
            pad_bottom = target_h - pred_h - pad_top
            
            # padding 顺序: (left, right, top, bottom)
            pred_final_padded = F.pad(pred_final_192, (pad_left, pad_right, pad_top, pad_bottom)) # -> [L, 1, 256, 256]
            
            # 还原为 Case-Level 的 3D Volume
            pred_vol = pred_final_padded.transpose(0, 1).unsqueeze(0) # -> [1, 1, L, 256, 256]
            gt_vol = original_gt_vol.unsqueeze(0) # -> [1, 1, L, 256, 256]

            dice_metric(y_pred=pred_vol, y=gt_vol)
            
            if gt_vol.sum() > 0 and pred_vol.sum() > 0:
                hd95_metric(y_pred=pred_vol, y=gt_vol)

    # 跨显卡同步指标合并 (Cross-GPU Metric Aggregation)
    try:
        local_dice = dice_metric.aggregate().item()
    except:
        local_dice = 0.0
        
    try:
        local_hd95 = hd95_metric.aggregate().item()
    except:
        local_hd95 = 0.0

    if is_distributed:
        metrics_tensor = torch.tensor([local_dice, local_hd95], dtype=torch.float32, device='cuda')
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        mean_dice = (metrics_tensor[0] / world_size).item()
        mean_hd95 = (metrics_tensor[1] / world_size).item()
    else:
        mean_dice = local_dice
        mean_hd95 = local_hd95

    dice_metric.reset()
    hd95_metric.reset()

    return mean_dice, mean_hd95