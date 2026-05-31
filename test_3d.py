#!/usr/bin/env python3

""" test network using pytorch
    Based on train_3d.py structure
"""

import os
import time
import torch
import argparse

import cfg
import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger, build_prompt_dict
from torch.utils.data import DataLoader
from func_3d.dataset.btats import BraTS
from tqdm import tqdm
import nibabel as nib
import numpy as np
import imageio
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as pytorch_transforms
import gc

# --- 引入 MONAI 验证指标 ---
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def main():
    args = cfg.parse_args()
    
    # --- 目录配置 ---
    test_checkpoint = getattr(args, 'sam_ckpt', 'outputs/epoch_9_WT.pth') # 默认读取第9个epoch
    test_gz_dir = 'test_results/gz'
    test_visual_dir = 'test_results/visual'
    
    # 获取测试所用的 GPU 
    GPUdevice = torch.device('cuda', args.gpu_device)
    print(f"[*] Testing on device: {GPUdevice}")
    
    # 初始化网络
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=False)
    # [修复] 与训练一致：权重 fp32，推理时用 bf16 autocast。checkpoint 现在保存的就是 fp32 权重。
    net = net.to(device=GPUdevice)

    # 加载预训练模型
    if os.path.exists(test_checkpoint):
        print(f"[*] Loading checkpoint from: {test_checkpoint}")
        checkpoint = torch.load(test_checkpoint, map_location=GPUdevice)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        net.load_state_dict(new_state_dict, strict=True)
        print("[*] Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {test_checkpoint}")
    
    # [修复] 不再全局 __enter__ autocast；推理在前向处用 with torch.autocast(...) 包裹
    
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ================= 适配 nnUNet V2 的动态数据读取逻辑 =================
    imagesTs_dir = os.path.join(args.data_path, 'imagesTs')
    test_files = []
    
    if os.path.exists(imagesTs_dir):
        for file_name in os.listdir(imagesTs_dir):
            if file_name.endswith('_0000.nii.gz'):
                base_name = file_name.replace('_0000.nii.gz', '')
                test_files.append(base_name)
    else:
        print(f"[*] 警告: 找不到测试文件夹 {imagesTs_dir}，尝试回退...")
        
    if len(test_files) > 0:
        print(f"[*] 成功扫描到 {len(test_files)} 个验证/测试样本！")
    else:
        print("[*] 警告: 未找到验证/测试集，请确认 imagesTs 文件夹是否存在匹配的数据。")
    # ======================================================================
    
    # 提取目标分类（WT, TC, ET）
    target_cls = getattr(args, 'target_class', 'WT')
    
    # 初始化 dataset
    brats_test_dataset = BraTS(args, test_files, transform=pytorch_transforms.Compose([pytorch_transforms.ToTensor()]), mode='test', prompt=args.prompt, target_class=target_cls)
    nice_test_loader = DataLoader(brats_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建输出目录
    os.makedirs(test_gz_dir, exist_ok=True)
    os.makedirs(test_visual_dir, exist_ok=True)
    
    # --- 初始化验证指标：WT/TC/ET 各一套，按二分类分别评估 ---
    SEG_CLASSES = ['WT', 'TC', 'ET']   # 通道序与数据集/解码器一致
    dice_metrics = {c: DiceMetric(include_background=True, reduction="mean") for c in SEG_CLASSES}
    hd95_metrics = {c: HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean") for c in SEG_CLASSES}
    
    print(f"[*] Starting testing... Target Class: {target_cls}")
    
    # 设置为评估模式
    net.eval()
    time_start = time.time()
    
    with torch.no_grad():
        n_val = len(nice_test_loader)
        with tqdm(total=n_val, desc='Testing round', unit='case') as pbar:
            for pack in nice_test_loader:
                imgs_tensor = pack['image'].to(GPUdevice).squeeze(0).float()
                original_gt_vol = pack['original_mask'].to(GPUdevice).squeeze(0)  # shape: [1, L, H, W]

                # box 提示模式：取出该 case 的逐帧框 [N,4]（256 坐标系）；自动模式为 None
                use_box_prompt = str(getattr(args, 'prompt', 'None')).lower() == 'bbox'
                case_box = None
                if use_box_prompt and ('box' in pack):
                    case_box = pack['box'].squeeze(0)  # [N,4]
                    if case_box.numel() == 0:
                        case_box = None

                image_id = pack['image_meta_dict']
                case_name = image_id[0] if isinstance(image_id, list) else image_id

                # =======================================================
                # 【显存终极优化】: 重叠滑动窗口推理 (Sliding Window Inference)
                # =======================================================
                L = imgs_tensor.shape[0] // 4
                chunk_slices = 12  # 每次最多送入 12 个 Slice (48帧)，显存极度安全
                overlap_slices = 6 # 为新预测提供前 6 个 Slice 的时序上下文
                
                all_preds = []
                start_slice = 0
                
                while start_slice < L:
                    # 确定当前窗口的实际提取范围
                    if start_slice == 0:
                        actual_start = 0
                        valid_start_local = 0
                    else:
                        actual_start = start_slice - overlap_slices
                        valid_start_local = overlap_slices
                        
                    actual_end = min(actual_start + chunk_slices, L)
                    
                    # 截取当前窗口的所有帧 (包含了 4 个模态)
                    chunk_tensor = imgs_tensor[actual_start * 4 : actual_end * 4]

                    # 当前窗口对应的逐帧框（窗口内重新从 0 计帧，与 net 的 frame_idx 对齐）
                    chunk_prompt = None
                    if case_box is not None:
                        chunk_box = case_box[actual_start * 4 : actual_end * 4]
                        chunk_prompt = build_prompt_dict(chunk_box, GPUdevice)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # 用 Eq.9 融合后的逐切片预测 Ŷ_t 作为最终输出
                        _, _, chunk_fused = net(chunk_tensor, prompt=chunk_prompt)
                        chunk_pred_final = (torch.sigmoid(chunk_fused) > 0.5).float()  # [当前窗口Slice数,1,H,W]

                    # 截除重叠的上下文部分，只保留全新的有效预测，并立刻移动到 CPU 释放显卡内存
                    valid_pred = chunk_pred_final[valid_start_local:].cpu()
                    all_preds.append(valid_pred)
                    
                    # 下一个窗口的起点就是当前窗口的终点
                    start_slice = actual_end
                    
                    # 清理当前窗口释放显存
                    del chunk_tensor, chunk_fused, chunk_pred_final
                    torch.cuda.empty_cache()
                    
                # 将所有局部预测拼接成完整的 3D 序列，并放回 GPU 准备后续的 Pad
                pred_labels = torch.cat(all_preds, dim=0).to(GPUdevice)
                # =======================================================
                
                # 严格按照验证阶段的逻辑，先回到192，再根据原始Mask Padding
                pred_final_192 = F.interpolate(pred_labels, size=(192, 192), mode='nearest')
                
                # 动态计算 Padding，将 192 恢复到 256 (或原尺寸)
                target_h, target_w = original_gt_vol.shape[-2], original_gt_vol.shape[-1]
                pred_h, pred_w = pred_final_192.shape[-2], pred_final_192.shape[-1]
                pad_left = (target_w - pred_w) // 2
                pad_right = target_w - pred_w - pad_left
                pad_top = (target_h - pred_h) // 2
                pad_bottom = target_h - pred_h - pad_top
                
                pred_final_padded = F.pad(pred_final_192, (pad_left, pad_right, pad_top, pad_bottom))

                # 计算当前 Case 的 Dice 和 HD95 指标（WT/TC/ET 分别按二分类计算）
                pred_vol = pred_final_padded.transpose(0, 1).unsqueeze(0)  # [1, 3, D, 256, 256]
                gt_vol = original_gt_vol.unsqueeze(0)                      # [1, 3, D, H, W]

                case_dices = {}
                for ci, c in enumerate(SEG_CLASSES):
                    p = pred_vol[:, ci:ci+1]
                    g = gt_vol[:, ci:ci+1]
                    case_dices[c] = dice_metrics[c](y_pred=p, y=g).item()
                    if g.sum() > 0 and p.sum() > 0:
                        hd95_metrics[c](y_pred=p, y=g)
                case_avg_dice = sum(case_dices.values()) / len(SEG_CLASSES)

                # 在进度条后面追加显示当前病例各类别分数
                pbar.set_postfix({
                    "Case": case_name,
                    "WT": f"{case_dices['WT']:.3f}",
                    "TC": f"{case_dices['TC']:.3f}",
                    "ET": f"{case_dices['ET']:.3f}",
                })

                # 合成可视化标签图：3 个二值通道 [WT,TC,ET] → 单标签图(WT→1, TC→2, ET→3)，
                # 因嵌套(WT⊇TC⊇ET)，按序覆盖使内层标签优先。
                pf = pred_final_padded.to(torch.uint8).cpu().numpy()       # [D, 3, 256, 256]
                label_map = np.zeros((pf.shape[0],) + pf.shape[2:], dtype=np.uint8)  # [D, 256, 256]
                label_map[pf[:, 0] > 0] = 1   # WT
                label_map[pf[:, 1] > 0] = 2   # TC
                label_map[pf[:, 2] > 0] = 3   # ET
                pred_labels_np = label_map.transpose(1, 2, 0)             # [256, 256, D]

                # Save as nii.gz
                save_nii_path = os.path.join(test_gz_dir, f"{case_name}.nii.gz")
                nii_img = nib.Nifti1Image(pred_labels_np, affine=np.eye(4))
                nib.save(nii_img, save_nii_path)

                # Save each slice as grayscale PNG for visualization
                visual_dir = os.path.join(test_visual_dir, str(case_name))
                os.makedirs(visual_dir, exist_ok=True)

                class_values = np.linspace(0, 255, 4, dtype=np.uint8)    # 0/1/2/3 → 灰度
                for idx in range(pred_labels_np.shape[2]):
                    slice_img = pred_labels_np[:, :, idx]
                    if slice_img.max() > 0:
                        slice_img_vis = class_values[slice_img.astype(np.int32)]
                        imageio.imwrite(os.path.join(visual_dir, f"{idx:03d}.png"), slice_img_vis)

                # 清理完整 Case
                del imgs_tensor, original_gt_vol, pred_labels, pred_final_192, pred_final_padded, pred_vol, gt_vol
                gc.collect()
                torch.cuda.empty_cache()

                pbar.update()

    # 汇总各类别(WT/TC/ET)平均指标
    dice_per_class, hd95_per_class = {}, {}
    for c in SEG_CLASSES:
        try:
            dice_per_class[c] = dice_metrics[c].aggregate().item()
        except Exception:
            dice_per_class[c] = 0.0
        try:
            hd95_per_class[c] = hd95_metrics[c].aggregate().item()
        except Exception:
            hd95_per_class[c] = 0.0

    mean_dice = sum(dice_per_class.values()) / len(SEG_CLASSES)
    mean_hd95 = sum(hd95_per_class.values()) / len(SEG_CLASSES)

    time_end = time.time()

    print("\n" + "="*50)
    print(f"[*] Testing finished! Total time: {(time_end - time_start) / 60:.2f} mins")
    print(f"[*] Dice  -> WT: {dice_per_class['WT']:.4f} | TC: {dice_per_class['TC']:.4f} | ET: {dice_per_class['ET']:.4f} | Avg: {mean_dice:.4f}")
    print(f"[*] HD95  -> WT: {hd95_per_class['WT']:.4f} | TC: {hd95_per_class['TC']:.4f} | ET: {hd95_per_class['ET']:.4f} | Avg: {mean_hd95:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()