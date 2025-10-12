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
from func_3d.utils import get_network, set_log_dir, create_logger
from torch.utils.data import DataLoader
from func_3d.dataset.btats import BraTS
from tqdm import tqdm
import nibabel as nib
import numpy as np
import imageio
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as pytorch_transforms
import json
import numpy as np
from scipy.ndimage import binary_fill_holes

def main():
    args = cfg.parse_args()
    args.sam_ckpt = 'checkpoints/sam2_hiera_small.pt'
    args.sam_config = 'sam2_hiera_s'
    args.data_path = 'preprocessed'
    test_checkpoint = 'outputs/epoch_6.pth'
    test_gz_dir = 'test_results/gz'
    test_visual_dir = 'test_results/visual'
    
    GPUdevice = torch.device('cuda', args.gpu_device)
    
    # 初始化网络
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)

    # 加载预训练模型
    if os.path.exists(test_checkpoint):
        print(f"Loading checkpoint from: {test_checkpoint}")
        checkpoint = torch.load(test_checkpoint, map_location=GPUdevice)
        if 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'], strict=True)
        else:
            net.load_state_dict(checkpoint, strict=True)
        print("Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {test_checkpoint}")
    
    # 设置混合精度
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    jsonfile1 = f'datasets/brats2024_met/data_split.json'
    with open(jsonfile1, 'r') as f:
        df = json.load(f)

    train_files = df['test']
    
    # 获取测试数据加载器
    brats_test_dataset = BraTS(args, train_files, transform = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ]), mode = 'test', prompt=args.prompt)
    nice_test_loader = DataLoader(brats_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    # 创建输出目录
    os.makedirs(test_gz_dir, exist_ok=True)
    os.makedirs(test_visual_dir, exist_ok=True)
    
    print("Starting testing...")
    
    # 设置为评估模式
    net.eval()
    
    # 开始测试
    time_start = time.time()
    
    with torch.no_grad():
        n_val = len(nice_test_loader)  # the number of batch
                                                 
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in nice_test_loader:
                imgs_tensor = Variable(pack['image'].cuda()).squeeze(0)
                imgs_tensor = torch.tensor(imgs_tensor, dtype=torch.float32)
                image_id = pack['image_meta_dict']
                bbox_prompt = Variable(pack['bbox']).squeeze(0)

                with torch.cuda.amp.autocast():
                    video_segments, _ = net(imgs_tensor, prompt=bbox_prompt)
                    
                    # Apply softmax and argmax to get predicted class per pixel
                    pred_labels = torch.sigmoid(video_segments)  # shape: [slices, classes, W, H]
                    pred_labels[pred_labels >= 0.5] = 1
                    pred_labels[pred_labels < 0.5] = 0
                    # 只保留第四个模态的结果（即每4个slice中的最后一个）

                    pred_labels = pred_labels[3::4]  # 假设顺序为1,2,3,4,1,2,3,4,...

                    # Resize each slice to 256x256
                    pred_labels = pred_labels.squeeze(1).to(torch.uint8).cpu()  # shape: [slices, 256, 256]
      
                    pred_labels_np = pred_labels.numpy()  # [slices, 256, 256]
                    pred_labels_np = pred_labels_np.transpose(1, 2, 0)  # [256, 256, slices]

                    # Save as nii.gz
                    save_nii_path = os.path.join(test_gz_dir, f"{image_id[0]}.nii.gz")
                    nii_img = nib.Nifti1Image(pred_labels_np, affine=np.eye(4))
                    nib.save(nii_img, save_nii_path)

                    # Save each slice as grayscale PNG
                    visual_dir = os.path.join(test_visual_dir, str(image_id[0]))
                    os.makedirs(visual_dir, exist_ok=True)
                    for idx in range(pred_labels_np.shape[2]):
                        slice_img = pred_labels_np[:, :, idx]
                        # Normalize to 0-255 for visualization
                        # Assign fixed grayscale values for each class (e.g., class 0: 0, class 1: 85, class 2: 170, class 3: 255)
                        num_classes = int(pred_labels_np.max()) + 1
                        class_values = np.linspace(0, 255, num_classes, dtype=np.uint8)
                        slice_img_vis = class_values[slice_img.astype(np.int32)]
                        imageio.imwrite(os.path.join(visual_dir, f"{idx:03d}.png"), slice_img_vis)

                pbar.update()


        
    
    time_end = time.time()
    test_time = time_end - time_start

if __name__ == '__main__':
    main()
