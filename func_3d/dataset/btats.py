""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from func_3d.utils import generate_bbox

class BraTS(Dataset):
    def __init__(self, args, data_list, transform=None, mode='train', prompt='click', seed=None, variation=0, target_class='WT'):

        # 先获取原始样本列表
        self.name_list = data_list
        original_count = len(self.name_list)
        
        # 内部定义需要排除的样本，根据不同模式选择不同的排除列表
        exclude_samples = {
            'train': [],  # 训练集需要排除的样本
            'valid': [],
            'test': []    # 测试集需要排除的样本
        }
        
        # 获取当前模式下需要排除的样本列表
        current_exclude_list = exclude_samples.get(mode, [])
        
        # 从name_list中移除指定的问题样本
        if current_exclude_list:
            self.name_list = [name for name in self.name_list if name not in current_exclude_list]
            excluded_count = original_count - len(self.name_list)
            print(f"在{mode}集中排除了 {excluded_count} 个问题样本，剩余样本数量: {len(self.name_list)}")
            
        # Set the basic information of the dataset
        self.data_path = args.data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size # 这里传入 1024 即可
        self.transform = transform
        self.seed = seed
        self.variation = variation
        self.target_class = target_class # 新增的类别：WT, TC, ET

        if mode == 'train':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        
        # 直接读取预处理好的 npz 文件
        npz_path = os.path.join(self.data_path, 'preprocessed', f"{name}.npz")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"找不到预处理文件 {npz_path}，请先运行 preprocess.py。")
            
        # 加载数据 (读入内存极快)
        data = np.load(npz_path)
        imgs = data['imgs']       # shape: (4, 256, 256, Z) or depending on array orientation
        mask_array = data['mask'] # shape: (256, 256, Z)

        img_t1c = imgs[0]
        img_t1n = imgs[1]
        img_t2w = imgs[2]
        img_t2f = imgs[3]
        
        # --- category-agnostic 多类别：保留原始多标签 mask，后续构造 WT/TC/ET 三个二值通道 ---
        # 不再按 self.target_class 二值化（那是 category-specific 的旧做法）。
        # 原始标签约定：1=坏死/非增强核(NCR), 2=水肿(ED), 3/4=增强肿瘤(ET)。
        # WT=(>0); TC=isin{1,3,4}; ET=isin{3,4}（三者嵌套 WT⊇TC⊇ET）。
        # 注：transform=ToTensor() 对 float 数组不做缩放，仅 HWC→CHW，故原始标签值可安全保留。
        mask_array = mask_array.astype(np.float32)

        num_frame = img_t1c.shape[-1]
        if self.video_length is None:
            video_length = num_frame
        else:
            if num_frame < self.video_length:
                video_length = num_frame
            else:
                video_length = self.video_length
                
        if num_frame > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0

        # 时间/切片维度裁剪
        img_t1c = img_t1c[ :, :, starting_frame:starting_frame+video_length]
        img_t1n = img_t1n[ :, :, starting_frame:starting_frame+video_length]
        img_t2w = img_t2w[ :, :, starting_frame:starting_frame+video_length]
        img_t2f = img_t2f[ :, :, starting_frame:starting_frame+video_length]
        mask_array = mask_array[ :, :, starting_frame:starting_frame+video_length]
        
        # ========================================================
        # [修改点 1] 先 Center Crop 到 192x192，再 Resize 到 1024x1024
        # ========================================================
        crop_size = 192
        target_size = self.img_size # 应为 1024
        
        h, w = img_t1c.shape[0], img_t1c.shape[1]
        ch, cw = crop_size, crop_size
        y1 = max(0, (h - ch) // 2)
        y2 = y1 + ch
        x1 = max(0, (w - cw) // 2)
        x2 = x1 + cw
        
        # 验证集需要保留原始未裁剪的 Mask 用于计算最终评估指标
        if self.mode != 'train':
            orig_mask = np.transpose(mask_array.copy(), (2, 0, 1))  # (D, H, W) 原始多标签
            om = torch.from_numpy(orig_mask).float()               # (D, H, W)
            # 构造 WT/TC/ET 三个二值通道 → (3, D, H, W)
            o_wt = (om > 0).float()
            o_tc = ((om == 1) | (om == 3) | (om == 4)).float()
            o_et = ((om == 3) | (om == 4)).float()
            original_mask_tensor = torch.stack([o_wt, o_tc, o_et], dim=0)  # (3, D, H, W) 通道序 [WT,TC,ET]
        else:
            original_mask_tensor = torch.tensor([]) # 训练集不需要

        # 空间维度裁剪到 192x192
        img_t1c = img_t1c[y1:y2, x1:x2, :]
        img_t1n = img_t1n[y1:y2, x1:x2, :]
        img_t2w = img_t2w[y1:y2, x1:x2, :]
        img_t2f = img_t2f[y1:y2, x1:x2, :]
        mask_array = mask_array[y1:y2, x1:x2, :]
        
        img_numpy = np.zeros((crop_size, crop_size, video_length*4))
        mask_numpy = np.zeros((crop_size, crop_size, video_length*4))
        
        img_numpy[ :, :, 0::4] = img_t1c
        img_numpy[ :, :, 1::4] = img_t1n
        img_numpy[ :, :, 2::4] = img_t2w
        img_numpy[ :, :, 3::4] = img_t2f
        mask_numpy[ :, :, 0::4] = mask_array
        mask_numpy[ :, :, 1::4] = mask_array
        mask_numpy[ :, :, 2::4] = mask_array
        mask_numpy[ :, :, 3::4] = mask_array

        # 转为 Tensor, 此时 shape 是 (C, 192, 192)
        img_tensor = self.transform(img_numpy)
        mask_tensor = self.transform(mask_numpy)
        
        # 通过 PyTorch 的 interpolate 缩放到 1024x1024
        # 图像使用 bilinear，mask 使用 nearest 确保标签为0或1
        img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(target_size, target_size), mode='nearest').squeeze(0)

        img_tensor = img_tensor.unsqueeze(1)  # (C, 1, 1024, 1024)
        mask_tensor = mask_tensor.unsqueeze(1)  # (C, 1, 1024, 1024)  原始多标签

        # ========================================================
        # [category-agnostic 多类别] 由原始多标签构造 WT/TC/ET 三个二值通道。
        # mask_tensor 现为原始标签值(经 nearest 缩放保留整数)。每帧得到 3 个二值掩码，
        # 叠成 (C, 3, 1024, 1024)。三者嵌套：WT⊇TC⊇ET。
        # ========================================================
        wt = (mask_tensor > 0).float()
        tc = ((mask_tensor == 1) | (mask_tensor == 3) | (mask_tensor == 4)).float()
        et = ((mask_tensor == 3) | (mask_tensor == 4)).float()
        mask_tensor = torch.cat([wt, tc, et], dim=1)  # (C, 3, 1024, 1024)，通道序 [WT, TC, ET]

        # ========================================================
        # box 提示：仅当 prompt=='bbox' 时生成。每个 slice 一个紧致框，
        # 从 192-crop 坐标缩放到 256 引导坐标系；4 个模态共享同一 slice 的框，
        # 按帧交错顺序展开成 [video_length*4, 4]。全 0 表示该帧无肿瘤(不提示)。
        # ========================================================
        if str(self.prompt).lower() == 'bbox':
            scale = 256.0 / crop_size
            per_slice_box = np.zeros((video_length, 4), dtype=np.float32)
            for k in range(video_length):
                ys, xs = np.where(mask_array[:, :, k] > 0)
                if xs.size > 0:
                    bx1 = xs.min() * scale
                    bx2 = (xs.max() + 1) * scale
                    by1 = ys.min() * scale
                    by2 = (ys.max() + 1) * scale
                    per_slice_box[k] = [bx1, by1, bx2, by2]
            box_per_frame = np.repeat(per_slice_box, 4, axis=0)  # [video_length*4, 4]
            box_tensor = torch.from_numpy(box_per_frame).float()
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return {
            'image':img_tensor,
            'label': mask_tensor,
            'original_mask': original_mask_tensor, # 返回原尺寸mask
            'box': box_tensor,                      # [N,4] 256 坐标系逐帧框（bbox 模式），否则空
            'image_meta_dict': name
        }

def generate_multi_bbox_with_generate_bbox(
    binary_mask: np.ndarray,
    variation: float = 0,
    seed: int = None,
    min_area_threshold: int = 10
) -> torch.Tensor:
    """
    从二值 mask 中提取所有连通区域的 bbox。对于每个连通区域，构造对应的
    二值 mask 并调用原有的 generate_bbox 函数计算 bbox。
    """
    # 1) 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    bboxes = []
    for label in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[label]
        if area < min_area_threshold:
            continue
        
        # 构造当前连通区域的二值 mask
        component_mask = (labels == label).astype(np.uint8) * 255
        
        # 调用现有的 generate_bbox，得到 ndarray([4], dtype=float)
        bbox = generate_bbox(component_mask, variation=variation, seed=seed)
        bboxes.append(bbox)
    
    # 如果没有检测到任何连通域，返回一行 NaN
    if not bboxes:
        arr = generate_bbox(np.array(binary_mask), variation=variation, seed=seed)
        arr = np.reshape(arr,(1,4))
        # Nan to 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # 转成 torch.Tensor 并返回
        arr = torch.from_numpy(arr).float()  # dtype=torch.float32
        return arr
    else:
        arr = np.stack(bboxes).astype(np.float32)  # shape [n,4]
        # 转成 torch.Tensor 并返回
        return torch.from_numpy(arr)  # dtype=torch.float32