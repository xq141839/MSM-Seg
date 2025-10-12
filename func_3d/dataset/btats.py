""" Dataloader for the BTCV dataset
    Yunli Qi
"""
from hmac import new
import os
import numpy as np
from scipy import interpolate
from sympy import im
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from func_3d.utils import random_click, generate_bbox
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
# from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice


class BraTS(Dataset):
    def __init__(self, args, data_list, transform=None, mode='train', prompt='bbox', seed=None, variation=0):

        # 先获取原始样本列表
        self.name_list = data_list
        original_count = len(self.name_list)
            
        # Set the basic information of the dataset
        self.data_path = '/home/Qing_Xu/miccai2025/mri/datasets/brats2024_met/data'
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.seed = seed
        self.variation = variation
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
        img_path1 = os.path.join(self.data_path, name, name+'-t1c.nii.gz')
        img_path2 = os.path.join(self.data_path, name, name+'-t1n.nii.gz')
        img_path3 = os.path.join(self.data_path, name, name+'-t2w.nii.gz')
        img_path4 = os.path.join(self.data_path, name, name+'-t2f.nii.gz')
        mask_path = os.path.join(self.data_path, name, name+'-seg.nii.gz')

        img_t1c = sitk.ReadImage(img_path1)
        img_t1n = sitk.ReadImage(img_path2)
        img_t2w = sitk.ReadImage(img_path3)
        img_t2f = sitk.ReadImage(img_path4) 
        mask = sitk.ReadImage(mask_path)

        # resize to Z*256*256
        # print((img_t1c.GetSize()[0], newsize[0], newsize[1]))
        img_t1c = resize_to_size(img_t1c, (img_t1c.GetSize()[0], newsize[0], newsize[1]), sitk.sitkLinear)
        img_t1n = resize_to_size(img_t1n, (img_t1n.GetSize()[0], newsize[0], newsize[1]), sitk.sitkLinear)
        img_t2w = resize_to_size(img_t2w, (img_t2w.GetSize()[0], newsize[0], newsize[1]), sitk.sitkLinear)
        img_t2f = resize_to_size(img_t2f, (img_t2f.GetSize()[0], newsize[0], newsize[1]), sitk.sitkLinear)
        mask = resize_to_size(mask, (mask.GetSize()[0], newsize[0], newsize[1]), sitk.sitkNearestNeighbor)

        img_t1c = sitk.GetArrayFromImage(img_t1c)
        img_t1n = sitk.GetArrayFromImage(img_t1n)
        img_t2w = sitk.GetArrayFromImage(img_t2w)
        img_t2f = sitk.GetArrayFromImage(img_t2f)
        mask = sitk.GetArrayFromImage(mask)
        mask[mask > 0] = 1

        img_t1c = ZScoreNormalization(img_t1c, mask)
        img_t1n = ZScoreNormalization(img_t1n, mask)
        img_t2w = ZScoreNormalization(img_t2w, mask)
        img_t2f = ZScoreNormalization(img_t2f, mask)

        num_frame = img_t1c.shape[-1]
        if self.video_length is None:
            # video_length = int(num_frame / 4)
            video_length = num_frame
        else:
            if num_frame < self.video_length:
                video_length = num_frame
            else:
                video_length = self.video_length
            # video_length = int(num_frame / 4)
        if num_frame > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0

        # print(starting_frame, video_length, num_frame)
        # print(img_t1c.shape, mask.shape)
        # crop
        img_t1c = img_t1c[ :, :, starting_frame:starting_frame+video_length]
        img_t1n = img_t1n[ :, :, starting_frame:starting_frame+video_length]
        img_t2w = img_t2w[ :, :, starting_frame:starting_frame+video_length]
        img_t2f = img_t2f[ :, :, starting_frame:starting_frame+video_length]
        mask = mask[ :, :, starting_frame:starting_frame+video_length]
        # print(img_t1c.shape, mask.shape)
        
        img_numpy = np.zeros((newsize[0], newsize[1], video_length*4))
        mask_numpy = np.zeros((newsize[0], newsize[1], video_length*4))
        
        img_numpy[ :, :, 0::4] = img_t1c
        img_numpy[ :, :, 1::4] = img_t1n
        img_numpy[ :, :, 2::4] = img_t2w
        img_numpy[ :, :, 3::4] = img_t2f
        mask_numpy[ :, :, 0::4] = mask
        mask_numpy[ :, :, 1::4] = mask
        mask_numpy[ :, :, 2::4] = mask
        mask_numpy[ :, :, 3::4] = mask

        bbox_dict = np.zeros((video_length*4, 4))
        for i in range(video_length*4):
            bbox = generate_bbox(mask_numpy[:, :, i], variation=self.variation, seed=self.seed)
            bbox_dict[i] = bbox

        img_tensor = self.transform(img_numpy).unsqueeze(1)  # C, D, H, W
        mask_tensor = self.transform(mask_numpy).unsqueeze(1)
        bbox_dict = torch.tensor(bbox_dict, dtype=torch.float32)

        return {
            'image':img_tensor,
            'label': mask_tensor,
            'bbox': bbox_dict,
            'image_meta_dict': name
        }
    
    
def ZScoreNormalization(image, seg):
    mask = seg >= 0
    mean = image[mask].mean()
    std = image[mask].std()
    image[mask] = (image[mask] - mean) / (max(std, 1e-8))
    return image

def resize_to_size(image, new_size, interpolator=sitk.sitkLinear):
    """Resize a 3D image to a new size using SimpleITK.

    Args:
        image (sitk.Image): The input 3D image.
        new_size (tuple): A tuple of three integers specifying the new size (width, height, depth).
        interpolator (int, optional): The interpolation method. Defaults to sitk.sitkLinear.

    Returns:
        sitk.Image: The resized 3D image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    new_size = [int(sz) for sz in new_size]
    new_spacing = [
        original_spacing[i] * (original_size[i] / new_size[i]) for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    
    resized_image = resample.Execute(image)
    
    return resized_image


