""" function for training and validation in one epoch
"""

import os
from cycler import V
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
from loss import *
import numpy as np
import cv2
from torch.autograd import Variable
import csv
from datetime import datetime
import cfg
from conf import settings
from func_3d.utils import eval_seg
import pandas as pd
import time  # 添加到文件顶部的imports
args = cfg.parse_args()


def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch):

    mask_loss = DiceCELoss(include_background=True, to_onehot_y=False, softmax=False, sigmoid=True) #lambda_dice=0.5, lambda_ce=5
    # mask_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, sigmoid=False)

    # train mode
    net.train()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    # self_prompt_loss_fn = sp_loss
    self_prompt_weight = 0.5
    lossfunc = mask_loss

    total_prompt_loss_list = []
    main_loss_list = []
    prompt_loss_list = []
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs_tensor = Variable(pack['image'].cuda()).squeeze(0)
            imgs_tensor = torch.tensor(imgs_tensor, dtype=torch.float32)
            mask_gt = Variable(pack['label'].cuda()).squeeze(0)
            bbox_prompt = Variable(pack['bbox']).squeeze(0)
            if len(torch.unique(mask_gt)) < 2:
                pbar.update()
                continue
            
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                video_segments, self_prompts = net(imgs_tensor, prompt=bbox_prompt)
                main_loss = lossfunc(video_segments, mask_gt)
                sp_gt = F.interpolate(mask_gt, scale_factor=0.25, mode='nearest')
                prompt_loss = lossfunc(self_prompts, sp_gt)
                total_prompt_loss = main_loss + self_prompt_weight * prompt_loss
                total_prompt_loss.backward()
                optimizer.step()

            total_prompt_loss_list.append(total_prompt_loss.item())
            main_loss_list.append(main_loss.item())
            prompt_loss_list.append(prompt_loss.item())
            pbar.update()
                

    return np.mean(total_prompt_loss_list), np.mean(main_loss_list), np.mean(prompt_loss_list)
