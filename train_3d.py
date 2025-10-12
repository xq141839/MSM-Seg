import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from torch.utils.data import DataLoader
from func_3d.dataset.btats import BraTS
import torchvision.transforms as pytorch_transforms
import wandb
import json

def main():

    args = cfg.parse_args()
    
    wandb.init(
    # set the wandb project where this run will be logged
        project="MSMSeg",
        # track hyperparameters and run metadata
        config={
            "architecture": "MSMSeg",
            "video_length": args.video_length,
            "image_size": args.image_size,
            "sam_config": args.sam_config,
            "epoch": settings.EPOCH,
        }
    )    
    
    GPUdevice = torch.device('cuda', args.gpu_device)
    device_ids = range(torch.cuda.device_count())
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net = net.to(dtype=torch.bfloat16)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999))
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()    
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    jsonfile1 = f'datasets/brats2024_met/data_split.json'
    with open(jsonfile1, 'r') as f:
        df = json.load(f)

    train_files = df['train']

    brats_train_dataset = BraTS(args, train_files, transform = pytorch_transforms.Compose([pytorch_transforms.ToTensor(), ]), mode = 'train', prompt=args.prompt)
    nice_train_loader = DataLoader(brats_train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join("outputs")
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path)

    for epoch in range(settings.EPOCH):

        net.train()
        time_start = time.time()
        loss, main_loss, prompt_loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch)
        logger.info(f'Train loss: {loss}, mask loss: {main_loss}, prompt_loss: {prompt_loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        
        wandb.log({"total-loss": loss, "main-loss": main_loss, "prompt-loss": prompt_loss})

        # net.eval()
        # # if (epoch % args.val_freq == 0 and epoch != 0) or epoch == settings.EPOCH-1:
        # loss, dice1, dice2, dice3, dice_b, dice_p, main_loss, self_prompt_loss, memory_loss = function.validation_sam(args, nice_test_loader, epoch, net)

        # logger.info(f'Dice C1: {dice1:.4f}, C2: {dice2:.4f}, C3: {dice3:.4f}, CB: {dice_b:.4f}, CP: {dice_p:.4f} || @ epoch {epoch}.')
        # logger.info(f'Train loss: {loss}, mask loss: {main_loss}, self_prompt_loss: {self_prompt_loss}, memory_loss: {memory_loss} || @ epoch {epoch}.')

        torch.save({'model': net.state_dict()}, os.path.join(checkpoint_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    main()