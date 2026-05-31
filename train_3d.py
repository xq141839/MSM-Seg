import os
import time
import math

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

# DDP 分布式训练相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import cfg
import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from torch.utils.data import DataLoader
from func_3d.dataset.btats import BraTS
import torchvision.transforms as pytorch_transforms
import wandb
import json
import datetime

def main():
    args = cfg.parse_args()
    
    # ================== DDP 初始化 ==================
    # 检查是否通过 torchrun 等分布式方式启动
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        # 【修复 超时死锁】: 将超时时间设置为 7200 秒 (2小时)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        local_rank = 0
        global_rank = 0
        device = torch.device('cuda')
    # ================================================

    target_cls = getattr(args, 'target_class', 'WT')
    
    # 只有主进程 (Rank 0) 负责初始化 wandb 和 logger
    if global_rank == 0:
        wandb.init(
            project="TMM-SAM2-Brain-Tumor-Segmentation",
            config={
                "architecture": "MSM-Seg",
                "video_length": args.video_length,
                "image_size": args.image_size,
                "sam_config": args.sam_config,
                "target_class": target_cls,
                "epoch": settings.EPOCH,
            }
        )    
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)
    else:
        logger = None

    # 初始化模型并送到对应设备的 GPU
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution=args.distributed)
    # [修复] 权重保持 fp32 (master weights)，混合精度只在前向用 autocast 实现。
    # 若像之前那样 net.to(bfloat16)，AdamW 的小更新量会被 bf16 的低尾数精度四舍五入吞掉，
    # 表现为几个 epoch 后 loss 不再下降。
    net = net.to(device=device)
    
    # 包装为 DDP 模型
    if is_distributed:
        # SAM2 模型中有部分层可能在不使用 prompt 的时候不参与前向传播，需开启 find_unused_parameters=True
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    # [修复] 不再全局 __enter__ 一个永不退出的 bf16 autocast；混合精度改为在 train_sam 内
    # 用 with torch.autocast(...) 精确包裹前向+loss，backward/step 在其外执行。

    if global_rank == 0:
        print(torch.version.cuda)
        print(torch.cuda.get_device_name(0))
        
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ================= 适配 nnUNet V2 的动态数据读取逻辑 =================
    imagesTr_dir = os.path.join(args.data_path, 'imagesTr')
    train_files = []
    
    if os.path.exists(imagesTr_dir):
        for file_name in os.listdir(imagesTr_dir):
            if file_name.endswith('_0000.nii.gz'):
                base_name = file_name.replace('_0000.nii.gz', '')
                train_files.append(base_name)
    else:
        raise FileNotFoundError(f"找不到训练文件夹: {imagesTr_dir}，请检查数据路径是否正确。")
        
    if global_rank == 0:
        print(f"成功扫描到 {len(train_files)} 个训练样本！")
    
    imagesTs_dir = os.path.join(args.data_path, 'imagesTs')
    test_files = []
    if os.path.exists(imagesTs_dir):
        for file_name in os.listdir(imagesTs_dir):
            if file_name.endswith('_0000.nii.gz'):
                base_name = file_name.replace('_0000.nii.gz', '')
                test_files.append(base_name)
    
    if global_rank == 0:
        if len(test_files) > 0:
            print(f"成功扫描到 {len(test_files)} 个验证/测试样本！")
        else:
            print("警告: 未找到验证/测试集，请确认 imagesTs 文件夹是否存在数据。")
    # ======================================================================

    dataset_kwargs = {'prompt': args.prompt}
    if hasattr(args, 'target_class'):
        dataset_kwargs['target_class'] = args.target_class

    brats_train_dataset = BraTS(args, train_files, transform=pytorch_transforms.Compose([pytorch_transforms.ToTensor()]), mode='train', **dataset_kwargs)
    
    # DDP 分布式数据采样器
    if is_distributed:
        train_sampler = DistributedSampler(brats_train_dataset, shuffle=True)
        nice_train_loader = DataLoader(brats_train_dataset, batch_size=1, sampler=train_sampler, pin_memory=True)
    else:
        train_sampler = None
        nice_train_loader = DataLoader(brats_train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    if len(test_files) > 0:
        brats_test_dataset = BraTS(args, test_files, transform=pytorch_transforms.Compose([pytorch_transforms.ToTensor()]), mode='test', **dataset_kwargs)
        nice_test_loader = DataLoader(brats_test_dataset, batch_size=1, shuffle=False, pin_memory=False)
    else:
        nice_test_loader = None

    checkpoint_path = os.path.join("outputs")
    if global_rank == 0 and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # [新增] warmup + cosine 学习率调度（按 step 更新）。
    # 全量微调 SAM2 时，恒定 lr 容易让编码器前期学得太猛、后期停滞；warmup 起步更稳，cosine 收尾更细。
    total_steps = max(1, settings.EPOCH * len(nice_train_loader))
    warmup_steps = max(1, int(0.03 * total_steps))
    def _lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    for epoch in range(settings.EPOCH):
        
        # 多卡训练时，必须在每轮开始前为 sampler 设置 epoch，保证各显卡数据打乱且不重复
        if is_distributed:
            train_sampler.set_epoch(epoch)

        net.train()
        time_start = time.time()
        
        # 调用训练函数
        loss, main_loss, prompt_loss = function.train_sam(args, net, optimizer, scheduler, nice_train_loader, epoch)
        
        # 训练日志只在主进程打印
        if global_rank == 0:
            logger.info(f'[{target_cls}] Train loss: {loss}, mask loss: {main_loss}, prompt_loss: {prompt_loss} || @ epoch {epoch}.')
            time_end = time.time()
            print('time_for_training ', time_end - time_start)
            wandb.log({"total-loss": loss, "main-loss": main_loss, "prompt-loss": prompt_loss, "epoch": epoch})

        # # ================== 验证阶段 ==================
        # if nice_test_loader is not None and (epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1):
        #     eval_net = net.module if is_distributed else net
            
        #     # 【修复掉线 & 提速】：让所有显卡一起进入验证函数，保持多卡通信始终活跃！
        #     mean_dice, mean_hd95 = function.validation_sam(args, eval_net, nice_test_loader, epoch)
            
        #     # 但只有主卡负责在终端打印分数和写入 Wandb，防止重复记录
        #     if global_rank == 0:
        #         logger.info(f'Validation || Epoch {epoch} - Case-level Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}')
        #         wandb.log({"val-dice": mean_dice, "val-hd95": mean_hd95, "epoch": epoch})
        
        # if is_distributed:
        #     dist.barrier()
        # # ==============================================

        # 模型权重只需保存一次
        if global_rank == 0:
            model_to_save = net.module.state_dict() if is_distributed else net.state_dict()
            torch.save({'model': model_to_save}, os.path.join(checkpoint_path, f'epoch_{epoch}_{target_cls}.pth'))

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()