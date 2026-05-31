import os
import re
import glob
import argparse

# =====================================================================
# 训练/测试入口（category-agnostic 多类别版本）
#   现在一次训练/测试 **同时** 输出 WT/TC/ET 三个区域(各为二分类)，
#   只需一个“整肿瘤”框提示，符合论文 category-agnostic 设计。
#   -target_class 不再决定分割哪个类别（三类始终联合训练），仅作为
#   实验名/权重文件名的标签，便于区分不同运行；测试会分别报告 WT/TC/ET 三类指标。
#   python run.py                  -> 训练并测试（标签 WT）
#   python run.py -prompt bbox     -> 用整肿瘤框提示（MCP-Encoder 提示/上界模式）
# =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-target_class', type=str, default='ALL', choices=['ALL', 'WT', 'TC', 'ET'],
                    help='仅作运行/权重命名标签（三类 WT/TC/ET 始终联合训练与评估），默认 ALL')
parser.add_argument('-prompt', type=str, default='bbox', choices=['None', 'bbox', 'click'],
                    help="提示模式：None=自动(无提示)；bbox=用 GT 自动生成的肿瘤框引导(MCP-Encoder 提示模式)。默认 None")
parser.add_argument('-gpus', type=str, default='0', help='使用的 GPU 编号（逗号分隔）')
parser.add_argument('-data_path', type=str, default='/data/qing_xu/SAM_3D/datasets/Dataset102_met',
                    help='数据集路径')
parser.add_argument('-test_epoch', type=int, default=20,
                    help='测试加载第几个 epoch 的权重；默认 -1 表示自动选该类别的最新 epoch')
parser.add_argument('-skip_test', action='store_true', help='只训练，不自动跑测试')
parser.add_argument('-skip_train', action='store_true', help='只测试，不训练')
args = parser.parse_args()

target_class = args.target_class
nproc = len([g for g in args.gpus.split(',') if g.strip() != ''])
exp_name = f'brats_MedSAM2_{target_class}'   # 日志/实验名带上类别，避免不同类别混在一起
mode_tag = 'bbox 提示模式' if str(args.prompt).lower() == 'bbox' else '自动模式'

print(f"==> 目标类别: {target_class} | {mode_tag} | GPUs: {args.gpus} (nproc={nproc})")

# --------------------------- 训练 ---------------------------
if not args.skip_train:
    train_cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpus} torchrun --nproc_per_node={nproc} train_3d.py "
        f"-target_class {target_class} -prompt {args.prompt} -exp_name {exp_name} "
        f"-sam_ckpt checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s "
        f"-dataset brats -data_path {args.data_path}"
    )
    print(f"[TRAIN] {train_cmd}")
    os.system(train_cmd)

# --------------------------- 测试 ---------------------------
if not args.skip_test:
    if args.test_epoch >= 0:
        test_ckpt = f"outputs/epoch_{args.test_epoch}_{target_class}.pth"
    else:
        # 自动挑选该类别 epoch 编号最大的权重，避免把 epoch 数写死
        def _ep(p):
            m = re.search(r'epoch_(\d+)_', os.path.basename(p))
            return int(m.group(1)) if m else -1
        ckpts = glob.glob(f"outputs/epoch_*_{target_class}.pth")
        test_ckpt = max(ckpts, key=_ep) if ckpts else None

    if test_ckpt is None or not os.path.exists(test_ckpt):
        print(f"[WARN] 找不到 {target_class} 的权重 (期望: {test_ckpt})，跳过测试。"
              f"请确认训练已完成或用 -test_epoch 指定。")
    else:
        test_cmd = (
            f"python test_3d.py -target_class {target_class} -prompt {args.prompt} -exp_name {exp_name} "
            f"-sam_ckpt {test_ckpt} -sam_config sam2_hiera_s "
            f"-dataset brats -data_path {args.data_path}"
        )
        print(f"[TEST ] {test_cmd}")
        os.system(test_cmd)