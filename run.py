import os


os.system('python train_3d.py -net sam2 -exp_name brain_mets -vis 0 -sam_ckpt checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -b 4 -dataset brats -data_path preprocessed')