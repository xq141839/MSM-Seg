# *MSM-Seg*: A Modality-and-Slice Memory Framework with Category-Agnostic Prompting for Multi-Modal Brain Tumor Segmentation


[[`arXiv`]()]  

[Yuxiang Luo]()<sup>1*</sup> [Qing Xu](https://scholar.google.com/citations?user=IzA-Ij8AAAAJ&hl=en&authuser=1)<sup>2,6*</sup> [Hai Huang]()<sup>3</sup> [Yuqi Ouyang]()<sup>4</sup> [Zhen Chen](https://franciszchen.github.io/)<sup>5✉</sup> [Wenting Duan](https://scholar.google.com/citations?user=H9C0tX0AAAAJ&hl=zh-CN&authuser=1)<sup>6</sup>

<sup>1</sup>Waseda University &emsp; <sup>2</sup>University of Nottingham &emsp; <sup>3</sup>Northeast Agricultural University &emsp; <sup>4</sup>Sichuan University &emsp; <sup>5</sup>Yale University &emsp; <sup>6</sup>Univeristy of Lincoln &emsp; 

<sup>*</sup> Equal Contribution.  <sup>✉</sup> Corresponding Author. 

-------------------------------------------
![introduction](figs/method.png)

## 📰News

- **[2025.10.12]** We have released the code for MSM-Seg!
## 🛠Setup

```bash
git clone https://github.com/xq141839/MSM-Seg.git
cd MSM-Seg
conda create -f environment.yaml
```

**Key requirements**: Cuda 12.2+, PyTorch 2.4+,

## 📚Data Preparation
- **BraTS**: [Challenge Link](https://www.synapse.org/brats2024)

The data structure is as follows.
```
MSM-Seg
├── brats2024_met
│   ├── data
│     ├── BraTS-MET-00001-000
          ├──BraTS-MET-00001-000-seg.nii
          ├──BraTS-MET-00001-000-t1c.nii
          ├──BraTS-MET-00001-000-t1n.nii
          ├──BraTS-MET-00001-000-t2f.nii
          ├──BraTS-MET-00001-000-t2w.nii
|     ├── ...
|   ├── data_split.json
```
The json structure is as follows.

    { 
     "train": ['BraTS-MET-00674-000',...],
     "valid": ['BraTS-MET-00283-000',...],
     "test":  ['BraTS-MET-00674-000',...] 
     }

## 🎪Quickstart
* Train the Co-Seg++ with the default settings:
```python

python train_3d.py -net sam2 -exp_name brain_mets -vis 0 -sam_ckpt checkpoints/$SAM2 CHECKPOINT$ -sam_config $SAM2 CONFIG$ -b 4 -dataset brats -data_path $YOUR DATASET PATH$


```



## Acknowledgements

* [SAM2](https://github.com/facebookresearch/sam2)
* [MedSAM2](https://github.com/bowang-lab/MedSAM2)
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)


