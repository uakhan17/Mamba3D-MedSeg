# Mamba3D-MedSeg

> Official repository for the paper  
> **"A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation"**  
> *Pattern Recognition, 2025*  
>  
>  
>   

---

## :eyes: Overview

This repository hosts the official implementation and experiment instructions for our paper "*A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation*".

Our study provides an extensive exploration of Mamba architectures for volumetric medical image segmentation, benchmarking them against established 3D CNN and Transformer-based methods.

---

## 🔍 Key Analyses

This section highlights several core findings and visual summaries from our paper.  

### Comparison of Model Architectures Between UlikeMamba and UlikeTrans

<div align="center">
  <img src="assets/mamba_pr1.png" width="80%">
</div>

*High-level illustration of the 3D Mamba-based segmentation pipeline and comparison baselines.*

---

### Network Configurations

<div align="center">
  <img src="assets/mamba_pr2.png" width="80%">
</div>

*Left: detailed configurations of UlikeMamba 3d network. Here, ‘K’: kernel size of Conv, DW-Conv, or TransposeConv; ‘C’: number of channels; and ‘S’: stride. Right: Detailed configurations of Ulike-Trans SRA network. Here, ‘R’: reduction ratio of SRA; ‘H’: head number of SRA; and ‘E’: expansion ratio of FFN.*

---

### Multiscale Feature Fusion Strategies

<div align="center">
  <img src="assets/mamba_pr3.png" width="80%">
</div>

*Four multi-scale modeling schemes for evaluating and comparing the long-range dependency modeling capabilities of Mamba and Transformers for multi-scale representation learning.*

---

### Scanning Strategies

<div align="center">
  <img src="assets/mamba_pr4.png" width="80%">
</div>

*UlikeMamba_3d with different sequential scanning strategies.*

---

### UlikeMamba_3dMT Architecture

<div align="center">
  <img src="assets/mamba_pr5.png" width="70%">
</div>

*Our proposed Mamba layer in UlikeMamba_3dMT, which modifies the original 1D depthwise convolution to 3D depthwise convolution, embraces a multi-scale strategy and incorporates tri-directional scanning to capture comprehensive spatial relationships in 3D volumetric data more effectively.*

---

## 🛠️: Installation
We tested our code on `Ubuntu 22.04.5 LTS`.

### 1. Create Virtual Environment

```bash
conda create -n mambaseg python=3.10 -y
conda activate mambaseg
```

### 2. Install Pytorch & other Key Packages
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.37.2 timm==1.0.7
```

### 3. Install Mamba & Causal-conv1d
Go to [Mamba Release Page](https://github.com/state-spaces/mamba/releases) and [Causal-conv1d Release Page](https://github.com/Dao-AILab/causal-conv1d/releases). Search for versions that are compatible with your system spec. Download wheels then install. For our experiments, we used `mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` and `causal_conv1d-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
```bash
pip install mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Clone the repository

```bash
git clone https://github.com/uakhan17/Mamba3D-MedSeg.git
cd Mamba3D-MedSeg
pip install -e .
```
**Sanity Check**
```
import torch
import mamba_ssm
```

## 📂: Data Preparation
Four datsets were involved in our analyses. Since all experiments are integrated into the nnUNet_v2 framework, please follow **nnUNet_v2** conventions for folder structure (such as setting environment variables for **nnUNet_raw, nnUNet_preprocessed, and nnUNet_results**) and data preprocessing.
- **CT Datasets**
  - AMOS2022_postChallenge_part1
    - task: Multi-organ abdominal segmentation on CT
    - Official info / access: [AMOSS2022 Challenge Page](https://zenodo.org/record/6361922)
  - TotalSegmentator-CT
    - task: Whole-body CT segmentation of >100 anatomical structures
    - Official info / access: [TotalSegmentator Project Page](https://totalsegmentator.com/)
- **MRI Datasets**
  - BraTS 2021
    - Task: Brain tumor segmentation on multi-parametric MRI
    - Official info / access: [BraTS Challenge Page](https://www.med.upenn.edu/cbica/brats/)
  - TotalSegmentatorMRI
    - task: Whole-body MRI segmentation with multiple anatomical structures
    - Official info / access: [TotalSegmentator Project Page](https://totalsegmentator.com/)

## 🚀: Model Training
We have integrated trainers for SegFormer3D, UNETR, SwinUNETR, Umamba, CoTr and ours UlikeMamba_3dMT into the codebase. For example, we assign ID 218 to AMOSS2022_postChallenge_task1 after proprocessing. The command for running experiment on it with 2 GPUs would be:

`MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,3 nnUNetv2_train 218 3d_fullres 0 -tr nnUNetTrainerUlikeMamba_3dMT -num_gpus 2`

Please refer to [nnUNet_v2 Usage Instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md) for more details.

## 🧾 Citation
If you find our work useful, please cite:

```bibtex
@article{wang2025comprehensive,
  title={A comprehensive analysis of Mamba for 3D volumetric medical image segmentation},
  author={Wang, Chaohan and Xie, Yutong and Chen, Qi and Zhou, Yuyin and Wu, Qi},
  journal={Pattern Recognition},
  year={2025}
}
