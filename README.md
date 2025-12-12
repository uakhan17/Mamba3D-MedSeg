# Mamba3D-MedSeg

> Official repository for the paper  
> **"A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation"**  
> *Pattern Recognition, 2025*  
>  
>  
>   

---

## 🧠 Overview

This repository will host the official implementation and experiment instructions for our paper *A Comprehensive Analysis of Mamba for 3D Volumetric Medical Image Segmentation*.

Our study provides an extensive exploration of Mamba architectures for volumetric medical image segmentation, benchmarking them against established 3D CNN and Transformer-based methods.

---

## 🔍 Key Analysis

This section highlights several core findings and visual summaries from the paper.  

### Figure 1 – Comparison of Model Architectures Between UlikeMamba and UlikeTrans

![Figure 1 – Mamba-based Network VS. Trannsformer-based Network](assets/mamba_pr1.png)

*High-level illustration of the 3D Mamba-based segmentation pipeline and comparison baselines.*

---

### Figure 2 – Network architecture

![Figure 2 – Network architecture](assets/fig2_architecture.png)

*Block-level architecture of the proposed 3D Mamba backbone and its integration into the nnU-Net style encoder–decoder.*

---

### Figure 3 – Performance across datasets

![Figure 3 – Performance across datasets](assets/fig3_performance.png)

*Summary Dice / HD95 across BraTS 2021, TotalSegmentator (CT), TotalSegmentatorMRI, and AMOS 2022.*

---

### Figure 4 – Efficiency vs. accuracy

![Figure 4 – Efficiency vs. accuracy](assets/fig4_efficiency.png)

*Trade-off between segmentation performance and computational efficiency (parameters, FLOPs, throughput).*

---

### Figure 5 – Qualitative examples

![Figure 5 – Qualitative examples](assets/fig5_qualitative.png)

*Representative 3D visualizations comparing Mamba-based models with CNN/Transformer baselines across different anatomies and modalities.*

---

## ⚙️ Installation

This section provides a template for setting up the environment used in our experiments.  
Fill in the exact commands according to your preferred package manager (conda, pip, etc.) and your hardware setup.

### 1. Clone the repository

```bash
# TODO: add the exact clone command
# Example:
# git clone https://github.com/uakhan17/Mamba3D-MedSeg.git
# cd Mamba3D-MedSeg
