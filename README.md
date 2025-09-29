# GB Ultrasound (9-class) — Lightweight CV pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bogomil-iliev/gb-ultrasound-multiclass-cv/blob/main/notebooks/gb_ultrasound_pipeline.ipynb)

Lightweight deep learning pipeline for **9-class gallbladder ultrasound** classification with strong reproducibility:
- **Patient-level 5-fold CV** (StratifiedGroupKFold) and 15% hold-out test.  
- **Blur quality control** via Variance-of-Laplacian; per-fold threshold at the **8th percentile** (train-only).  
- **Class-imbalance mitigation** (weighted sampling / loss).  
- **Backbones**: GhostNet-1.0, TinyViT-11M, ResNet-50 (pretrained via `timm`).  
- **Grad-CAM++** visualisation for model interpretability.

> Project for MSc AI thesis: **“Detection and Classification of Gallbladder Diseases through Lightweight Deep Learning Techniques.”**

## Dataset
This repo does **not** host data. Use the notebook or:
```bash
python scripts/download_dataset.py

## Dataset Download Link: https://data.mendeley.com/datasets/r6h24d2d3y/1

