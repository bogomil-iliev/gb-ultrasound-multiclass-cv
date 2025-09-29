# GB Ultrasound (9-class) — Lightweight classification with patient-level CV

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bogomil-iliev/gb-ultrasound-multiclass-cv/blob/main/notebooks/gb_ultrasound_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end **PyTorch** pipeline for **gallbladder ultrasound** (9 classes) with **patient-level 5-fold cross-validation**, blur-based quality control, class-imbalance handling, and **Grad-CAM++** explainability.

> MSc AI Thesis: _Detection and Classification of Gallbladder Diseases through Lightweight Deep Learning Techniques._

---

## Highlights
- **Leakage-safe evaluation:** `StratifiedGroupKFold` on `patient_id` with a separate hold-out test split.
- **Image quality control:** Variance-of-Laplacian (VoL); per-fold threshold at the **8th percentile** of the train set only.
- **Lightweight backbones:** GhostNet-1.0, TinyViT-11M, ResNet-50 (via `timm`); mixed precision, early stopping.
- **Class imbalance aware:** class-weighted loss/sampling; report **macro-F1** with class-wise breakdowns.
- **Explainability:** **Grad-CAM++** heatmaps for qualitative checks.

---
## Thesis Abstract
Ultrasound (US) is the most utilised imaging modality for gallbladder disease (GBD), but diagnostical accuracy is linked to operator skill, scanner settings and patient factors. The current work investigates whether lightweight deep learning models, that are designed for real-time use on modest hardware, can reliably classify nine gallbladder pathologies from US samples while meeting important validation standards that reflect clinical deployment.
A reproducible pipeline is designed around three algorithms – ResNet-50 (serving as a baseline), TinyVit with 11 million parameters and GhostNet-1.0 (for benchmarking purposes). To improve robustness on the limited GBD US data that is frequently noisy, the input geometry is standardised, z-score normalisation is applied to each image, and QC filtering is applied. This is done together with conservative augmentations including Gaussian blur. Data leakage is controlled by patient-level grouping: a 5-fold stratified cross validation. Training employs linear probe phase followed by phased unfreezing with discriminative learning rates and early stopping on macro-F1. Class imbalances handling is also further tested with a weighted sampler. Model behaviour is examined with Grad-CAM++ and error analysis through confusion matrix and precision-recall curves.
Under strict patient-level splits, all models achieve modest macro-F1 scores, with clear signs of overfitting once deeper layers are unfrozen. Grad-CAM++ reveals attention to US artefacts alongside pathology, which explains the limited generalisation. A deliberately “easy” demonstration aiming at splitting of US data without patient controls, is produced to showcase much higher headline scores. Thus, confirming that leakage can inflate performance and mask true clinical value and difficulty.
The research concludes that on this multi-class task, methodological discipline (e.g. patient-level evaluation and training) is decisive to provide real generalisable results.

---

## Dataset
This repository **does not** include images. Use the public dataset (Turki et al., 2024) and follow the license on the source page.
Dataset Download Link: https://data.mendeley.com/datasets/r6h24d2d3y/1

## Training and Validation Best Results
- **Achieved with the original methodology**
- **Best performing model is GhostNet-1.0.**

<img width="501" height="85" alt="image" src="https://github.com/user-attachments/assets/42f14567-b609-4fc0-9aba-d3725001f0c5" />

---

## Testing and Evaluation Results of GhostNet-1.0.

## Confusion matrix and PR Curves
<img width="864" height="707" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/f273634b-3697-49e4-9a8b-232c9c2998ee" />
<img width="2148" height="1841" alt="PR Curves" src="https://github.com/user-attachments/assets/635744b5-584b-4e85-9f60-1ac1c3d1a459" />

---

## Grad-CAM++ Results
<img width="864" height="651" alt="Combined FigureCAM" src="https://github.com/user-attachments/assets/9fcc51d1-e392-400f-9e25-afe261faed8c" />

---
## Held-out Test Set Evaluation
<img width="724" height="336" alt="Held out Test Results Table" src="https://github.com/user-attachments/assets/3718085e-9afc-479e-80f9-7b1888ece57f" />

---

## Ethics & limitations

- **Research/educational use only — not a medical device and not for clinical decision-making.**

- **Patient-level cross-validation prevents leakage but does not equal external validation; performance may differ across scanners/sites.**

- **Grad-CAM++ provides qualitative insight; it does not prove causality.**

---

## Citation

If you use this work, please cite:

Bogomil Iliev, Detection and Classification of Gallbladder Diseases through Lightweight Deep Learning Techniques, MSc Thesis, 2025.

@thesis{Iliev2025GBUltrasound,
  author = {Iliev, Bogomil},
  title  = {Detection and Classification of Gallbladder Diseases through Lightweight Deep Learning Techniques},
  school = {University of Greater Manchester (Bolton)},
  year   = {2025},
  note   = {GitHub: https://github.com/bogomil-iliev/gb-ultrasound-multiclass-cv}
}

---

## License

This project is released under the MIT License.







