# Photoacoustic Image Reconstruction

📄 Report: [report.pdf](./report.pdf)

---

## Overview

This project reconstructs optical absorption maps from dual-wavelength (755 nm, 808 nm) photoacoustic pressure signals using deep learning.

We compare:
- **ResNet18 U-Net (CNN baseline)**
- **SwinV2 U-Net (Transformer-based encoder)**

The task is formulated as supervised image-to-image regression:

> pressure (2 channels) → absorption (2 channels)

---

## Dataset

The dataset is not included in this repository due to confidentiality.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

### Baseline (ResNet18 U-Net)

```bash
python scripts/train.py --config configs/baseline.yaml 
```

## SwinV2 U-Net
```bash
python scripts/train.py --config configs/swin.yaml
```

## Evaluation
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/<run_name>/checkpoint_best_ssim.pth \
  --output-dir ./eval_results
```

## Evaluation Outputs
```bash
eval_results/<run_name>/
```
