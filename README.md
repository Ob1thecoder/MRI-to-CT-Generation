# MRI-to-CT-Generation

Generate synthetic CT images from given MRI scans. This aims to reduce the need for additional CT scans in order to save costs and more importantly, to minimize radiation exposure for patients.

## Overview

This project implements a Pix2Pix conditional GAN (cGAN) model to generate synthetic CT images from T1-weighted MRI scans. The model is trained on paired MRI-CT data and can produce high-quality synthetic CT images that maintain anatomical consistency with the input MRI.

## Features

- **Pix2Pix cGAN Architecture**: Uses a U-Net generator with skip connections and a PatchGAN discriminator
- **Robust Preprocessing**: Implements percentile-based normalization and data augmentation
- **Paired Data Loading**: Efficiently handles paired MRI-CT datasets with automatic patient matching
- **Comprehensive Metrics**: Tracks PSNR, SSIM, and GAN losses during training
- **Docker Support**: Containerized environment for easy deployment and reproducibility
## Training the Model

1. Open the [pix2pix.ipynb](utils/pix2pix.ipynb) notebook
2. Update the data paths in the training section:
   ```python
   ct_folder = "../data/CT/PNG"
   t1_folder = "../data/T1-MRI/PNG"
   ```
3. Configure training parameters:
   ```python
   IMAGE_SIZE = (256, 256)
   BATCH = 8
   EPOCHS = 100
   ```
4. Run the training cells to start training

### Key Components

#### Data Preprocessing ([preprocess.py](utils/preprocess.py))
- **[`PairedCTT2Loader`](utils/preprocess.py)**: Custom data loader for paired MRI-CT datasets
- **[`robust_norm`](utils/preprocess.py)**: Percentile-based normalization to [-1, 1] range
- **[`natural_key`](utils/preprocess.py)**: Natural sorting for consistent slice ordering

#### Model Architecture ([pix2pix.py](utils/pix2pix.py))
- **[`build_generator`](utils/pix2pix.py)**: U-Net architecture with skip connections
- **[`build_discriminator`](utils/pix2pix.py)**: PatchGAN discriminator for realistic texture generation
- **[`Pix2Pix`](utils/pix2pix.py)**: Main model class with adversarial + L1 loss

## Architecture
 
| Component | Details |
|-----------|---------|
| Generator | U-Net (encoder-decoder with skip connections) |
| Discriminator | PatchGAN (70×70 receptive field) |
| Loss | Adversarial (BCE) + L1 reconstruction loss |
| Framework | PyTorch |
 
---
 
## Results
 
The model learns to map structural features from MRI (soft tissue contrast) to the Hounsfield unit ranges expected in CT (bone = bright, air = dark, soft tissue = mid-range).
 
Evaluation metrics:
- **SSIM** (Structural Similarity Index) — measures perceptual similarity
- **MAE** (Mean Absolute Error) — measures pixel-level reconstruction accuracy
 
---
 
## Setup
 
```bash
git clone https://github.com/Ob1thecoder/MRI-to-CT-Generation
cd MRI-to-CT-Generation
pip install torch torchvision opencv-python matplotlib jupyter
```
 
Open `MRI_to_CT.ipynb` in Jupyter and run cells top to bottom.
 
**Dataset:** Paired MRI–CT volumes (e.g. [Gold Atlas](https://www.cancerimagingarchive.net/) or similar). Place in `data/train/` with subdirectories `mri/` and `ct/`.
 
---
 
## Project Structure
 
```
MRI-to-CT-Generation/
├── MRI_to_CT.ipynb      # Main notebook (data loading, training, evaluation)
├── data/
│   ├── train/
│   │   ├── mri/
│   │   └── ct/
│   └── test/
└── outputs/             # Generated images saved here
```
 
---
## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **L1 Loss**: Mean Absolute Error between real and synthetic CT
- **Adversarial Loss**: GAN loss for realistic image generation

## Output

The model generates:
- Individual synthetic CT images in `outputs/manual_eval/synthetic_only/`
- Side-by-side comparison images (MRI | Real CT | Synthetic CT) in `outputs/manual_eval/triptych/`
