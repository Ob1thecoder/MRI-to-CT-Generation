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

## Model Architecture

### Generator (U-Net)
- Encoder: 8 downsampling blocks with increasing filters (64→512)
- Decoder: 7 upsampling blocks with skip connections
- Output: Tanh activation for [-1, 1] range

### Discriminator (PatchGAN)
- Classifies 70×70 patches as real/fake
- Takes both condition (MRI) and target (CT) as input
- 4 convolutional layers with batch normalization

## Training Details

- **Loss Function**: Adversarial loss + 100×L1 loss
- **Optimizer**: Adam (lr=2e-4, β₁=0.5, β₂=0.999)
- **Batch Size**: 8 (adjustable)
- **Image Size**: 256×256 pixels
- **Data Augmentation**: Random flips, rotations, and scaling

## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **L1 Loss**: Mean Absolute Error between real and synthetic CT
- **Adversarial Loss**: GAN loss for realistic image generation

## Output

The model generates:
- Individual synthetic CT images in `outputs/manual_eval/synthetic_only/`
- Side-by-side comparison images (MRI | Real CT | Synthetic CT) in `outputs/manual_eval/triptych/`
