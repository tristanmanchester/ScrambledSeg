# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This project uses `pixi` for package management and environment setup:

### Environment Setup
```bash
pixi install          # Install dependencies
pixi shell           # Activate environment
```

### Development Commands
```bash
pixi run lint        # Format code with black
pixi run format      # Sort imports with isort
pixi run run         # Train model with default config
```

### Training
```bash
# Train with specific config
pixi run python -m scrambledSeg.training.train configs/training_config.yaml

# Or with explicit PYTHONPATH
python scrambledSeg/training/train.py configs/training_config.yaml
```

### Data Preprocessing
```bash
python scrambledSeg/data/preprocess_volumes.py --data-dir /path/to/raw/data --label-dir /path/to/raw/labels --output-dir /path/to/processed/data
```

### Inference
```bash
# Basic prediction
python predict_cli.py /path/to/input.h5 /path/to/checkpoint.ckpt

# Multi-axis prediction
python predict_cli.py /path/to/input.h5 /path/to/checkpoint.ckpt --mode THREE_AXIS --output_dir predictions --batch_size 8
```

## Architecture Overview

ScrambledSeg is a PyTorch Lightning-based deep learning pipeline for synchrotron X-ray tomography segmentation:

### Core Components

- **Model**: Modified SegFormer architecture (`scrambledSeg.models.segformer.CustomSegformer`)
  - Uses Hugging Face transformers backbone (mit-b0 through mit-b5 variants)
  - Single-channel input for grayscale tomographic data
  - Supports both binary and multi-phase segmentation

- **Training Pipeline** (`scrambledSeg.training/`):
  - `train.py`: Main training script with YAML configuration
  - `trainer.py`: PyTorch Lightning module (`SegformerTrainer`)
  - Supports mixed precision training (BF16)
  - Combined losses: BCE+Dice (binary) or CrossEntropy+Dice (multi-class)

- **Data Processing** (`scrambledSeg.data/`):
  - `datasets.py`: Custom dataset for H5/TIFF tomographic data
  - `preprocess_volumes.py`: Converts 3D volumes to 2D slices with train/val/test splits
  - Extracts slices from all orientations (X, Y, Z axes)

- **Prediction** (`scrambledSeg.prediction/`):
  - `predictor.py`: Core prediction engine
  - `axis.py`: Multi-axis prediction strategies (SINGLE_AXIS, THREE_AXIS, TWELVE_AXIS)
  - Supports ensemble predictions from multiple orientations

### Key Features

1. **Multi-Phase Segmentation**: Configure `num_classes` in YAML for multi-material identification
2. **Data Augmentation**: Albumentations pipeline with rotation, flipping, brightness/contrast, noise
3. **Visualization**: Real-time training visualization with sample predictions
4. **Flexible Prediction**: Single or multi-axis inference for enhanced accuracy

### Configuration

Training parameters are specified in YAML files (`configs/training_config.yaml`):
- Model architecture (encoder backbone, number of classes)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and preprocessing settings
- Loss function configuration
- Augmentation parameters

### Output Structure

Training generates:
- Model checkpoints: `lightning_logs/version_*/checkpoints/`
- Training metrics: `logs/metrics/metrics.csv`
- Visualization plots: `logs/plots/`
- PyTorch Lightning logs: `lightning_logs/`

### Dependencies

Primary dependencies include PyTorch, PyTorch Lightning, transformers (Hugging Face), albumentations for augmentation, and h5py/tifffile for data I/O. GPU with 8+ GB memory recommended for training.