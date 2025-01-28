# ScrambledSeg

<p align="center">
  <img src="https://github.com/tristanmanchester/ScrambledSeg/blob/main/scrambled%20seg.png" width="500">
</p>

A deep learning-based segmentation pipeline for in situ synchrotron X-ray computed tomography data, demonstrated through copper oxide dissolution studies.

## Overview

This project implements a modified SegFormer architecture to automatically segment synchrotron tomography data. The key innovation is training on transformed high-quality laboratory XCT data to handle artefact-rich synchrotron data, achieving over 80% IoU while reducing processing time from hours to seconds per volume.

## Key Features

- Custom SegFormer implementation optimized for single-channel tomographic data
- Data preprocessing pipeline to transform lab XCT data for synchrotron conditions
- Comprehensive data augmentation using Albumentations library
- Efficient architecture supporting 512³ volume processing
- Combined BCE and Dice loss function for robust segmentation
- Support for multiple prediction axes (X, Y, Z)

## Example Training Visualisations

| Epoch 0 Sample | Epoch 13 Sample |
|:-------------:|:-------------:|
| <img src="https://github.com/tristanmanchester/ScrambledSeg/blob/main/epoch_0_sample_1.png" width="500"> | <img src="https://github.com/tristanmanchester/ScrambledSeg/blob/main/epoch_13_sample_0.png" width="500"> |

## Technical Details

- Built on PyTorch/PyTorch Lightning
- Uses Hugging Face Transformers library for SegFormer backbone
- Supports CUDA acceleration
- Processes 16-bit grayscale input data
- Configurable via YAML for experiment parameters

## Requirements

- Python 3.10
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- Albumentations >= 2.0.0
- transformers >= 4.30.0
- CUDA-capable GPU with 8+ GB memory recommended

See `pixi.toml` for complete dependency list.

## Usage

### Environment Setup

This project uses `pixi` for environment management. To get started:

1. Install pixi if you haven't already:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Create and activate the environment:
```bash
pixi install
pixi shell
```

### Data Preprocessing

The preprocessing script prepares 3D volumes for training by:
1. Extracting 2D slices from all orientations (X, Y, Z)
2. Organizing slices into train (80%), validation (10%), and test (10%) sets

```bash
# If in pixi shell:
python data/preprocess_volumes.py --data-dir /path/to/raw/data --label-dir /path/to/raw/labels --output-dir /path/to/processed/data

# Or using pixi run:
pixi run -- python data/preprocess_volumes.py --data-dir /path/to/raw/data --label-dir /path/to/raw/labels --output-dir /path/to/processed/data
```

The script expects:
- `data-dir`: Directory containing H5 files with raw volume data
- `label-dir`: Directory containing corresponding H5 label files
- `output-dir`: Where to save the processed datasets

### Training

Training is configured via YAML files in the `configs/` directory:

```bash
pixi run -- python -m synchrotron_segmentation.training.train configs/synchrotron_config_slices.yaml
```

The configuration file specifies training parameters, data paths, and model architecture settings.

Training outputs are saved in several locations:
- Model checkpoints: `lightning_logs/version_*/checkpoints/*.ckpt`
- PyTorch Lightning logs: `lightning_logs/version_*/`
- Detailed metrics: `logs/metrics/metrics.csv`
- Visualizations:
  - Training metrics: `logs/metrics/`
  - Sample predictions: `logs/plots/`

### Inference

The CLI supports single-axis, three-axis, and twelve-axis prediction modes:

```bash
# Basic usage
pixi run -- python predict_cli.py /path/to/input.h5 /path/to/checkpoint.ckpt

# Advanced options
pixi run -- python predict_cli.py \
    /path/to/input.h5 \
    /path/to/checkpoint.ckpt \
    --mode THREE_AXIS \
    --output_dir predictions \
    --batch_size 8 \
    --data_path /data
```

Available prediction modes:
- `SINGLE_AXIS`: Standard single-direction prediction
- `THREE_AXIS`: Predictions from X, Y, and Z directions
- `TWELVE_AXIS`: Enhanced multi-angle predictions for maximum accuracy

## Citations

If you use this code in your research, please cite our paper:

```
Manchester, T., & Connolly, B. J. (2025). Leveraging Modified Ex Situ Tomography Data for 
Segmentation of In Situ Synchrotron X-Ray Computed Tomography.
