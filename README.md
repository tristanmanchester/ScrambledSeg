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
- Combined BCE and Dice loss for binary segmentation, and Cross-Entropy with Dice loss for multi-phase segmentation
- Support for both binary and multi-phase (multiple class) segmentation
- Support for multiple prediction axes (X, Y, Z)

## Example Training Visualisations

| Epoch 0 Sample | Epoch 13 Sample |
|:-------------:|:-------------:|
| <img src="https://github.com/tristanmanchester/ScrambledSeg/blob/main/epoch_0_sample_1.jpg" width="500"> | <img src="https://github.com/tristanmanchester/ScrambledSeg/blob/main/epoch_13_sample_0.jpg" width="500"> |

## Technical Details

- Built on PyTorch/PyTorch Lightning
- Uses Hugging Face Transformers library for SegFormer backbone
- Supports CUDA acceleration
- Processes 16-bit grayscale input data
- Configurable via YAML for experiment parameters

## Multi-Phase Segmentation

ScrambledSeg now supports multi-phase segmentation, allowing simultaneous identification of multiple materials or phases in tomographic data:

- **Label Format**: Supports integer-valued labels (0, 1, 2, 3, etc.) where each value represents a distinct phase or material
- **Configuration**: Set `num_classes` in the training configuration to the number of phases (including background)
- **Loss Function**: Automatically switches to an optimized combination of Cross-Entropy and multi-class Dice loss
- **Metrics**: Uses multi-class IoU (Jaccard Index) for accurate performance tracking
- **Visualization**: Improved visualization with class-appropriate colormaps to distinguish different phases
- **Inference**: Multi-phase prediction produces integer-valued output maps with class indices

This extension makes ScrambledSeg suitable for complex material science applications including battery materials, multi-phase alloys, and other composite material systems.

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
pixi run python -m scrambledSeg.training.train .\configs\training_config.yaml
```

The configuration file specifies training parameters, data paths, and model architecture settings.

#### Multi-Phase Segmentation Configuration

To train a model for multi-phase segmentation:

1. Modify the config file to specify multiple classes:
   ```yaml
   # Set number of classes (including background)
   num_classes: 4  # For a 4-phase segmentation (0, 1, 2, 3)
   
   # Update loss function settings
   loss:
     type: "crossentropy_dice"  # Multi-class loss
     params:
       ce_weight: 0.7      # Weight for Cross Entropy component
       dice_weight: 0.3    # Weight for Dice component
       smooth: 0.1         # Smoothing factor for Dice loss
   
   # Update visualization settings
   visualization:
     cmap: tab10  # Discrete colormap suitable for multi-class visualizations
   ```

2. Prepare your training data with integer labels:
   - Each pixel should have a single integer value representing its class
   - Classes should be consecutive integers starting from 0 (background)
   - The preprocessing pipeline will automatically detect and handle multi-class labels

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

#### Multi-Phase Prediction

For multi-phase segmentation models:

1. The prediction pipeline automatically detects multi-class models based on the `num_classes` parameter
2. Multi-phase predictions are output as integer-valued arrays where each value represents a distinct class
3. The output format depends on file type:
   - H5 files: Integer arrays with class indices (0, 1, 2, 3, etc.)
   - TIFF files: Integer arrays with class indices, saved as uint8/uint16
4. For visualization, use a discrete colormap (like 'tab10', 'Set1', or 'viridis') to view the results

Example visualization in Python:
```python
import matplotlib.pyplot as plt
import h5py

# Load multi-phase predictions
with h5py.File('prediction.h5', 'r') as f:
    pred = f['/data'][:]

# Plot with a discrete colormap
plt.figure(figsize=(10, 10))
plt.imshow(pred[50], cmap='tab10')  # View slice 50
plt.colorbar(label='Phase')
plt.title('Multi-Phase Segmentation')
plt.savefig('multi_phase_result.png')
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{manchester2025leveraging,
  title={Leveraging Modified Ex Situ Tomography Data for Segmentation of In Situ Synchrotron X-Ray Computed Tomography},
  author={Manchester, Tristan and Anders, Adam and Spadotto, Julio and Eccleston, Hannah and Beavan, William and Arcis, Hugues and Connolly, Brian J.},
  journal={arXiv preprint arXiv:2504.19200},
  year={2025},
  doi={10.48550/arXiv.2504.19200}
}
```
You can also cite it as:

Manchester, T., Anders, A., Spadotto, J., Eccleston, H., Beavan, W., Arcis, H., & Connolly, B. J. (2025). Leveraging Modified Ex Situ Tomography Data for Segmentation of In Situ Synchrotron X-Ray Computed Tomography. arXiv:2504.19200. https://doi.org/10.48550/arXiv.2504.19200
