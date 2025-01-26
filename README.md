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

## Citations

If you use this code in your research, please cite our paper:

```
Manchester, T., & Connolly, B. J. (2025). Leveraging Modified Ex Situ Tomography Data for 
Segmentation of In Situ Synchrotron X-Ray Computed Tomography.
```
