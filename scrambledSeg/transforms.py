"""Transforms for data augmentation."""
from typing import Dict
import logging
import albumentations as A
import cv2

logger = logging.getLogger(__name__)

# Map string border modes to cv2 constants
BORDER_MODES = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'replicate': cv2.BORDER_REPLICATE,
    'wrap': cv2.BORDER_WRAP
}

def create_train_transform(config: Dict) -> A.Compose:
    """Create training augmentation pipeline based on config.
    
    Args:
        config: Configuration dictionary with augmentation settings
        
    Returns:
        Albumentations Compose object with transforms
    """
    transforms = []
    aug_config = config.get('augmentation', {})
    
    # Basic transforms
    transforms.extend([
        A.RandomRotate90(p=aug_config.get('rotate_prob', 1.0)),
        A.HorizontalFlip(p=aug_config.get('flip_prob', 0.5)),
        A.VerticalFlip(p=aug_config.get('flip_prob', 0.5)),
    ])
    
    # Rotation with border handling
    if aug_config.get('rotate_limit', 0) > 0:
        border_mode = BORDER_MODES[aug_config.get('rotate_border_mode', 'constant')]
        transforms.append(
            A.Rotate(
                limit=aug_config.get('rotate_limit', 360),
                border_mode=border_mode,
                fill=aug_config.get('rotate_border_value', 0),  # Changed from border_value to fill
                fill_mask=aug_config.get('rotate_mask_border_value', 0),  # Added fill_mask
                p=aug_config.get('rotate_prob', 1.0)
            )
        )
    
    # Brightness and contrast
    if aug_config.get('brightness_contrast_prob', 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness_limit', [-0.2, 0.2]),
                contrast_limit=aug_config.get('contrast_limit', [-0.2, 0.2]),
                brightness_by_max=aug_config.get('brightness_by_max', True),
                p=aug_config.get('brightness_contrast_prob', 0.5)
            )
        )
    
    # Gamma correction
    if aug_config.get('gamma_prob', 0) > 0:
        # Get gamma limits from config (e.g., [80, 120])
        gamma_limits = aug_config.get('gamma_limit', [80, 120])
        transforms.append(
            A.RandomGamma(
                gamma_limit=gamma_limits,  # Pass directly as albumentations handles the conversion
                p=aug_config.get('gamma_prob', 0.5)
            )
        )
    
    # Blur
    if aug_config.get('blur_prob', 0) > 0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=aug_config.get('blur_limit', [3, 7]),
                p=aug_config.get('blur_prob', 0.1)
            )
        )
    
    # Noise
    if aug_config.get('gaussian_noise_prob', 0) > 0:
        noise_limit = aug_config.get('gaussian_noise_limit', [0.005,0.02])
        transforms.append(
            A.GaussNoise(
                std_range=noise_limit,  
                p=aug_config.get('gaussian_noise_prob', 0.2)
            )
        )

    # Always ensure the output format is consistent
    transforms.append(
        A.Normalize(mean=0, std=1),  # Changed from mean=0.5, std=0.5 to keep original range
    )

    transform = A.Compose(
        transforms,
        additional_targets={'mask': 'mask'}
    )
    logger.info("Created training transform pipeline with %d transforms", len(transforms))
    return transform

def create_val_transform(config: Dict) -> A.Compose:
    """Create validation augmentation pipeline based on config.
    
    Args:
        config: Configuration dictionary with augmentation settings
        
    Returns:
        Albumentations Compose object with transforms
    """
    transforms = []
    # Always ensure the output format is consistent
    transforms.append(
        A.Normalize(mean=0, std=1),  # Changed from mean=0.5, std=0.5 to keep original range
    )

    transform = A.Compose(
        transforms,
        additional_targets={'mask': 'mask'}
    )
    logger.info("Created validation transform pipeline with normalization only")
    return transform
