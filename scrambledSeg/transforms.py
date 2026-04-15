"""Transforms for data augmentation."""

import logging
from typing import TypedDict

import albumentations as A
import cv2

logger = logging.getLogger(__name__)


class AugmentationConfig(TypedDict, total=False):
    """Augmentation settings supported by the training transform builder."""

    rotate_prob: float
    rotate_limit: int
    rotate_border_mode: str
    rotate_border_value: int
    rotate_mask_border_value: int
    flip_prob: float
    brightness_contrast_prob: float
    brightness_limit: float | list[float]
    contrast_limit: float | list[float]
    brightness_by_max: bool
    gamma_prob: float
    gamma_limit: list[int]
    blur_prob: float
    blur_limit: list[int]
    gaussian_noise_prob: float
    gaussian_noise_limit: list[float]


class TransformConfig(TypedDict, total=False):
    """Subset of the training configuration needed to build transforms."""

    augmentation: AugmentationConfig


# Map string border modes to cv2 constants
BORDER_MODES = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
}


def _resolve_border_mode(name: str) -> int:
    """Translate a configured border-mode name into the matching OpenCV constant."""

    try:
        return BORDER_MODES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rotate_border_mode '{name}'. Choose from {sorted(BORDER_MODES)}."
        ) from exc


def _create_normalize_transform() -> A.Normalize:
    """Create the shared normalization transform used for train and validation."""

    return A.Normalize(
        mean=0,
        std=1,
        max_pixel_value=1.0,
    )


def create_train_transform(config: TransformConfig) -> A.Compose:
    """Create training augmentation pipeline based on config.

    Args:
        config: Configuration dictionary with augmentation settings

    Returns:
        Albumentations Compose object with transforms
    """
    transforms = []
    aug_config = config.get("augmentation", {})

    # Basic transforms
    transforms.extend(
        [
            A.RandomRotate90(p=aug_config.get("rotate_prob", 1.0)),
            A.HorizontalFlip(p=aug_config.get("flip_prob", 0.5)),
            A.VerticalFlip(p=aug_config.get("flip_prob", 0.5)),
        ]
    )

    # Rotation with border handling
    if aug_config.get("rotate_limit", 0) > 0:
        border_mode = _resolve_border_mode(aug_config.get("rotate_border_mode", "constant"))
        transforms.append(
            A.Rotate(
                limit=aug_config.get("rotate_limit", 360),
                border_mode=border_mode,
                fill=aug_config.get("rotate_border_value", 0),  # Changed from border_value to fill
                fill_mask=aug_config.get("rotate_mask_border_value", 0),  # Added fill_mask
                p=aug_config.get("rotate_prob", 1.0),
            )
        )

    # Brightness and contrast
    if aug_config.get("brightness_contrast_prob", 0) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get("brightness_limit", [-0.2, 0.2]),
                contrast_limit=aug_config.get("contrast_limit", [-0.2, 0.2]),
                brightness_by_max=aug_config.get("brightness_by_max", True),
                p=aug_config.get("brightness_contrast_prob", 0.5),
            )
        )

    # Gamma correction
    if aug_config.get("gamma_prob", 0) > 0:
        # Get gamma limits from config (e.g., [80, 120])
        gamma_limits = aug_config.get("gamma_limit", [80, 120])
        transforms.append(
            A.RandomGamma(
                gamma_limit=gamma_limits,  # Pass directly as albumentations handles the conversion
                p=aug_config.get("gamma_prob", 0.5),
            )
        )

    # Blur
    if aug_config.get("blur_prob", 0) > 0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=aug_config.get("blur_limit", [3, 7]),
                p=aug_config.get("blur_prob", 0.1),
            )
        )

    # Noise
    if aug_config.get("gaussian_noise_prob", 0) > 0:
        noise_limit = aug_config.get("gaussian_noise_limit", [0.005, 0.02])
        transforms.append(
            A.GaussNoise(std_range=noise_limit, p=aug_config.get("gaussian_noise_prob", 0.2))
        )

    transforms.append(_create_normalize_transform())
    logger.info(f"Created training transform pipeline with {len(transforms)} transforms")

    return A.Compose(transforms, additional_targets={"mask": "mask"})


def create_val_transform(config: TransformConfig) -> A.Compose:
    """Create validation augmentation pipeline based on config.

    Args:
        config: Configuration dictionary with augmentation settings

    Returns:
        Albumentations Compose object with transforms
    """
    transforms = []
    # Always ensure the output format is consistent
    transforms.append(_create_normalize_transform())
    logger.info("Created validation transform pipeline with normalization only")
    return A.Compose(transforms, additional_targets={"mask": "mask"})
