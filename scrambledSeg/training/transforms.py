"""Training-only transforms for data augmentation."""

import logging

import albumentations as A
import cv2

from .config import AugmentationConfig

logger = logging.getLogger(__name__)


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


def create_train_transform(config: AugmentationConfig) -> A.Compose:
    """Create the training augmentation pipeline."""
    transforms = []

    transforms.extend(
        [
            A.RandomRotate90(p=config.rotate_prob),
            A.HorizontalFlip(p=config.flip_prob),
            A.VerticalFlip(p=config.flip_prob),
        ]
    )

    if config.rotate_limit > 0:
        border_mode = _resolve_border_mode(config.rotate_border_mode)
        transforms.append(
            A.Rotate(
                limit=config.rotate_limit,
                border_mode=border_mode,
                fill=config.rotate_border_value,
                fill_mask=config.rotate_mask_border_value,
                p=config.rotate_prob,
            )
        )

    if config.brightness_contrast_prob > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                brightness_by_max=config.brightness_by_max,
                p=config.brightness_contrast_prob,
            )
        )

    if config.gamma_prob > 0:
        transforms.append(
            A.RandomGamma(
                gamma_limit=config.gamma_limit,
                p=config.gamma_prob,
            )
        )

    if config.blur_prob > 0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=config.blur_limit,
                p=config.blur_prob,
            )
        )

    if config.gaussian_noise_prob > 0:
        transforms.append(
            A.GaussNoise(std_range=config.gaussian_noise_limit, p=config.gaussian_noise_prob)
        )

    transforms.append(_create_normalize_transform())
    logger.info(f"Created training transform pipeline with {len(transforms)} transforms")

    return A.Compose(transforms, additional_targets={"mask": "mask"})


def create_val_transform() -> A.Compose:
    """Create the validation transform pipeline."""
    transforms = [_create_normalize_transform()]
    logger.info("Created validation transform pipeline with normalization only")
    return A.Compose(transforms, additional_targets={"mask": "mask"})
