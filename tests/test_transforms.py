"""Tests for augmentation and normalization transforms."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")

import scrambledSeg.training.transforms as transforms_module
from scrambledSeg.training.config import AugmentationConfig


def _augmentation_config() -> AugmentationConfig:
    return AugmentationConfig(
        rotate_prob=0.0,
        flip_prob=0.0,
        rotate_limit=0,
        brightness_contrast_prob=0.0,
        gamma_prob=0.0,
        blur_prob=0.0,
        gaussian_noise_prob=0.0,
    )


def test_create_train_transform_preserves_pre_normalized_image_range(
    caplog: pytest.LogCaptureFixture,
) -> None:
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    with caplog.at_level(logging.INFO):
        transform = transforms_module.create_train_transform(_augmentation_config())

    transformed = transform(image=image, mask=mask)

    assert np.allclose(transformed["image"], image)
    assert np.array_equal(transformed["mask"], mask)
    assert "Created training transform pipeline with 4 transforms" in caplog.text


def test_create_val_transform_preserves_pre_normalized_image_range() -> None:
    image = np.array([[0.0, 0.1], [0.9, 1.0]], dtype=np.float32)
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    transform = transforms_module.create_val_transform()
    transformed = transform(image=image, mask=mask)

    assert np.allclose(transformed["image"], image)
    assert np.array_equal(transformed["mask"], mask)


def test_create_train_transform_rejects_unknown_rotate_border_mode() -> None:
    config = replace(
        _augmentation_config(),
        rotate_limit=45,
        rotate_border_mode="mirror-ball",
    )

    with pytest.raises(ValueError, match="Unsupported rotate_border_mode"):
        transforms_module.create_train_transform(config)
