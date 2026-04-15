"""Unit tests for prediction data normalization."""

from __future__ import annotations

from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from scrambledSeg.prediction.data import TomoDataset


def test_normalize_slice_min_max_scales_uint16_inputs() -> None:
    """Uint16 slices should be normalized to the full [0, 1] interval."""

    slice_data = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)

    normalized = TomoDataset().normalize_slice(slice_data)

    expected = torch.tensor(
        [[[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]]],
        dtype=torch.float32,
    )
    assert normalized.shape == (1, 1, 2, 2)
    assert normalized.dtype == torch.float32
    assert torch.allclose(normalized, expected)


def test_normalize_slice_matches_training_style_for_uint8_inputs() -> None:
    """Uint8 inputs should also use min-max normalization like training."""

    slice_data = np.array([[10, 20], [30, 40]], dtype=np.uint8)

    normalized = TomoDataset().normalize_slice(slice_data)

    expected = torch.tensor(
        [[[[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)


def test_normalize_slice_handles_constant_inputs() -> None:
    """Constant slices should normalize without NaNs or infinities."""

    slice_data = np.full((2, 2), 42, dtype=np.uint16)

    normalized = TomoDataset().normalize_slice(slice_data)

    assert torch.count_nonzero(normalized) == 0
    assert torch.isfinite(normalized).all()
