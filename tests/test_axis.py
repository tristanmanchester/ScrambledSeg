"""Tests for axis-aligned volume slicing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")

from scrambledSeg.prediction.axis import Axis, AxisPredictor


def test_axis_predictor_uses_three_distinct_slice_families() -> None:
    """XY, XZ, and YZ should each front a different source dimension."""

    handler = AxisPredictor()
    volume = np.arange(24).reshape(2, 3, 4)

    assert handler.get_volume_shape(Axis.XY, volume.shape) == (2, 3, 4)
    assert handler.get_volume_shape(Axis.XZ, volume.shape) == (3, 2, 4)
    assert handler.get_volume_shape(Axis.YZ, volume.shape) == (4, 2, 3)

    assert np.array_equal(handler.get_slice(volume, Axis.XY, 1), volume[1, :, :])
    assert np.array_equal(handler.get_slice(volume, Axis.XZ, 2), volume[:, 2, :])
    assert np.array_equal(handler.get_slice(volume, Axis.YZ, 3), volume[:, :, 3])


def test_axis_predictor_supports_multichannel_volume_views() -> None:
    """Channel-aware volumes should preserve channels while slicing along each axis."""

    handler = AxisPredictor()
    volume = np.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)

    assert handler.get_volume_shape(Axis.XY, volume.shape) == (2, 5, 3, 4)
    assert handler.get_volume_shape(Axis.XZ, volume.shape) == (3, 5, 2, 4)
    assert handler.get_volume_shape(Axis.YZ, volume.shape) == (4, 5, 2, 3)

    assert np.array_equal(handler.get_slice(volume, Axis.XZ, 1), volume[:, :, 1, :].transpose(1, 0, 2))
    assert np.array_equal(handler.get_slice(volume, Axis.YZ, 2), volume[:, :, :, 2].transpose(1, 0, 2))
