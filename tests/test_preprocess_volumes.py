"""Tests for 3D volume preprocessing across all orientations."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from scrambledSeg.data.preprocess_volumes import extract_tiles_from_slice


def test_extract_tiles_from_slice_covers_all_orientations_for_non_cubic_volume(tmp_path: Path) -> None:
    """Preprocessing should extract D/H/W slice families even when H and W exceed D."""

    data_volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    label_volume = (data_volume > 10).astype(np.uint8)

    data_path = tmp_path / "data.tif"
    label_path = tmp_path / "label.tif"
    tifffile.imwrite(data_path, data_volume)
    tifffile.imwrite(label_path, label_volume)

    tile_info = extract_tiles_from_slice(
        str(data_path),
        str(label_path),
        tmp_path / "out",
        tile_size=2,
        overlap=0,
    )

    orientations = {info.orientation for info in tile_info}
    counts = Counter(name.split("_slice")[0] for name in orientations)

    assert counts["xy"] == 2
    assert counts["xz"] == 3
    assert counts["yz"] == 4
