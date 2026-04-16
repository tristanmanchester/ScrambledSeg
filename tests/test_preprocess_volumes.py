"""Tests for 3D volume preprocessing across all orientations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from scrambledSeg.data.preprocess_volumes import (
    SliceInfo,
    create_datasets,
    extract_tiles_from_slice,
)


def test_extract_tiles_from_slice_covers_all_orientations_for_non_cubic_volume(
    tmp_path: Path,
) -> None:
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


def test_create_datasets_raises_when_expected_tile_files_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dataset creation should fail if the extracted tiles were never persisted."""

    monkeypatch.setattr(
        "scrambledSeg.data.preprocess_volumes.calculate_total_tiles",
        lambda data_paths, tile_size=256, overlap=64: 1,
    )
    monkeypatch.setattr(
        "scrambledSeg.data.preprocess_volumes._process_slice",
        lambda args: (
            args[0],
            args[1],
            [SliceInfo("volume", "xy_slice000", 0)],
        ),
    )

    with pytest.raises(FileNotFoundError, match="Expected extracted tile files"):
        create_datasets(
            data_paths=["/tmp/data.tif"],
            label_paths=["/tmp/label.tif"],
            output_dir=tmp_path / "out",
            split_ratios={"train": 1.0},
        )


def test_create_datasets_passes_tile_geometry_to_total_tile_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dataset creation should estimate tile counts with the requested overlap."""

    captured: dict[str, object] = {}

    def _capture_total_tiles(data_paths, tile_size=256, overlap=32):
        captured["data_paths"] = data_paths
        captured["tile_size"] = tile_size
        captured["overlap"] = overlap
        return 1

    monkeypatch.setattr(
        "scrambledSeg.data.preprocess_volumes.calculate_total_tiles",
        _capture_total_tiles,
    )
    monkeypatch.setattr(
        "scrambledSeg.data.preprocess_volumes._process_slice",
        lambda args: (
            args[0],
            args[1],
            [SliceInfo("volume", "xy_slice000", 0)],
        ),
    )

    with pytest.raises(FileNotFoundError):
        create_datasets(
            data_paths=["/tmp/data.tif"],
            label_paths=["/tmp/label.tif"],
            output_dir=tmp_path / "out",
            split_ratios={"train": 1.0},
            tile_size=128,
            overlap=12,
        )

    assert captured == {
        "data_paths": ["/tmp/data.tif"],
        "tile_size": 128,
        "overlap": 12,
    }
