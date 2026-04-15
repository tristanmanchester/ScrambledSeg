"""Tests for TIFF dataset loading and caching."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from scrambledSeg.data.datasets import SynchrotronDataset


def _write_dataset(root: Path, n_samples: int = 5) -> Path:
    split_dir = root / "train"
    data_dir = split_dir / "data"
    labels_dir = split_dir / "labels"
    data_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for idx in range(n_samples):
        image = np.full((4, 4), idx, dtype=np.float32)
        mask = np.full((4, 4), idx + 10, dtype=np.uint16)
        tifffile.imwrite(data_dir / f"{idx:03d}.tif", image)
        tifffile.imwrite(labels_dir / f"{idx:03d}.tif", mask)

    return root


def _disable_tensor_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SynchrotronDataset,
        "_process_image",
        lambda self, image, is_mask=False: image,
    )
    monkeypatch.setattr(
        SynchrotronDataset,
        "_apply_transforms",
        lambda self, image, mask: (image, mask),
    )


def test_synchrotron_dataset_loads_without_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """cache_size=0 should still create a usable dataset and load the right sample."""

    dataset_root = _write_dataset(tmp_path)
    dataset = SynchrotronDataset(dataset_root, split="train", cache_size=0)
    _disable_tensor_conversion(monkeypatch)

    sample = dataset[0]

    assert dataset._cache == {}
    assert np.array_equal(sample["image"], tifffile.imread(dataset.data_files[0])[None, ...])
    assert np.array_equal(sample["mask"], tifffile.imread(dataset.label_files[0])[None, ...])


def test_synchrotron_dataset_subset_cache_matches_selected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached samples should align with the subset-selected file list."""

    dataset_root = _write_dataset(tmp_path, n_samples=6)
    dataset = SynchrotronDataset(
        dataset_root,
        split="train",
        subset_fraction=0.5,
        random_seed=7,
        cache_size=3,
    )
    _disable_tensor_conversion(monkeypatch)

    assert len(dataset._cache) == len(dataset)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        expected_image = tifffile.imread(dataset.data_files[idx])[None, ...]
        expected_mask = tifffile.imread(dataset.label_files[idx])[None, ...]
        assert np.array_equal(sample["image"], expected_image)
        assert np.array_equal(sample["mask"], expected_mask)


def test_synchrotron_dataset_subset_without_cache_matches_selected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Subset selection should also work when caching is disabled."""

    dataset_root = _write_dataset(tmp_path, n_samples=6)
    dataset = SynchrotronDataset(
        dataset_root,
        split="train",
        subset_fraction=0.5,
        random_seed=11,
        cache_size=0,
    )
    _disable_tensor_conversion(monkeypatch)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        expected_image = tifffile.imread(dataset.data_files[idx])[None, ...]
        expected_mask = tifffile.imread(dataset.label_files[idx])[None, ...]
        assert np.array_equal(sample["image"], expected_image)
        assert np.array_equal(sample["mask"], expected_mask)
