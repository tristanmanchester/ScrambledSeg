"""Tests for the SynchrotronDataset data loader."""

from pathlib import Path

import numpy as np
import torch
import tifffile

from scrambledSeg.data.datasets import SynchrotronDataset


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array)


def _create_dataset(root: Path, num_samples: int = 3, num_classes: int = 3) -> Path:
    """Create a small synthetic dataset for testing."""
    for split in ("train", "val"):
        for idx in range(num_samples):
            image = np.full((16, 16), fill_value=idx, dtype=np.uint16)
            label = np.full((16, 16), fill_value=idx % num_classes, dtype=np.uint8)

            _write_tiff(root / split / "data" / f"sample_{idx}.tiff", image)
            _write_tiff(root / split / "labels" / f"sample_{idx}.tiff", label)

    return root


def test_synthetic_multiclass_dataset(tmp_path: Path) -> None:
    """SynchrotronDataset should handle multi-class labels without caching issues."""
    dataset_root = _create_dataset(tmp_path / "dataset", num_samples=4, num_classes=3)

    dataset = SynchrotronDataset(
        data_dir=dataset_root,
        split="train",
        cache_size=2,
        normalize=False,
    )

    assert len(dataset) == 4
    assert dataset.multi_class is True
    assert set(dataset.class_values.tolist()) == {0, 1, 2}
    assert len(dataset._cache) == 2  # type: ignore[attr-defined]

    sample = dataset[0]
    assert sample["image"].shape == (1, 16, 16)
    assert sample["mask"].dtype == torch.int64


def test_binary_dataset_masks_are_float(tmp_path: Path) -> None:
    """Binary datasets should expose float32 masks after preprocessing."""
    dataset_root = _create_dataset(tmp_path / "binary", num_samples=2, num_classes=2)

    dataset = SynchrotronDataset(
        data_dir=dataset_root,
        split="val",
        cache_size=0,
        normalize=True,
    )

    sample = dataset[1]
    assert sample["mask"].dtype == torch.float32
    mask_values = torch.unique(sample["mask"], sorted=True).numpy()
    assert set(mask_values.tolist()).issubset({0.0, 1.0})
