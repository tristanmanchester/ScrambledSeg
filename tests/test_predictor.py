"""Tests for the prediction utilities."""

from pathlib import Path

import h5py
import numpy as np
import torch
import tifffile

from scrambledSeg.prediction.predictor import EnsembleMethod, PredictionMode, Predictor


class _DummyModel(torch.nn.Module):
    """Simple model that deterministically selects the last class."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch, _, height, width = x.shape
        logits = torch.arange(
            self.num_classes, dtype=x.dtype, device=x.device
        ).view(1, self.num_classes, 1, 1)
        return logits.expand(batch, -1, height, width)


def test_predict_image_multiclass(tmp_path: Path) -> None:
    """Predictor should tile TIFF inputs and return class indices."""
    image = (np.arange(25, dtype=np.uint8).reshape(5, 5))
    input_path = tmp_path / "input.tiff"
    tifffile.imwrite(str(input_path), image)

    predictor = Predictor(
        model=_DummyModel(num_classes=3),
        prediction_mode=PredictionMode.SINGLE_AXIS,
        ensemble_method=EnsembleMethod.MEAN,
        batch_size=2,
        device="cpu",
        tile_size=4,
        tile_overlap=2,
        precision="32",
    )

    output_path = tmp_path / "output.tiff"
    predictor.predict_image(input_path, output_path)

    result = tifffile.imread(str(output_path))
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all(result == 2)


def test_predict_volume_multiclass(tmp_path: Path) -> None:
    """Volume predictions should collapse to discrete class indices."""
    volume = np.arange(32, dtype=np.uint16).reshape(2, 4, 4)
    input_path = tmp_path / "volume.h5"
    with h5py.File(input_path, "w") as f:
        f.create_dataset("/data", data=volume)

    predictor = Predictor(
        model=_DummyModel(num_classes=3),
        prediction_mode=PredictionMode.SINGLE_AXIS,
        ensemble_method=EnsembleMethod.MEAN,
        batch_size=1,
        device="cpu",
        precision="32",
    )

    output_path = tmp_path / "prediction.h5"
    predictor.predict_volume(input_path, output_path)

    with h5py.File(output_path, "r") as f:
        result = f["/data"][:]

    assert result.shape == volume.shape
    assert result.dtype == np.uint8
    assert np.all(result == 2)
