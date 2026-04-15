"""Tests for volume prediction behavior."""

from __future__ import annotations

from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from scrambledSeg.prediction.axis import Axis
from scrambledSeg.prediction.predictor import PredictionMode, Predictor


class ConstantLogitsModel(torch.nn.Module):
    """Return fixed multi-class logits for every pixel."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        return self.logits.view(1, -1, 1, 1).expand(batch_size, -1, height, width)


def test_predict_axis_returns_channel_aware_multiclass_accumulators() -> None:
    """Multi-class axis prediction should return a channel-aware volume."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 1.0, 2.0]),
        prediction_mode=PredictionMode.SINGLE_AXIS,
        batch_size=2,
        device="cpu",
        precision="32",
    )
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

    output, counts = predictor._predict_axis(volume, Axis.XY, rotation_angle=0)

    expected_probs = np.array(torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0).tolist(), dtype=np.float32)

    assert output.shape == (2, 3, 3, 4)
    assert counts.shape == output.shape
    assert np.allclose(output[0, :, 0, 0], expected_probs)
    assert np.allclose(counts[0, :, 0, 0], np.ones(3, dtype=np.float32))


def test_predict_volume_handles_multiclass_three_axis_non_cubic_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three-axis volume prediction should not crash on multi-class non-cubic volumes."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 1.0, 2.0]),
        prediction_mode=PredictionMode.THREE_AXIS,
        batch_size=2,
        device="cpu",
        precision="32",
    )
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    captured: dict[str, object] = {}

    monkeypatch.setattr(predictor.data_handler, "load_h5", lambda path, dataset_path: volume)

    def _capture_save(data, output_path, dataset_path) -> None:
        captured["data"] = data
        captured["output_path"] = output_path
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(predictor.data_handler, "save_h5", _capture_save)

    predictor.predict_volume("input.h5", "output.h5", "/seg")

    saved = captured["data"]
    assert isinstance(saved, np.ndarray)
    assert saved.shape == (2, 3, 3, 4)
    assert np.allclose(saved.sum(axis=1), 1.0)
    assert captured["dataset_path"] == "/seg"
