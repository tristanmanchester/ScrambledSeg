"""Tests for volume prediction behavior."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
torch = pytest.importorskip("torch")

from scrambledSeg.axis import Axis
from scrambledSeg.prediction.errors import PredictionInputError
from scrambledSeg.prediction.predictor import PredictionMode, Predictor
from scrambledSeg.prediction.tiff_utils import TiffOutputKind


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

    expected_probs = np.array(
        torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0).tolist(), dtype=np.float32
    )

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


def test_predictor_accepts_cli_and_legacy_prediction_mode_spellings() -> None:
    """String-based prediction mode inputs should accept both public spellings."""

    cli_predictor = Predictor(
        model=ConstantLogitsModel([0.0, 1.0, 2.0]),
        prediction_mode="THREE_AXIS",
        batch_size=2,
        device="cpu",
        precision="32",
    )
    legacy_predictor = Predictor(
        model=ConstantLogitsModel([0.0, 1.0, 2.0]),
        prediction_mode="three",
        batch_size=2,
        device="cpu",
        precision="32",
    )

    assert cli_predictor.prediction_mode is PredictionMode.THREE_AXIS
    assert legacy_predictor.prediction_mode is PredictionMode.THREE_AXIS


def test_predictor_preserves_explicit_zero_tile_overlap() -> None:
    """An explicit zero overlap should stay zero instead of falling back to a default."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 1.0, 2.0]),
        tile_overlap=0,
        batch_size=2,
        device="cpu",
        precision="32",
    )

    assert predictor.tiff_handler.overlap == 0


def test_predict_tiff_writes_image_predictions(tmp_path) -> None:
    """Canonical TIFF prediction should save image label outputs."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 2.0]),
        batch_size=2,
        device="cpu",
        tile_size=2,
        tile_overlap=0,
        precision="32",
    )

    image = np.arange(16, dtype=np.uint8).reshape((4, 4))
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    tifffile.imwrite(input_path, image)

    predictor.predict_tiff(input_path, output_path, input_kind="image")

    saved = tifffile.imread(output_path)
    assert saved.shape == image.shape
    assert np.all(saved == 1)


def test_predict_tiff_writes_stack_predictions(tmp_path) -> None:
    """Canonical TIFF prediction should save multipage stack label outputs."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 2.0]),
        batch_size=2,
        device="cpu",
        tile_size=2,
        tile_overlap=0,
        precision="32",
    )

    image = np.arange(32, dtype=np.uint8).reshape((2, 4, 4))
    input_path = tmp_path / "stack.tif"
    output_path = tmp_path / "stack_output.tif"
    tifffile.imwrite(input_path, image)

    predictor.predict_tiff(input_path, output_path, input_kind="stack")

    saved = tifffile.imread(output_path)
    assert saved.shape == image.shape
    assert np.all(saved == 1)


def test_predict_tiff_requires_explicit_kind_for_ambiguous_3d_tiffs(tmp_path) -> None:
    """The canonical TIFF API should reject ambiguous 3D TIFF inputs."""

    predictor = Predictor(
        model=ConstantLogitsModel([0.0, 2.0]),
        batch_size=2,
        device="cpu",
        tile_size=2,
        tile_overlap=0,
        precision="32",
    )

    image = np.arange(32, dtype=np.uint8).reshape((2, 4, 4))
    input_path = tmp_path / "ambiguous.tif"
    output_path = tmp_path / "ambiguous_output.tif"
    tifffile.imwrite(input_path, image)

    with pytest.raises(PredictionInputError, match="Ambiguous 3D TIFF shape"):
        predictor.predict_tiff(input_path, output_path)


def test_predict_tiff_can_save_probability_outputs(tmp_path) -> None:
    """Probability output should preserve floating-point TIFF scores."""

    predictor = Predictor(
        model=ConstantLogitsModel([2.0]),
        batch_size=2,
        device="cpu",
        tile_size=2,
        tile_overlap=0,
        precision="32",
    )

    image = np.arange(16, dtype=np.uint8).reshape((4, 4))
    input_path = tmp_path / "prob_input.tif"
    output_path = tmp_path / "prob_output.tif"
    tifffile.imwrite(input_path, image)

    predictor.predict_tiff(
        input_path,
        output_path,
        input_kind="image",
        output_kind=TiffOutputKind.PROBABILITIES,
    )

    saved = tifffile.imread(output_path)
    assert saved.dtype == np.float32
    assert np.allclose(saved, 0.880797, atol=1e-5)
