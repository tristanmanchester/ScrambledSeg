"""Tests for prediction CLI checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scrambledSeg.prediction import cli as predict_cli
from scrambledSeg.prediction.errors import ModelLoadError
from scrambledSeg.prediction.tiff_utils import TiffInputKind, TiffOutputKind


class FakeSegformer:
    """Minimal model stub for checkpoint-loading tests."""

    def __init__(self) -> None:
        self.expected_state = {"encoder.weight": torch.tensor([1.0])}
        self.load_calls: list[tuple[set[str], bool]] = []

    def load_state_dict(self, state_dict, strict: bool = True):
        self.load_calls.append((set(state_dict.keys()), strict))
        if set(state_dict.keys()) != set(self.expected_state.keys()):
            raise RuntimeError("state_dict mismatch")
        return None

    def state_dict(self):
        return self.expected_state


def test_load_model_strips_optional_model_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should accept checkpoints whose keys are prefixed with ``model.``."""

    model = FakeSegformer()
    load_kwargs: dict[str, object] = {}
    monkeypatch.setattr(predict_cli, "CustomSegformer", lambda **kwargs: model)

    def _fake_load(checkpoint_path, **kwargs):
        load_kwargs.update(kwargs)
        return {
            "state_dict": {"model.encoder.weight": torch.tensor([1.0])},
            "hyper_parameters": {
                "model_config": {
                    "encoder_name": "nvidia/mit-b0",
                    "num_classes": 1,
                    "pretrained": False,
                }
            },
        }

    monkeypatch.setattr(predict_cli.torch, "load", _fake_load)

    loaded_model = predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")

    assert loaded_model is model
    assert model.load_calls == [
        ({"model.encoder.weight"}, True),
        ({"encoder.weight"}, True),
    ]
    assert load_kwargs == {"map_location": "cpu", "weights_only": True}


def test_load_model_rejects_mismatched_checkpoint_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should fail explicitly instead of silently loading an incomplete checkpoint."""

    model = FakeSegformer()
    monkeypatch.setattr(predict_cli, "CustomSegformer", lambda **kwargs: model)
    monkeypatch.setattr(
        predict_cli.torch,
        "load",
        lambda checkpoint_path, **kwargs: {
            "state_dict": {"model.decoder.weight": torch.tensor([1.0])},
            "hyper_parameters": {
                "model_config": {
                    "encoder_name": "nvidia/mit-b0",
                    "num_classes": 1,
                    "pretrained": False,
                }
            },
        },
    )

    with pytest.raises(ModelLoadError, match="Checkpoint state_dict does not match the model"):
        predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")

    assert model.load_calls == [
        ({"model.decoder.weight"}, True),
        ({"decoder.weight"}, True),
    ]


def test_load_model_uses_checkpoint_model_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint metadata should drive model reconstruction instead of hard-coded defaults."""

    model = FakeSegformer()
    captured_model_kwargs: dict[str, object] = {}

    def _fake_model_factory(**kwargs):
        captured_model_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(predict_cli, "CustomSegformer", _fake_model_factory)
    monkeypatch.setattr(
        predict_cli.torch,
        "load",
        lambda checkpoint_path, **kwargs: {
            "state_dict": {"encoder.weight": torch.tensor([1.0])},
            "hyper_parameters": {
                "model_config": {
                    "encoder_name": "nvidia/mit-b0",
                    "encoder_revision": "deadbeef",
                    "num_classes": 2,
                    "pretrained": False,
                }
            },
        },
    )

    loaded_model = predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")

    assert loaded_model is model
    assert captured_model_kwargs == {
        "encoder_name": "nvidia/mit-b0",
        "encoder_revision": "deadbeef",
        "num_classes": 2,
        "pretrained": False,
    }


def test_load_model_requires_checkpoint_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint loading should fail when required model metadata is absent."""

    monkeypatch.setattr(predict_cli.torch, "load", lambda checkpoint_path, **kwargs: {"state_dict": {}})

    with pytest.raises(ModelLoadError, match="hyper_parameters.model_config"):
        predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")


def test_build_argument_parser_accepts_dataset_path_aliases() -> None:
    """The CLI should normalize old and new dataset-path flags onto one field."""

    parser = predict_cli.build_argument_parser()

    renamed_args = parser.parse_args(
        ["input.h5", "checkpoint.ckpt", "--dataset-path", "/seg"]
    )
    legacy_args = parser.parse_args(["input.h5", "checkpoint.ckpt", "--data_path", "/legacy"])

    assert renamed_args.dataset_path == "/seg"
    assert legacy_args.dataset_path == "/legacy"


def test_build_argument_parser_prefers_hyphenated_prediction_flags() -> None:
    """Canonical CLI flags should use hyphenated spellings while preserving legacy aliases."""

    parser = predict_cli.build_argument_parser()

    renamed_args = parser.parse_args(
        [
            "input.h5",
            "checkpoint.ckpt",
            "--tile-size",
            "256",
            "--tile-overlap",
            "16",
            "--output-dir",
            "predictions",
            "--batch-size",
            "4",
        ]
    )
    legacy_args = parser.parse_args(
        [
            "input.h5",
            "checkpoint.ckpt",
            "--tile_size",
            "128",
            "--tile_overlap",
            "8",
            "--output_dir",
            "legacy",
            "--batch_size",
            "2",
        ]
    )

    assert renamed_args.tile_size == 256
    assert renamed_args.tile_overlap == 16
    assert renamed_args.output_dir == "predictions"
    assert renamed_args.batch_size == 4
    assert legacy_args.tile_size == 128
    assert legacy_args.tile_overlap == 8
    assert legacy_args.output_dir == "legacy"
    assert legacy_args.batch_size == 2


def test_process_h5_file_dispatches_to_volume_predictor() -> None:
    """The H5 helper should call the dedicated volume prediction API."""

    calls: list[tuple[object, ...]] = []
    predictor = SimpleNamespace(
        predict_volume=lambda input_path, output_path, dataset_path: calls.append(
            ("predict_volume", input_path, output_path, dataset_path)
        ),
    )

    predict_cli.process_h5_file(Path("volume.h5"), Path("output.h5"), predictor, "/seg")

    assert calls == [("predict_volume", Path("volume.h5"), Path("output.h5"), "/seg")]


def test_process_tiff_file_dispatches_to_canonical_tiff_predictor() -> None:
    """The TIFF helper should call the canonical TIFF prediction API."""

    calls: list[tuple[object, ...]] = []
    predictor = SimpleNamespace(
        predict_tiff=lambda input_path, output_path, input_kind, output_kind: calls.append(
            ("predict_tiff", input_path, output_path, input_kind, output_kind)
        ),
    )

    predict_cli.process_tiff_file(Path("image.tif"), Path("output.h5"), predictor)

    assert calls == [
        (
            "predict_tiff",
            Path("image.tif"),
            Path("output.h5"),
            TiffInputKind.AUTO,
            TiffOutputKind.LABELS,
        )
    ]


def test_process_tiff_file_dispatches_to_explicit_stack_contract() -> None:
    """An explicit TIFF stack contract should pass stack mode to the canonical API."""

    calls: list[tuple[object, ...]] = []
    predictor = SimpleNamespace(
        predict_tiff=lambda input_path, output_path, input_kind, output_kind: calls.append(
            ("predict_tiff", input_path, output_path, input_kind, output_kind)
        ),
    )

    predict_cli.process_tiff_file(
        Path("stack.tif"),
        Path("output.h5"),
        predictor,
        tiff_input_kind=TiffInputKind.STACK.value,
        tiff_output_kind=TiffOutputKind.PROBABILITIES.value,
    )

    assert calls == [
        (
            "predict_tiff",
            Path("stack.tif"),
            Path("output.h5"),
            TiffInputKind.STACK,
            TiffOutputKind.PROBABILITIES,
        )
    ]
