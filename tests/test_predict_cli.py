"""Tests for prediction CLI checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

import predict_cli


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
    monkeypatch.setattr(predict_cli, "CustomSegformer", lambda **kwargs: model)
    monkeypatch.setattr(
        predict_cli.torch,
        "load",
        lambda checkpoint_path, map_location: {
            "state_dict": {"model.encoder.weight": torch.tensor([1.0])}
        },
    )

    loaded_model = predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")

    assert loaded_model is model
    assert model.load_calls == [
        ({"model.encoder.weight"}, True),
        ({"encoder.weight"}, True),
    ]


def test_load_model_rejects_mismatched_checkpoint_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should fail explicitly instead of silently loading an incomplete checkpoint."""

    model = FakeSegformer()
    monkeypatch.setattr(predict_cli, "CustomSegformer", lambda **kwargs: model)
    monkeypatch.setattr(
        predict_cli.torch,
        "load",
        lambda checkpoint_path, map_location: {
            "state_dict": {"model.decoder.weight": torch.tensor([1.0])}
        },
    )

    with pytest.raises(RuntimeError, match="Checkpoint state_dict does not match the model"):
        predict_cli.load_model(Path("checkpoint.ckpt"), "cpu")

    assert model.load_calls == [
        ({"model.decoder.weight"}, True),
        ({"decoder.weight"}, True),
    ]
