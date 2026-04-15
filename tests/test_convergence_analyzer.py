"""Tests for convergence analysis callbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scrambledSeg.analysis.convergence_analyzer import ConvergenceCallback


class _DummyLogger:
    def __init__(self) -> None:
        self.logged = []

    def log_metrics(self, metrics, step) -> None:
        self.logged.append((metrics, step))


def _make_callback(**metric_config) -> ConvergenceCallback:
    return ConvergenceCallback(
        convergence_configs={"loss": {"patience": 2, **metric_config}},
        log_interval=100,
    )


def test_convergence_callback_accepts_tensor_training_step_output() -> None:
    callback = _make_callback()
    trainer = SimpleNamespace(global_step=1, logger=_DummyLogger())

    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=None,
        outputs=torch.tensor(0.75),
        batch=None,
        batch_idx=0,
    )

    detector = callback.analyzer.detectors["loss"]
    assert list(detector.values) == [pytest.approx(0.75)]


def test_convergence_callback_still_handles_dict_outputs() -> None:
    callback = _make_callback()
    trainer = SimpleNamespace(global_step=1, logger=_DummyLogger())

    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=None,
        outputs={"loss": torch.tensor(0.5), "ignored": "value"},
        batch=None,
        batch_idx=0,
    )

    detector = callback.analyzer.detectors["loss"]
    assert list(detector.values) == [pytest.approx(0.5)]
