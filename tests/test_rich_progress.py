"""Direct tests for the rich training progress callback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scrambledSeg.training.rich_progress import RichProgressCallback


def test_rich_progress_callback_collects_scalar_metrics() -> None:
    """Scalar callback metrics should be split into train and validation buckets."""

    callback = RichProgressCallback()
    trainer = SimpleNamespace(
        callback_metrics={
            "train_loss": torch.tensor(1.25),
            "val_iou": 0.75,
            "ignored_histogram": torch.tensor([1.0, 2.0]),
        }
    )

    callback._update_metrics(trainer)

    assert callback.train_metrics == {"train_loss": 1.25}
    assert callback.val_metrics == {"val_iou": 0.75}


def test_rich_progress_layout_handles_uninitialized_progress() -> None:
    """The fallback layout should render even before training starts."""

    callback = RichProgressCallback()

    layout = callback._get_layout()

    assert layout.title == "Training Progress"
