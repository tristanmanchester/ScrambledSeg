"""Tests for visualization callback metric logging."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from scrambledSeg.visualization.callbacks import VisualizationCallback


class StubVisualizer:
    """Minimal visualizer for callback tests."""

    def __init__(self) -> None:
        self.metrics_file = None

    def plot_metrics(self, window_size: int, save_path: str) -> None:
        return None

    def plot_comprehensive_metrics(self, window_size: int, save_path: str) -> None:
        return None

    def plot_per_class_metrics(self, window_size: int, save_path: str) -> None:
        return None

    def find_interesting_slices(self, masks, num_samples: int):
        return []

    def visualize_prediction(self, image, mask, prediction, save_path: str, class_values=None) -> None:
        return None


def _build_callback(tmp_path: Path, num_classes: int = 3) -> VisualizationCallback:
    return VisualizationCallback(
        output_dir=str(tmp_path / "visualizations"),
        metrics_dir=str(tmp_path / "logs"),
        visualizer=StubVisualizer(),
        num_classes=num_classes,
    )


def test_callback_logs_full_train_and_validation_metrics(tmp_path: Path) -> None:
    """The callback should persist comprehensive scalar and per-class metrics."""

    callback = _build_callback(tmp_path, num_classes=3)

    train_trainer = SimpleNamespace(current_epoch=2, global_step=1, is_global_zero=True)
    train_outputs = {
        "loss": torch.tensor(1.25),
        "train_iou": torch.tensor(0.41),
        "train_precision": "tensor(0.52)",
        "train_recall": np.float32(0.63),
        "train_f1": torch.tensor(0.57),
        "train_dice": 0.6,
        "train_specificity": "0.91",
        "per_class_metrics": {
            "iou": torch.tensor([0.11, 0.22, 0.33]),
            "precision": [0.14, 0.25, 0.36],
            "recall": np.array([0.15, 0.26, 0.37], dtype=np.float32),
            "f1": ["tensor(0.16)", "tensor(0.27, device='cuda:0')", "0.38"],
        },
    }

    callback.on_train_batch_end(
        trainer=train_trainer,
        pl_module=SimpleNamespace(),
        outputs=train_outputs,
        batch=None,
        batch_idx=0,
    )

    val_trainer = SimpleNamespace(
        current_epoch=2,
        global_step=1,
        is_global_zero=True,
        callback_metrics={
            "val_loss": torch.tensor(0.95),
            "val_iou": torch.tensor(0.44),
            "val_precision": "tensor(0.61)",
            "val_recall": torch.tensor(0.62),
            "val_f1": torch.tensor(0.63),
            "val_dice": np.float32(0.64),
            "val_specificity": "0.94",
            "val_iou_class_0": torch.tensor(0.21),
            "val_iou_class_1": "tensor(0.31)",
            "val_iou_class_2": torch.tensor(0.41),
            "val_precision_class_0": torch.tensor(0.22),
            "val_precision_class_1": torch.tensor(0.32),
            "val_precision_class_2": torch.tensor(0.42),
            "val_recall_class_0": torch.tensor(0.23),
            "val_recall_class_1": torch.tensor(0.33),
            "val_recall_class_2": torch.tensor(0.43),
            "val_f1_class_0": torch.tensor(0.24),
            "val_f1_class_1": "tensor(0.34)",
            "val_f1_class_2": torch.tensor(0.44),
            "val_precision_epoch": torch.tensor(9.99),
        },
    )

    callback.on_validation_epoch_end(
        trainer=val_trainer,
        pl_module=SimpleNamespace(device="cpu"),
    )

    df = pd.read_csv(callback.metrics_file)

    assert len(df) == 2
    train_row = df.iloc[0]
    val_row = df.iloc[1]

    assert train_row["train_loss"] == pytest.approx(1.25)
    assert train_row["train_precision"] == pytest.approx(0.52)
    assert train_row["train_recall_class_2"] == pytest.approx(0.37)
    assert train_row["train_f1_class_1"] == pytest.approx(0.27)

    assert val_row["val_loss"] == pytest.approx(0.95)
    assert val_row["val_precision"] == pytest.approx(0.61)
    assert val_row["val_dice"] == pytest.approx(0.64)
    assert val_row["val_iou_class_1"] == pytest.approx(0.31)
    assert val_row["val_f1_class_1"] == pytest.approx(0.34)

    assert "train_iou_class_2" in df.columns
    assert "val_precision_class_2" in df.columns
    assert "train_iou_class_3" not in df.columns
    assert "val_precision_epoch" not in df.columns


def test_callback_skips_side_effects_for_non_global_zero(tmp_path: Path) -> None:
    """Non-rank-zero trainers should not write metrics artifacts."""

    callback = _build_callback(tmp_path)
    trainer = SimpleNamespace(
        current_epoch=0,
        global_step=1,
        is_global_zero=False,
        callback_metrics={"val_loss": torch.tensor(1.0)},
    )

    callback.on_train_batch_end(
        trainer=trainer,
        pl_module=SimpleNamespace(),
        outputs={"loss": torch.tensor(1.0)},
        batch=None,
        batch_idx=0,
    )
    callback.on_validation_epoch_end(
        trainer=trainer,
        pl_module=SimpleNamespace(device="cpu"),
    )

    assert not callback.metrics_file.exists()
