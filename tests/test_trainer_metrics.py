"""Tests for trainer metric state isolation."""

from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from scrambledSeg.training.trainer import SegformerTrainer


class DummySegModel(torch.nn.Module):
    """Minimal model that produces stable multi-class logits."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, num_classes, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            for class_idx in range(num_classes):
                self.conv.weight[class_idx, 0, 0, 0] = float(class_idx + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FakeMetric(torch.nn.Module):
    """A fake metric that tracks updates, resets, and compute calls."""

    def __init__(self, step_value, compute_value=None) -> None:
        super().__init__()
        self.step_value = torch.as_tensor(step_value, dtype=torch.float32)
        self.compute_value = torch.as_tensor(
            self.step_value if compute_value is None else compute_value,
            dtype=torch.float32,
        )
        self.calls = 0
        self.reset_calls = 0

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.step_value.clone()

    def compute(self) -> torch.Tensor:
        return self.compute_value.clone()

    def reset(self) -> None:
        self.reset_calls += 1


class ExplodingMetric(FakeMetric):
    """Metric stub whose epoch-end compute fails."""

    def compute(self) -> torch.Tensor:
        raise RuntimeError("metric explosion")


class FakeConfusionMatrix(torch.nn.Module):
    """A fake confusion matrix metric for validation-only assertions."""

    def __init__(self, matrix) -> None:
        super().__init__()
        self.matrix = torch.as_tensor(matrix, dtype=torch.float32)
        self.updates = 0
        self.reset_calls = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.updates += 1

    def compute(self) -> torch.Tensor:
        return self.matrix.clone()

    def reset(self) -> None:
        self.reset_calls += 1


def _build_metric_collection(prefix: float) -> torch.nn.ModuleDict:
    return torch.nn.ModuleDict(
        {
            "iou": FakeMetric(prefix + 0.1),
            "precision": FakeMetric(prefix + 0.2),
            "recall": FakeMetric(prefix + 0.3),
            "f1": FakeMetric(prefix + 0.4),
            "dice": FakeMetric(prefix + 0.5),
            "specificity": FakeMetric(prefix + 0.6),
            "iou_per_class": FakeMetric([prefix + 0.7, prefix + 0.8], compute_value=[prefix + 1.7, prefix + 1.8]),
            "precision_per_class": FakeMetric([prefix + 0.9, prefix + 1.0], compute_value=[prefix + 1.9, prefix + 2.0]),
            "recall_per_class": FakeMetric([prefix + 1.1, prefix + 1.2], compute_value=[prefix + 2.1, prefix + 2.2]),
            "f1_per_class": FakeMetric([prefix + 1.3, prefix + 1.4], compute_value=[prefix + 2.3, prefix + 2.4]),
        }
    )


def _build_trainer(tmp_path: Path) -> SegformerTrainer:
    model = DummySegModel(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SegformerTrainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        train_dataloader=[],
        val_dataloader=[],
        optimizer=optimizer,
        scheduler=None,
        visualization={},
        log_dir=str(tmp_path),
        test_mode=True,
        num_classes=2,
    )
    trainer.log_dict = lambda *args, **kwargs: None
    trainer.log = lambda *args, **kwargs: None
    return trainer


def _sample_batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.ones((1, 1, 2, 2), dtype=torch.float32),
        "mask": torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
    }


def test_trainer_uses_separate_metric_state_for_train_and_validation(tmp_path: Path) -> None:
    """Training should update only train metrics; validation should update only val metrics."""

    trainer = _build_trainer(tmp_path)
    trainer.train_metrics = _build_metric_collection(10.0)
    trainer.val_metrics = _build_metric_collection(20.0)
    trainer.val_confusion_matrix = FakeConfusionMatrix([[2, 0], [0, 2]])

    batch = _sample_batch()

    trainer.training_step(batch, 0)

    assert trainer.train_metrics["iou"].calls == 1
    assert trainer.val_metrics["iou"].calls == 0
    assert trainer.val_confusion_matrix.updates == 0

    trainer.validation_step(batch, 0)

    assert trainer.train_metrics["iou"].calls == 1
    assert trainer.val_metrics["iou"].calls == 1
    assert trainer.val_confusion_matrix.updates == 1


def test_trainer_resets_phase_metrics_and_reports_validation_confusion_matrix(tmp_path: Path) -> None:
    """Epoch hooks should reset phase-local metrics and log validation-only results."""

    trainer = _build_trainer(tmp_path)
    trainer.train_metrics = _build_metric_collection(1.0)
    trainer.val_metrics = _build_metric_collection(2.0)
    trainer.val_confusion_matrix = FakeConfusionMatrix([[3, 1], [0, 4]])

    logged: dict[str, torch.Tensor] = {}
    trainer.log = lambda name, value, **kwargs: logged.__setitem__(name, value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(value))
    trainer.log_dict = lambda *args, **kwargs: None

    captured: dict[str, torch.Tensor] = {}
    trainer._log_confusion_matrix_analysis = lambda cm: captured.__setitem__("analysis", cm.clone())
    trainer._save_confusion_matrix = lambda cm, epoch: captured.__setitem__("saved", cm.clone())

    trainer.on_train_epoch_start()
    assert trainer.train_metrics["iou"].reset_calls == 1
    assert trainer.val_metrics["iou"].reset_calls == 0

    trainer.on_validation_epoch_start()
    assert trainer.val_metrics["iou"].reset_calls == 1
    assert trainer.val_confusion_matrix.reset_calls == 1

    trainer.validation_step_outputs = [{"val_loss": torch.tensor(1.5)}]
    trainer.on_validation_epoch_end()

    assert logged["val_loss"].item() == pytest.approx(1.5)
    assert logged["val_iou"].item() == pytest.approx(trainer.val_metrics["iou"].compute_value.item())
    assert logged["val_precision"].item() == pytest.approx(trainer.val_metrics["precision"].compute_value.item())
    assert logged["val_iou_class_0"].item() == pytest.approx(trainer.val_metrics["iou_per_class"].compute_value[0].item())
    assert torch.equal(captured["analysis"], trainer.val_confusion_matrix.matrix)
    assert torch.equal(captured["saved"], trainer.val_confusion_matrix.matrix)
    assert trainer.val_confusion_matrix.reset_calls == 2
    assert trainer.validation_step_outputs == []


def test_validation_epoch_end_propagates_metric_compute_failures(tmp_path: Path) -> None:
    """Epoch-end validation metric errors should surface immediately."""

    trainer = _build_trainer(tmp_path)
    trainer.val_metrics = _build_metric_collection(2.0)
    trainer.val_metrics["iou"] = ExplodingMetric(0.0)
    trainer.val_confusion_matrix = FakeConfusionMatrix([[1, 0], [0, 1]])
    trainer.validation_step_outputs = [{"val_loss": torch.tensor(1.5)}]

    with pytest.raises(RuntimeError, match="metric explosion"):
        trainer.on_validation_epoch_end()
