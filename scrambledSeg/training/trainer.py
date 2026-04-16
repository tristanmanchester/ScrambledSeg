"""Training module for SegFormer model."""

import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..visualization.callbacks import VisualizationCallback
from ..visualization.core import SegmentationVisualizer
from .config import ModelConfigSnapshot, PredictionParams, VisualizationConfig

logger = logging.getLogger(__name__)


class OutputStats(TypedDict):
    """Per-batch output summary statistics."""

    out_min: float
    out_max: float
    out_mean: float
    out_std: float
    unique_values: int
    zeros_pct: float


class TrainingProgressMetrics(TypedDict):
    """Performance counters emitted during training."""

    batch_time: float
    samples_per_second: float
    learning_rate: float
    gradient_norm: float
    avg_gradient_norm: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    cpu_percent: float
    total_samples_seen: int


class PerClassMetrics(TypedDict):
    """Per-class metric tensors returned from a step."""

    iou: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor


class ValidationLossRecord(TypedDict):
    """Cached validation loss for epoch-end aggregation."""

    val_loss: torch.Tensor


class TrainingStepOutput(TypedDict):
    """Structured training-step payload consumed by callbacks/tests."""

    loss: torch.Tensor
    train_iou: torch.Tensor
    train_precision: torch.Tensor
    train_recall: torch.Tensor
    train_f1: torch.Tensor
    train_dice: torch.Tensor
    train_specificity: torch.Tensor
    output_stats: OutputStats
    pred_stats: dict[str, float]
    training_metrics: TrainingProgressMetrics
    per_class_metrics: PerClassMetrics


class ValidationStepOutput(TypedDict):
    """Structured validation-step payload consumed by callbacks/tests."""

    val_loss: torch.Tensor
    val_iou: torch.Tensor
    val_precision: torch.Tensor
    val_recall: torch.Tensor
    val_f1: torch.Tensor
    val_dice: torch.Tensor
    val_specificity: torch.Tensor
    per_class_metrics: PerClassMetrics


BatchTensors = dict[str, torch.Tensor]


class SegformerTrainer(pl.LightningModule):
    """Trainer for SegFormer model."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        threshold_params: PredictionParams | None = None,
        vis_dir: str | None = None,
        enable_adaptive_batch_size: bool = False,
        target_gpu_util: float = 0.9,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        num_epochs: int = 100,
        gradient_clip_val: float = 1.0,
        visualization: VisualizationConfig | None = None,
        log_dir: str = "logs",
        test_mode: bool = False,
        num_classes: int = 2,
        model_config: ModelConfigSnapshot | None = None,
    ):
        """Initialize trainer."""
        super().__init__()

        # Store parameters
        self.model = model
        self.criterion = criterion
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.enable_adaptive_batch_size = enable_adaptive_batch_size
        self.target_gpu_util = target_gpu_util
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.test_mode = test_mode
        self.validation_step_outputs: list[ValidationLossRecord] = []
        self.model_config = model_config or ModelConfigSnapshot(num_classes=num_classes)
        self.prediction_params = threshold_params or PredictionParams()

        self.log_dir = Path(log_dir)
        if test_mode:
            self.log_dir = self.log_dir / "test"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = Path(vis_dir) if vis_dir else self.log_dir / "plots"
        self.metrics_dir = self.log_dir / "metrics"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        vis_config = visualization or VisualizationConfig()
        self.vis_callback = VisualizationCallback(
            output_dir=str(self.vis_dir),
            metrics_dir=str(self.metrics_dir),
            num_samples=vis_config.num_samples,
            min_coverage=vis_config.min_coverage,
            dpi=vis_config.dpi,
            num_classes=num_classes,
            visualizer=SegmentationVisualizer(
                metrics_file=str(self.metrics_dir / "metrics.csv"),
                min_coverage=vis_config.min_coverage,
                style=vis_config.style,
                dpi=vis_config.dpi,
                cmap=vis_config.cmap,
            ),
        )

        self.train_metrics = self._build_metric_collection(num_classes)
        self.val_metrics = self._build_metric_collection(num_classes)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        self.class_names: list[str] = [f"Class_{i}" for i in range(num_classes)]
        self.batch_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.total_training_samples = 0

        if not test_mode:
            logger.info("Saving hyperparameters...")
            save_params = {
                "enable_adaptive_batch_size": enable_adaptive_batch_size,
                "target_gpu_util": target_gpu_util,
                "min_batch_size": min_batch_size,
                "max_batch_size": max_batch_size,
                "num_epochs": num_epochs,
                "gradient_clip_val": gradient_clip_val,
                "test_mode": test_mode,
                "model_config": asdict(self.model_config),
                "vis_dir": str(self.vis_dir),
            }
            self.save_hyperparameters(save_params)
        else:
            logger.info("Test mode: Skipping hyperparameter logging")

    @staticmethod
    def _build_metric_collection(num_classes: int) -> nn.ModuleDict:
        """Create a phase-local metric collection."""
        from torchmetrics.segmentation import DiceScore

        return nn.ModuleDict(
            {
                "iou": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "dice": DiceScore(num_classes=num_classes, average="macro", input_format="index"),
                "specificity": torchmetrics.Specificity(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "iou_per_class": torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "precision_per_class": torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "recall_per_class": torchmetrics.Recall(
                    task="multiclass", num_classes=num_classes, average=None
                ),
                "f1_per_class": torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes, average=None
                ),
            }
        )

    @staticmethod
    def _reset_metric_collection(metrics: nn.ModuleDict) -> None:
        """Reset all metrics in a collection."""
        for metric in metrics.values():
            metric.reset()

    @staticmethod
    def _compute_phase_metrics(
        metrics: nn.ModuleDict,
        pred_labels: torch.Tensor,
        masks_for_metrics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Update and return the current metric values for a phase."""
        return {
            "iou": metrics["iou"](pred_labels, masks_for_metrics),
            "precision": metrics["precision"](pred_labels, masks_for_metrics),
            "recall": metrics["recall"](pred_labels, masks_for_metrics),
            "f1": metrics["f1"](pred_labels, masks_for_metrics),
            "dice": metrics["dice"](pred_labels, masks_for_metrics),
            "specificity": metrics["specificity"](pred_labels, masks_for_metrics),
            "iou_per_class": metrics["iou_per_class"](pred_labels, masks_for_metrics),
            "precision_per_class": metrics["precision_per_class"](pred_labels, masks_for_metrics),
            "recall_per_class": metrics["recall_per_class"](pred_labels, masks_for_metrics),
            "f1_per_class": metrics["f1_per_class"](pred_labels, masks_for_metrics),
        }

    def get_predicted_labels(
        self,
        predictions: torch.Tensor,
        _targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get predicted class labels using argmax and optional cleanup.

        Args:
            predictions: Model predictions (B, C, H, W)

        Returns:
            Class prediction tensor (B, H, W) with integer class labels
        """
        if predictions.ndim != 4:
            raise ValueError(f"Expected 4D predictions tensor, got shape {predictions.shape}")

        if predictions.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            pred_probs = F.softmax(predictions, dim=1)
            pred_classes = torch.argmax(pred_probs, dim=1)
        else:
            pred_classes = torch.argmax(predictions, dim=1)

        pred_classes = pred_classes.long()

        num_classes = predictions.size(1)
        pred_one_hot = F.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()

        if self.prediction_params.enable_cleanup:
            kernel_size = self.prediction_params.cleanup_kernel_size
            cleanup_threshold = self.prediction_params.cleanup_threshold

            kernel_size = max(3, kernel_size + (kernel_size + 1) % 2)
            pad_size = kernel_size // 2

            cleanup_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=predictions.device)

            cleaned_pred_one_hot = torch.zeros_like(pred_one_hot)

            for c in range(num_classes):
                class_mask = pred_one_hot[:, c : c + 1, :, :]
                padded_mask = F.pad(
                    class_mask, (pad_size, pad_size, pad_size, pad_size), mode="constant", value=0
                )
                connected = F.conv2d(padded_mask, cleanup_kernel, padding=0)

                auto_threshold = (kernel_size * kernel_size) / 2
                final_threshold = min(cleanup_threshold, auto_threshold)

                cleaned_mask = torch.where(
                    connected >= final_threshold, class_mask, torch.zeros_like(class_mask)
                )

                cleaned_pred_one_hot[:, c : c + 1, :, :] = cleaned_mask

            cleaned_sum = torch.sum(cleaned_pred_one_hot, dim=1, keepdim=True)
            valid_mask = (cleaned_sum > 0).squeeze(1)

            cleaned_classes = torch.argmax(cleaned_pred_one_hot, dim=1)

            cleaned_classes = cleaned_classes.to(pred_classes.dtype)
            final_pred = torch.where(valid_mask, cleaned_classes, pred_classes)

            return final_pred.long()

        return pred_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: BatchTensors, batch_idx: int) -> TrainingStepOutput | None:
        """Training step with enhanced monitoring."""
        batch_start_time = time.time()

        images = batch["image"]
        masks = batch["mask"]
        batch_size = images.size(0)
        self.total_training_samples += batch_size

        if masks.dtype == torch.long:
            mask_coverage = (masks > 0).float().mean().item()
        elif masks.dtype == torch.bool:
            mask_coverage = masks.float().mean().item()
        else:
            mask_coverage = masks.mean().item()

        img_stats = {
            "img_min": images.min().item(),
            "img_max": images.max().item(),
            "img_mean": images.mean().item(),
            "mask_coverage": mask_coverage,
        }
        self.log_dict({f"input/{k}": v for k, v in img_stats.items()}, on_step=True)

        if torch.isnan(images).any() or torch.isnan(masks).any():
            logger.error("NaN detected in input tensors")
            logger.error(f"Input stats: {img_stats}")
            self.trainer.should_stop = True
            return None

        outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if outputs.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            output_stats = {
                "out_min": outputs.min().item(),
                "out_max": outputs.max().item(),
                "out_mean": outputs.mean().item(),
                "out_std": outputs.std().item(),
                "unique_values": len(torch.unique(outputs)),
                "zeros_pct": (outputs == 0).float().mean().item() * 100,
            }
        else:
            outputs_float = outputs.float()
            output_stats = {
                "out_min": outputs_float.min().item(),
                "out_max": outputs_float.max().item(),
                "out_mean": outputs_float.mean().item(),
                "out_std": outputs_float.std().item(),
                "unique_values": len(torch.unique(outputs)),
                "zeros_pct": (outputs == 0).float().mean().item() * 100,
            }
        self.log_dict({f"output/{k}": v for k, v in output_stats.items()}, on_step=True)

        if output_stats.get("unique_values", 20) < 10:
            logger.warning(f"Low variance in outputs: {output_stats}")
        if output_stats.get("zeros_pct", 0) > 90:
            logger.warning(f"Output mostly zeros: {output_stats}")

        if torch.isnan(outputs).any():
            logger.error("NaN detected in model outputs")
            logger.error(f"Output stats: {output_stats}")
            self.trainer.should_stop = True
            return None

        loss = self.criterion(outputs, masks)
        loss_stats = {"loss_value": loss.item()}
        self.log_dict({f"loss/{k}": v for k, v in loss_stats.items()}, on_step=True)

        if torch.isnan(loss).any():
            logger.error("NaN detected in loss calculation")
            logger.error(f"Loss stats: {loss_stats}")
            self.trainer.should_stop = True
            return None

        pred_labels = self.get_predicted_labels(outputs, masks)

        masks_for_metrics = masks.clone()
        if masks_for_metrics.dim() == 4 and masks_for_metrics.size(1) == 1:
            masks_for_metrics = masks_for_metrics.squeeze(1)

        if masks_for_metrics.dtype != torch.long:
            if masks_for_metrics.dtype == torch.bool:
                masks_for_metrics = masks_for_metrics.long()
            else:
                masks_for_metrics = torch.round(masks_for_metrics).long()

        metric_values = self._compute_phase_metrics(
            self.train_metrics, pred_labels, masks_for_metrics
        )
        iou = metric_values["iou"]
        precision = metric_values["precision"]
        recall = metric_values["recall"]
        f1 = metric_values["f1"]
        dice = metric_values["dice"]
        specificity = metric_values["specificity"]
        iou_per_class = metric_values["iou_per_class"]
        precision_per_class = metric_values["precision_per_class"]
        recall_per_class = metric_values["recall_per_class"]
        f1_per_class = metric_values["f1_per_class"]

        batch_time = time.time() - batch_start_time
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0

        current_lr = self._optimizer.param_groups[0]["lr"] if self._optimizer else 0.0

        total_grad_norm = 0.0
        param_count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
                param_count += param.numel()

        total_grad_norm = (total_grad_norm**0.5) if total_grad_norm > 0 else 0.0
        avg_grad_norm = total_grad_norm / max(param_count, 1)

        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.memory_reserved() / 1024**3

        cpu_percent = psutil.cpu_percent()

        training_metrics = {
            "batch_time": batch_time,
            "samples_per_second": samples_per_second,
            "learning_rate": current_lr,
            "gradient_norm": total_grad_norm,
            "avg_gradient_norm": avg_grad_norm,
            "gpu_memory_used_gb": gpu_memory_used,
            "gpu_memory_total_gb": gpu_memory_total,
            "cpu_percent": cpu_percent,
            "total_samples_seen": self.total_training_samples,
        }

        self.log_dict({f"progress/{k}": v for k, v in training_metrics.items()}, on_step=True)

        num_classes = outputs.size(1)
        pred_labels_float = pred_labels.float()

        pred_stats = {
            "pred_coverage": (pred_labels_float > 0).float().mean().item(),
        }

        for c in range(num_classes):
            pred_stats[f"class_{c}_pct"] = (pred_labels == c).float().mean().item() * 100
        self.log_dict({f"pred/{k}": v for k, v in pred_stats.items()}, on_step=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_precision", precision, on_step=True, on_epoch=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True)
        self.log("train_dice", dice, on_step=True, on_epoch=True)
        self.log("train_specificity", specificity, on_step=True, on_epoch=True)

        for i, class_name in enumerate(self.class_names):
            if i < len(iou_per_class):
                self.log(
                    f"train_iou_{class_name.lower()}",
                    iou_per_class[i],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"train_precision_{class_name.lower()}",
                    precision_per_class[i],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"train_recall_{class_name.lower()}",
                    recall_per_class[i],
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"train_f1_{class_name.lower()}", f1_per_class[i], on_step=False, on_epoch=True
                )

        return {
            "loss": loss,
            "train_iou": iou,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
            "train_dice": dice,
            "train_specificity": specificity,
            "output_stats": output_stats,
            "pred_stats": pred_stats,
            "training_metrics": training_metrics,
            "per_class_metrics": {
                "iou": iou_per_class,
                "precision": precision_per_class,
                "recall": recall_per_class,
                "f1": f1_per_class,
            },
        }

    def validation_step(self, batch: BatchTensors, batch_idx: int) -> ValidationStepOutput:
        """Validation step."""
        images = batch["image"]
        masks = batch["mask"]

        outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        masks_for_loss = masks.clone()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or hasattr(
            self.criterion, "ce_criterion"
        ):
            if masks_for_loss.dim() == 4 and masks_for_loss.size(1) == 1:
                masks_for_loss = masks_for_loss.squeeze(1)

            if masks_for_loss.dtype != torch.long:
                masks_for_loss = masks_for_loss.long()

        try:
            loss = self.criterion(outputs, masks_for_loss)
        except RuntimeError as e:
            logger.error(f"Error in validation loss calculation: {e}")
            logger.error(f"Output shape: {outputs.shape}, dtype: {outputs.dtype}")
            logger.error(f"Masks shape: {masks_for_loss.shape}, dtype: {masks_for_loss.dtype}")
            logger.error(f"Unique mask values: {torch.unique(masks_for_loss)}")
            raise

        pred_labels = self.get_predicted_labels(outputs, masks)

        masks_for_metrics = masks.clone()
        if masks_for_metrics.dim() == 4 and masks_for_metrics.size(1) == 1:
            masks_for_metrics = masks_for_metrics.squeeze(1)

        if masks_for_metrics.dtype != torch.long:
            if masks_for_metrics.dtype == torch.bool:
                masks_for_metrics = masks_for_metrics.long()
            else:
                masks_for_metrics = torch.round(masks_for_metrics).long()

        metric_values = self._compute_phase_metrics(
            self.val_metrics, pred_labels, masks_for_metrics
        )
        iou = metric_values["iou"]
        precision = metric_values["precision"]
        recall = metric_values["recall"]
        f1 = metric_values["f1"]
        dice = metric_values["dice"]
        specificity = metric_values["specificity"]
        iou_per_class = metric_values["iou_per_class"]
        precision_per_class = metric_values["precision_per_class"]
        recall_per_class = metric_values["recall_per_class"]
        f1_per_class = metric_values["f1_per_class"]

        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.val_confusion_matrix.update(pred_labels, masks_for_metrics)

        self.validation_step_outputs.append({"val_loss": loss.detach()})

        return {
            "val_loss": loss,
            "val_iou": iou,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_dice": dice,
            "val_specificity": specificity,
            "per_class_metrics": {
                "iou": iou_per_class,
                "precision": precision_per_class,
                "recall": recall_per_class,
                "f1": f1_per_class,
            },
        }

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        if self.validation_step_outputs:
            val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        else:
            val_loss = torch.tensor(0.0, device=self.device)

        val_iou = self.val_metrics["iou"].compute()
        val_precision = self.val_metrics["precision"].compute()
        val_recall = self.val_metrics["recall"].compute()
        val_f1 = self.val_metrics["f1"].compute()
        val_dice = self.val_metrics["dice"].compute()
        val_specificity = self.val_metrics["specificity"].compute()
        iou_per_class = self.val_metrics["iou_per_class"].compute()
        precision_per_class = self.val_metrics["precision_per_class"].compute()
        recall_per_class = self.val_metrics["recall_per_class"].compute()
        f1_per_class = self.val_metrics["f1_per_class"].compute()

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_iou", val_iou, prog_bar=True, sync_dist=True)
        self.log("val_precision", val_precision, prog_bar=False, sync_dist=True)
        self.log("val_recall", val_recall, prog_bar=False, sync_dist=True)
        self.log("val_f1", val_f1, prog_bar=False, sync_dist=True)
        self.log("val_dice", val_dice, prog_bar=False, sync_dist=True)
        self.log("val_specificity", val_specificity, prog_bar=False, sync_dist=True)
        self.log("val_loss_epoch", val_loss, prog_bar=True)
        self.log("val_iou_epoch", val_iou, prog_bar=True)
        self.log("val_precision_epoch", val_precision, prog_bar=True)
        self.log("val_recall_epoch", val_recall, prog_bar=True)
        self.log("val_f1_epoch", val_f1, prog_bar=True)
        self.log("val_dice_epoch", val_dice, prog_bar=True)
        self.log("val_specificity_epoch", val_specificity, prog_bar=True)

        for i, class_name in enumerate(self.class_names):
            if i < len(iou_per_class):
                self.log(
                    f"val_iou_{class_name.lower()}",
                    iou_per_class[i],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"val_precision_{class_name.lower()}",
                    precision_per_class[i],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"val_recall_{class_name.lower()}",
                    recall_per_class[i],
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    f"val_f1_{class_name.lower()}", f1_per_class[i], prog_bar=False, sync_dist=True
                )

        cm = self.val_confusion_matrix.compute()
        if cm is not None and cm.numel() > 0:
            self.log("confusion_matrix_trace", torch.trace(cm), prog_bar=False)
            self._log_confusion_matrix_analysis(cm)
            self._save_confusion_matrix(cm, self.current_epoch)
            self.val_confusion_matrix.reset()

        self.validation_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        """Clear the validation outputs at the start of each validation epoch."""
        self.validation_step_outputs = []
        self._reset_metric_collection(self.val_metrics)
        self.val_confusion_matrix.reset()

    def on_train_epoch_start(self) -> None:
        """Track epoch start time and reset counters."""
        self.epoch_start_time = time.time()
        self.total_training_samples = 0
        self._reset_metric_collection(self.train_metrics)

    def on_train_epoch_end(self) -> None:
        """Log epoch-level training metrics."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.log("epoch_time_minutes", epoch_time / 60.0, on_epoch=True)
            self.log("samples_per_epoch", self.total_training_samples, on_epoch=True)

            if epoch_time > 0:
                self.log(
                    "epoch_samples_per_second",
                    self.total_training_samples / epoch_time,
                    on_epoch=True,
                )

    def _log_confusion_matrix_analysis(self, cm: torch.Tensor):
        """Log detailed confusion matrix analysis."""
        try:
            num_classes = cm.size(0)

            class_accuracies = torch.diag(cm) / torch.sum(cm, dim=1)

            true_positives = torch.diag(cm)
            false_positives = torch.sum(cm, dim=0) - true_positives
            false_negatives = torch.sum(cm, dim=1) - true_positives

            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            for i, class_name in enumerate(self.class_names):
                if i < num_classes:
                    self.log(
                        f"cm_accuracy_{class_name.lower()}", class_accuracies[i], prog_bar=False
                    )
                    self.log(f"cm_precision_{class_name.lower()}", precision[i], prog_bar=False)
                    self.log(f"cm_recall_{class_name.lower()}", recall[i], prog_bar=False)
                    self.log(f"cm_f1_{class_name.lower()}", f1[i], prog_bar=False)

            overall_accuracy = torch.trace(cm) / torch.sum(cm)
            macro_precision = torch.mean(precision)
            macro_recall = torch.mean(recall)
            macro_f1 = torch.mean(f1)

            class_support = torch.sum(cm, dim=1)
            weighted_precision = torch.sum(precision * class_support) / torch.sum(class_support)
            weighted_recall = torch.sum(recall * class_support) / torch.sum(class_support)
            weighted_f1 = torch.sum(f1 * class_support) / torch.sum(class_support)

            self.log("cm_overall_accuracy", overall_accuracy, prog_bar=False)
            self.log("cm_macro_precision", macro_precision, prog_bar=False)
            self.log("cm_macro_recall", macro_recall, prog_bar=False)
            self.log("cm_macro_f1", macro_f1, prog_bar=False)
            self.log("cm_weighted_precision", weighted_precision, prog_bar=False)
            self.log("cm_weighted_recall", weighted_recall, prog_bar=False)
            self.log("cm_weighted_f1", weighted_f1, prog_bar=False)

            class_distribution = class_support / torch.sum(class_support)
            entropy = -torch.sum(class_distribution * torch.log(class_distribution + 1e-8))
            max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float))
            normalized_entropy = entropy / max_entropy

            self.log("cm_class_entropy", entropy, prog_bar=False)
            self.log("cm_normalized_entropy", normalized_entropy, prog_bar=False)

        except Exception as e:
            logger.error(f"Error in confusion matrix analysis: {str(e)}")

    def _save_confusion_matrix(self, cm: torch.Tensor, epoch: int):
        """Save confusion matrix to file for later analysis."""
        try:
            import json

            cm_dir = self.log_dir / "confusion_matrices"
            cm_dir.mkdir(parents=True, exist_ok=True)

            cm_np = cm.detach().cpu().numpy()

            cm_data = {
                "epoch": epoch,
                "matrix": cm_np.tolist(),
                "class_names": self.class_names,
                "total_samples": int(torch.sum(cm).item()),
                "timestamp": datetime.now().isoformat(),
            }

            cm_file = cm_dir / f"confusion_matrix_epoch_{epoch:03d}.json"
            with open(cm_file, "w") as f:
                json.dump(cm_data, f, indent=2)

            latest_file = cm_dir / "confusion_matrix_latest.json"
            with open(latest_file, "w") as f:
                json.dump(cm_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving confusion matrix: {str(e)}")

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        if self._scheduler is None:
            return self._optimizer
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {"scheduler": self._scheduler, "interval": "step"},
        }

    def train_dataloader(self):
        """Return training dataloader."""
        return self._train_dataloader

    def val_dataloader(self):
        """Return validation dataloader."""
        return self._val_dataloader
