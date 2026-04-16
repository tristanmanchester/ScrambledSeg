"""Callbacks for visualization during training."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .core import SegmentationVisualizer

logger = logging.getLogger(__name__)

MetricScalar = torch.Tensor | np.generic | np.ndarray | int | float | str
MetricSequence = torch.Tensor | np.ndarray | list[MetricScalar] | tuple[MetricScalar, ...]
PerClassMetricMap = dict[str, MetricSequence | None]
MetricValue = MetricScalar | PerClassMetricMap | None
FlatMetricValue = MetricScalar | None
BatchTensors = dict[str, torch.Tensor]
ValidationDataloader = DataLoader | list[DataLoader] | None


class VisualizationCallback(pl.Callback):
    """Callback for visualizing segmentation results."""

    SCALAR_METRICS = ("loss", "iou", "precision", "recall", "f1", "dice", "specificity")
    PER_CLASS_METRICS = ("iou", "precision", "recall", "f1")
    PROGRESS_METRICS = (
        "batch_time",
        "samples_per_second",
        "learning_rate",
        "gradient_norm",
        "avg_gradient_norm",
        "gpu_memory_used_gb",
        "gpu_memory_total_gb",
        "cpu_percent",
        "total_samples_seen",
    )

    def __init__(
        self,
        output_dir: str = "visualizations",
        metrics_dir: str = "logs",
        num_samples: int = 4,
        min_coverage: float = 0.05,
        dpi: int = 300,
        visualizer: Optional[SegmentationVisualizer] = None,
        num_classes: int = 2,
    ):
        """Initialize callback."""
        super().__init__()

        self.num_samples = num_samples
        self.num_classes = num_classes

        # Set up directories
        self.output_dir = Path(output_dir)
        self.metrics_dir = Path(metrics_dir)
        self.plots_dir = self.output_dir / "plots"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set up metrics file
        self.metrics_file = self.metrics_dir / "metrics.csv"

        # Initialize visualizer
        self.visualizer = visualizer or SegmentationVisualizer(
            metrics_file=str(self.metrics_file),
            min_coverage=min_coverage,
            dpi=dpi,
        )

        # Ensure the visualizer has the correct metrics file path
        self.visualizer.metrics_file = self.metrics_file

        # Store validation batch for visualization
        self.val_batch: BatchTensors | None = None

    def _build_metrics_columns(self) -> list[str]:
        """Return the default CSV columns for tracked metrics."""
        columns = [
            "step",
            "epoch",
            "train_loss",
            "val_loss",
            "train_iou",
            "val_iou",
            "train_precision",
            "val_precision",
            "train_recall",
            "val_recall",
            "train_f1",
            "val_f1",
            "train_dice",
            "val_dice",
            "train_specificity",
            "val_specificity",
        ]

        for metric in self.PER_CLASS_METRICS:
            for class_idx in range(self.num_classes):
                columns.extend(
                    [f"train_{metric}_class_{class_idx}", f"val_{metric}_class_{class_idx}"]
                )

        columns.extend(f"train_{metric}" for metric in self.PROGRESS_METRICS)

        return columns

    def _is_global_zero(self, trainer: pl.Trainer) -> bool:
        """Return whether the current process should write side effects."""
        return bool(getattr(trainer, "is_global_zero", True))

    @staticmethod
    def _to_float(value: MetricScalar | None) -> float:
        """Convert supported numeric-like values to floats without raising."""
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float("nan")
        if isinstance(value, np.generic):
            return float(value.item())
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            return float("nan")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return float("nan")
            tensor_match = re.search(
                r"tensor\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)",
                text,
            )
            if tensor_match:
                try:
                    return float(tensor_match.group(1))
                except ValueError:
                    return float("nan")
            try:
                return float(text)
            except ValueError:
                return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _flatten_metrics(
        self, metrics: dict[str, MetricValue], split: Optional[str] = None
    ) -> dict[str, FlatMetricValue]:
        """Normalize callback metrics into flat CSV column/value pairs."""
        flattened: dict[str, FlatMetricValue] = {}

        if split == "train" and "loss" in metrics:
            flattened["train_loss"] = metrics["loss"]

        prefixes = [split] if split in {"train", "val"} else ["train", "val"]
        scalar_names = set(self.SCALAR_METRICS)
        per_class_names = set(self.PER_CLASS_METRICS)
        per_class_pattern = re.compile(r"^(train|val)_(iou|precision|recall|f1)_class_(\d+)$")

        for key, value in metrics.items():
            if not isinstance(key, str):
                continue
            match = per_class_pattern.fullmatch(key)
            if match and match.group(1) in prefixes:
                flattened[key] = value
                continue

            for prefix in prefixes:
                if key == f"{prefix}_loss":
                    flattened[key] = value
                    break
                if not key.startswith(f"{prefix}_"):
                    continue
                metric_name = key[len(prefix) + 1 :]
                if metric_name in scalar_names:
                    flattened[key] = value
                    break

        if split in {"train", "val"} and isinstance(metrics.get("per_class_metrics"), dict):
            for metric_name, metric_values in metrics["per_class_metrics"].items():
                if metric_name not in per_class_names or metric_values is None:
                    continue

                if isinstance(metric_values, torch.Tensor):
                    values = metric_values.detach().cpu().reshape(-1).tolist()
                elif isinstance(metric_values, np.ndarray):
                    values = metric_values.reshape(-1).tolist()
                elif isinstance(metric_values, (list, tuple)):
                    values = list(metric_values)
                else:
                    values = [metric_values]

                for class_idx, class_value in enumerate(values):
                    flattened[f"{split}_{metric_name}_class_{class_idx}"] = class_value

        if split == "train" and isinstance(metrics.get("training_metrics"), dict):
            for metric_name, metric_value in metrics["training_metrics"].items():
                if metric_name in self.PROGRESS_METRICS:
                    flattened[f"train_{metric_name}"] = metric_value

        return flattened

    def _collect_callback_metrics(
        self, callback_metrics: dict[str, MetricValue], prefix: str
    ) -> dict[str, MetricValue]:
        """Collect tracked metrics from trainer callback metrics."""
        collected: dict[str, MetricValue] = {}
        allowed_scalars = {f"{prefix}_{metric}" for metric in self.SCALAR_METRICS}
        per_class_pattern = re.compile(
            rf"^{prefix}_(?:{'|'.join(self.PER_CLASS_METRICS)})_class_\d+$"
        )

        for key, value in callback_metrics.items():
            if not isinstance(key, str):
                continue
            if key in allowed_scalars or per_class_pattern.fullmatch(key):
                collected[key] = value

        return collected

    def _update_metrics_csv(
        self, metrics: dict[str, MetricValue], epoch: int, split: Optional[str] = None
    ) -> None:
        """Update metrics CSV file with new values."""
        try:
            # Create metrics file if it doesn't exist with comprehensive columns
            if not os.path.exists(self.metrics_file):
                df = pd.DataFrame(columns=self._build_metrics_columns())
                df.to_csv(self.metrics_file, index=False)

            # Read existing metrics
            df = pd.read_csv(self.metrics_file)

            flattened_metrics = self._flatten_metrics(metrics, split=split)

            # Create new row, preserving existing values if not in current metrics
            new_row = {"step": len(df), "epoch": epoch}

            for metric_key, metric_value in flattened_metrics.items():
                new_row[metric_key] = self._to_float(metric_value)

            # Append new row using a temporary DataFrame (handle empty values)
            new_row_df = pd.DataFrame([new_row])

            # Only concatenate if we have actual data
            if (
                len(new_row) > 2
                and not new_row_df.drop(columns=["step", "epoch"], errors="ignore")
                .isna()
                .all()
                .all()
            ):
                all_columns = list(dict.fromkeys([*df.columns, *new_row_df.columns]))
                if df.empty:
                    new_df = new_row_df.reindex(columns=all_columns)
                else:
                    new_df = pd.concat(
                        [df.reindex(columns=all_columns), new_row_df.reindex(columns=all_columns)],
                        ignore_index=True,
                    )
                new_df.to_csv(self.metrics_file, index=False)
            else:
                logger.debug("Skipping empty metrics row")

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            logger.error(f"Metrics: {metrics}")

    def _safe_plot_metrics(self, window_size: int, save_path: str) -> None:
        """Thread-safe metric plotting."""
        try:
            with plt.ioff():  # Disable interactive mode
                self.visualizer.plot_metrics(window_size=window_size, save_path=save_path)
                plt.close("all")  # Clean up all figures
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")

    def _safe_plot_comprehensive_metrics(self, window_size: int, save_path: str) -> None:
        """Thread-safe comprehensive metrics plotting."""
        try:
            with plt.ioff():  # Disable interactive mode
                self.visualizer.plot_comprehensive_metrics(
                    window_size=window_size, save_path=save_path
                )
                plt.close("all")  # Clean up all figures
        except Exception as e:
            logger.error(f"Error plotting comprehensive metrics: {str(e)}")

    def _safe_plot_per_class_metrics(self, window_size: int, save_path: str) -> None:
        """Thread-safe per-class metrics plotting."""
        try:
            with plt.ioff():  # Disable interactive mode
                self.visualizer.plot_per_class_metrics(window_size=window_size, save_path=save_path)
                plt.close("all")  # Clean up all figures
        except Exception as e:
            logger.error(f"Error plotting per-class metrics: {str(e)}")

    @staticmethod
    def _dataset_class_values(dataloader: ValidationDataloader) -> np.ndarray | None:
        """Return dataset class values from a dataloader or dataloader list."""
        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "class_values"):
            return dataloader.dataset.class_values

        if isinstance(dataloader, list):
            for candidate in dataloader:
                class_values = VisualizationCallback._dataset_class_values(candidate)
                if class_values is not None:
                    return class_values

        return None

    def _resolve_validation_dataloader(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> ValidationDataloader:
        """Return the validation dataloader using the available Lightning access pattern."""
        if hasattr(trainer, "val_dataloaders"):
            return trainer.val_dataloaders
        if hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "val_dataloader"):
            return trainer.datamodule.val_dataloader()
        return getattr(pl_module, "_val_dataloader", None)

    def _visualize_batch(
        self,
        *,
        batch: BatchTensors,
        pl_module: pl.LightningModule,
        save_name_template: str,
        class_values: np.ndarray | None = None,
    ) -> None:
        """Run qualitative visualizations for the most informative batch samples."""
        images = batch["image"]
        masks = batch["mask"]
        indices = self.visualizer.find_interesting_slices(masks, self.num_samples)

        with torch.no_grad(), plt.ioff():
            for i, idx in enumerate(indices):
                try:
                    image = images[idx].to(pl_module.device)
                    mask = masks[idx]

                    prediction = pl_module(image.unsqueeze(0))
                    if isinstance(prediction, tuple):
                        prediction = prediction[0]

                    self.visualizer.visualize_prediction(
                        image=image,
                        mask=mask,
                        prediction=prediction,
                        save_path=str(self.plots_dir / save_name_template.format(sample_idx=i)),
                        class_values=class_values,
                    )
                    plt.close("all")
                except Exception as e:
                    logger.error(f"Error visualizing sample {i}: {str(e)}")
                    continue

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, MetricValue] | None,
        batch: BatchTensors | None,
        batch_idx: int,
    ) -> None:
        """Log training metrics after each batch."""
        if not self._is_global_zero(trainer):
            return

        if isinstance(outputs, dict) and "loss" in outputs:
            try:
                # Update metrics CSV
                self._update_metrics_csv(outputs, trainer.current_epoch, split="train")

                # Update plot every 10 steps
                if trainer.global_step % 10 == 0:
                    self._safe_plot_metrics(
                        window_size=200, save_path=str(self.metrics_dir / "metrics.png")
                    )
            except Exception as e:
                logger.error(f"Error in on_train_batch_end: {str(e)}")

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: BatchTensors,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store first validation batch for visualization."""
        if batch_idx == 0:
            self.val_batch = batch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handle end of validation epoch including visualizations."""
        if not self._is_global_zero(trainer):
            self.val_batch = None
            return

        try:
            # Get the validation metrics
            val_metrics = self._collect_callback_metrics(trainer.callback_metrics, prefix="val")

            # Update metrics CSV with validation results
            self._update_metrics_csv(val_metrics, trainer.current_epoch, split="val")

            # Plot updated metrics - generate multiple plot types
            self._safe_plot_metrics(
                window_size=200,
                save_path=str(self.metrics_dir / f"metrics_epoch_{trainer.current_epoch}.png"),
            )

            # Generate comprehensive metrics plot
            self._safe_plot_comprehensive_metrics(
                window_size=200,
                save_path=str(
                    self.metrics_dir / f"comprehensive_metrics_epoch_{trainer.current_epoch}.png"
                ),
            )

            # Generate per-class metrics plot
            self._safe_plot_per_class_metrics(
                window_size=200,
                save_path=str(
                    self.metrics_dir / f"per_class_metrics_epoch_{trainer.current_epoch}.png"
                ),
            )

            # Create visualizations using validation batch
            if self.val_batch is not None:
                class_values = None
                try:
                    class_values = self._dataset_class_values(
                        self._resolve_validation_dataloader(trainer, pl_module)
                    )
                except Exception as e:
                    logger.warning(f"Could not access dataset class values: {e}")

                self._visualize_batch(
                    batch=self.val_batch,
                    pl_module=pl_module,
                    save_name_template=f"val_epoch_{trainer.current_epoch}_sample_{{sample_idx}}.png",
                    class_values=class_values,
                )

                # Clear stored validation batch
                self.val_batch = None

        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {str(e)}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create visualization at the beginning of each training epoch."""
        if not self._is_global_zero(trainer):
            return

        try:
            # Get first batch from train dataloader
            train_loader = trainer.train_dataloader
            batch = next(iter(train_loader))
            class_values = None
            try:
                class_values = self._dataset_class_values(train_loader)
            except Exception as e:
                logger.warning(f"Could not access training dataset class values: {e}")

            self._visualize_batch(
                batch=batch,
                pl_module=pl_module,
                save_name_template=f"epoch_{trainer.current_epoch}_sample_{{sample_idx}}.png",
                class_values=class_values,
            )

        except Exception as e:
            logger.error(f"Error in on_train_epoch_start: {str(e)}")
