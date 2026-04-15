"""Rich progress callback for PyTorch Lightning training."""

from collections.abc import Mapping

import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

NumericMetricValue = torch.Tensor | float | int
CallbackOutputs = torch.Tensor | Mapping[str, NumericMetricValue] | None
BatchTensors = Mapping[str, torch.Tensor] | None

console = Console()


class RichProgressCallback(pl.Callback):
    """Rich progress bar callback for PyTorch Lightning."""

    def __init__(self) -> None:
        super().__init__()
        self.progress: Progress | None = None
        self.epoch_task: int | None = None
        self.train_task: int | None = None
        self.val_task: int | None = None
        self.live: Live | None = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_metrics: dict[str, float] = {}
        self.val_metrics: dict[str, float] = {}
        self.last_update_step = 0
        self.update_frequency = 10  # Update display every N steps

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize progress tracking."""
        try:
            self.total_epochs = trainer.max_epochs

            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=True,
            )

            self.epoch_task = self.progress.add_task(
                "[bold green]Epochs",
                total=self.total_epochs,
            )

            self.live = Live(self._get_layout(), console=console, refresh_per_second=1)
            self.live.start()

        except Exception as exc:
            console.print(f"[red]Error initializing Rich progress display: {exc}[/red]")
            self.progress = None
            self.live = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start epoch progress tracking."""
        if not self.progress:
            return

        self.current_epoch = trainer.current_epoch

        if self.train_task is not None:
            self.progress.remove_task(self.train_task)

        self.train_task = self.progress.add_task(
            f"[cyan]  Epoch {self.current_epoch + 1}/{self.total_epochs} - Training",
            total=len(trainer.train_dataloader),
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: CallbackOutputs,
        batch: BatchTensors,
        batch_idx: int,
    ) -> None:
        """Update training progress."""
        if self.progress and self.train_task is not None:
            self.progress.advance(self.train_task, 1)

        self._update_metrics(trainer)

        if (
            self.live
            and hasattr(self.live, "update")
            and (batch_idx - self.last_update_step) >= self.update_frequency
        ):
            self.live.update(self._get_layout())
            self.last_update_step = batch_idx

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start validation progress tracking."""
        if not self.progress:
            return

        if hasattr(trainer, "val_dataloaders") and trainer.val_dataloaders:
            val_loader = (
                trainer.val_dataloaders[0]
                if isinstance(trainer.val_dataloaders, list)
                else trainer.val_dataloaders
            )

            if self.val_task is not None:
                self.progress.remove_task(self.val_task)

            self.val_task = self.progress.add_task(
                f"[yellow]  Epoch {self.current_epoch + 1}/{self.total_epochs} - Validation",
                total=len(val_loader),
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: CallbackOutputs,
        batch: BatchTensors,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation progress."""
        if self.progress and self.val_task is not None:
            self.progress.advance(self.val_task, 1)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Complete validation epoch."""
        self._update_metrics(trainer)

        if self.live and hasattr(self.live, "update"):
            self.live.update(self._get_layout())

        if self.progress and self.val_task is not None:
            self.progress.remove_task(self.val_task)
            self.val_task = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Complete training epoch."""
        if not self.progress:
            return

        if self.epoch_task is not None:
            self.progress.advance(self.epoch_task, 1)

        if self.train_task is not None:
            self.progress.remove_task(self.train_task)
            self.train_task = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up progress tracking."""
        if self.live:
            self.live.stop()

    def _update_metrics(self, trainer: pl.Trainer) -> None:
        """Update metrics from trainer's logged metrics."""
        try:
            if hasattr(trainer, "callback_metrics"):
                logged_metrics = trainer.callback_metrics
            elif hasattr(trainer, "logged_metrics"):
                logged_metrics = trainer.logged_metrics
            else:
                return

            for key, value in logged_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    float_value = float(value.item())
                elif isinstance(value, (int, float)):
                    float_value = float(value)
                else:
                    continue

                if key.startswith("train_"):
                    self.train_metrics[key] = float_value
                elif key.startswith("val_"):
                    self.val_metrics[key] = float_value

        except Exception as exc:
            console.log(f"Skipping Rich metrics update due to display error: {exc}")

    def _get_layout(self) -> Panel:
        """Get the rich layout for live display."""
        try:
            from rich.console import Group

            if not self.progress:
                return Panel(
                    "Progress display not initialized",
                    title="Training Progress",
                    border_style="red",
                )

            layout_elements = [self.progress]

            if self.train_metrics or self.val_metrics:
                metrics_table = Table(
                    title=f"Epoch {self.current_epoch + 1} Metrics",
                    show_header=True,
                    header_style="bold magenta",
                )
                metrics_table.add_column("Metric", style="cyan", no_wrap=True)
                metrics_table.add_column("Value", style="white", justify="right")

                key_metrics = [
                    "train_loss",
                    "train_iou",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "val_loss",
                    "val_iou",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                ]

                for key in key_metrics:
                    if key in self.train_metrics and self.train_metrics[key] is not None:
                        display_name = key.replace("_", " ").title()
                        metrics_table.add_row(display_name, f"{self.train_metrics[key]:.4f}")

                for key in key_metrics:
                    if key in self.val_metrics and self.val_metrics[key] is not None:
                        display_name = key.replace("_", " ").title()
                        metrics_table.add_row(display_name, f"{self.val_metrics[key]:.4f}")

                layout_elements.append(metrics_table)

            return Panel(
                Group(*layout_elements),
                title="Training Progress",
                border_style="blue",
            )

        except Exception as exc:
            return Panel(
                f"Training in progress... (display error: {exc})",
                title="Training Progress",
                border_style="yellow",
            )
