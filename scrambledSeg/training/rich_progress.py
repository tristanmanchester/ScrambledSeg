"""Rich progress callback for PyTorch Lightning training."""
import pytorch_lightning as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import torch
from typing import Any, Dict, Optional

console = Console()

class RichProgressCallback(pl.Callback):
    """Rich progress bar callback for PyTorch Lightning."""
    
    def __init__(self):
        super().__init__()
        self.progress = None
        self.epoch_task = None
        self.train_task = None
        self.val_task = None
        self.live = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_metrics = {}
        self.val_metrics = {}
        self.last_update_step = 0
        self.update_frequency = 10  # Update display every N steps
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize progress tracking."""
        try:
            self.total_epochs = trainer.max_epochs
            
            # Create progress instance
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                expand=True
            )
            
            # Add epoch task
            self.epoch_task = self.progress.add_task(
                f"[bold green]Epochs", 
                total=self.total_epochs
            )
            
            # Start live display with slower refresh to avoid spam
            self.live = Live(self._get_layout(), console=console, refresh_per_second=1)
            self.live.start()
            
        except Exception as e:
            console.print(f"[red]Error initializing Rich progress display: {e}[/red]")
            # Fallback: disable rich display
            self.progress = None
            self.live = None
        
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start epoch progress tracking."""
        if not self.progress:
            return
            
        self.current_epoch = trainer.current_epoch
        
        # Add or update train task
        if self.train_task is not None:
            self.progress.remove_task(self.train_task)
            
        self.train_task = self.progress.add_task(
            f"[cyan]  Epoch {self.current_epoch + 1}/{self.total_epochs} - Training",
            total=len(trainer.train_dataloader)
        )
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update training progress."""
        if self.progress and self.train_task is not None:
            self.progress.advance(self.train_task, 1)
            
        # Update metrics from trainer's callback_metrics and logged_metrics
        self._update_metrics(trainer)
        
        # Update live display with throttling
        if (self.live and hasattr(self.live, 'update') and 
            (batch_idx - self.last_update_step) >= self.update_frequency):
            self.live.update(self._get_layout())
            self.last_update_step = batch_idx
            
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Start validation progress tracking."""
        if not self.progress:
            return
            
        if hasattr(trainer, 'val_dataloaders') and trainer.val_dataloaders:
            val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            
            # Add or update validation task
            if self.val_task is not None:
                self.progress.remove_task(self.val_task)
                
            self.val_task = self.progress.add_task(
                f"[yellow]  Epoch {self.current_epoch + 1}/{self.total_epochs} - Validation",
                total=len(val_loader)
            )
            
    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update validation progress."""
        if self.progress and self.val_task is not None:
            self.progress.advance(self.val_task, 1)
            
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Complete validation epoch."""
        # Update metrics from trainer
        self._update_metrics(trainer)
        
        # Update live display with final metrics
        if self.live and hasattr(self.live, 'update'):
            self.live.update(self._get_layout())
            
        # Remove validation task
        if self.progress and self.val_task is not None:
            self.progress.remove_task(self.val_task)
            self.val_task = None
            
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Complete training epoch."""
        if not self.progress:
            return
            
        # Update epoch progress
        if self.epoch_task is not None:
            self.progress.advance(self.epoch_task, 1)
        
        # Remove training task
        if self.train_task is not None:
            self.progress.remove_task(self.train_task)
            self.train_task = None
            
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up progress tracking."""
        if self.live:
            self.live.stop()
    
    def _update_metrics(self, trainer: pl.Trainer):
        """Update metrics from trainer's logged metrics."""
        try:
            # Get metrics from trainer's callback_metrics (most reliable)
            if hasattr(trainer, 'callback_metrics'):
                logged_metrics = trainer.callback_metrics
            elif hasattr(trainer, 'logged_metrics'):
                logged_metrics = trainer.logged_metrics
            else:
                return
                
            # Convert tensor values to floats and filter by type
            for key, value in logged_metrics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    float_value = value.item()
                elif isinstance(value, (int, float)):
                    float_value = float(value)
                else:
                    continue  # Skip non-numeric values
                    
                if key.startswith('train_'):
                    self.train_metrics[key] = float_value
                elif key.startswith('val_'):
                    self.val_metrics[key] = float_value
                    
        except Exception as e:
            # Silently continue if metrics update fails
            pass
            
    def _get_layout(self):
        """Get the rich layout for live display."""
        try:
            from rich.console import Group
            
            if not self.progress:
                return Panel("Progress display not initialized", title="Training Progress", border_style="red")
            
            layout_elements = [self.progress]
            
            # Add metrics table if we have metrics
            if self.train_metrics or self.val_metrics:
                metrics_table = Table(title=f"Epoch {self.current_epoch + 1} Metrics", show_header=True, header_style="bold magenta")
                metrics_table.add_column("Metric", style="cyan", no_wrap=True)
                metrics_table.add_column("Value", style="white", justify="right")
                
                # Key metrics to display (limit to most important ones)
                key_metrics = ['train_loss', 'train_iou', 'train_precision', 'train_recall', 'train_f1', 
                              'val_loss', 'val_iou', 'val_precision', 'val_recall', 'val_f1']
                
                # Add training metrics (only key ones)
                for key in key_metrics:
                    if key in self.train_metrics and self.train_metrics[key] is not None:
                        value = self.train_metrics[key]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        display_name = key.replace('_', ' ').title()
                        metrics_table.add_row(display_name, f"{value:.4f}")
                    
                # Add validation metrics (only key ones)
                for key in key_metrics:
                    if key in self.val_metrics and self.val_metrics[key] is not None:
                        value = self.val_metrics[key]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        display_name = key.replace('_', ' ').title()
                        metrics_table.add_row(display_name, f"{value:.4f}")
                    
                layout_elements.append(metrics_table)
                
            # Use Group to properly combine Rich renderables
            return Panel(
                Group(*layout_elements),
                title="Training Progress",
                border_style="blue"
            )
            
        except Exception as e:
            # Fallback to simple text if Rich fails
            return Panel(f"Training in progress... (display error: {e})", title="Training Progress", border_style="yellow")