"""Callbacks for visualization during training."""
import os
from pathlib import Path
import logging
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import pandas as pd
from .core import SegmentationVisualizer

logger = logging.getLogger(__name__)

class VisualizationCallback(pl.Callback):
    """Callback for visualizing segmentation results."""

    def __init__(
        self,
        output_dir: str = 'visualizations',
        metrics_dir: str = 'logs',
        num_samples: int = 4,
        min_coverage: float = 0.05,
        dpi: int = 300,
        enable_memory_tracking: bool = False,
        visualizer: Optional[SegmentationVisualizer] = None
    ):
        """Initialize callback."""
        super().__init__()
        
        self.num_samples = num_samples
        
        # Set up directories
        self.output_dir = Path(output_dir)
        self.metrics_dir = Path(metrics_dir)
        self.plots_dir = self.output_dir / 'plots'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up metrics file
        self.metrics_file = self.metrics_dir / 'metrics.csv'
        
        # Initialize visualizer
        self.visualizer = visualizer or SegmentationVisualizer(
            metrics_file=str(self.metrics_file),
            min_coverage=min_coverage,
            dpi=dpi,
        )
        
        # Ensure the visualizer has the correct metrics file path
        self.visualizer.metrics_file = self.metrics_file
        
        # Store validation batch for visualization
        self.val_batch = None

    def _update_metrics_csv(self, metrics: Dict[str, float], epoch: int) -> None:
        """Update metrics CSV file with new values."""
        try:
            # Create metrics file if it doesn't exist
            if not os.path.exists(self.metrics_file):
                df = pd.DataFrame(columns=['step', 'epoch', 'train_loss', 'val_loss', 'train_iou', 'val_iou'])
                df.to_csv(self.metrics_file, index=False)
            
            # Read existing metrics
            df = pd.read_csv(self.metrics_file)
            
            # Helper function to convert tensor to float
            def to_float(value):
                if value is None:
                    return float('nan')
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().item()
                if isinstance(value, str) and 'tensor' in value:
                    return float(value.split('tensor(')[1].split(',')[0])
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float('nan')

            # Create new row, preserving existing values if not in current metrics
            new_row = {
                'step': len(df),
                'epoch': epoch
            }
            
            # Update metrics that are present
            if 'loss' in metrics:
                new_row['train_loss'] = to_float(metrics['loss'])
            if 'train_iou' in metrics:
                new_row['train_iou'] = to_float(metrics['train_iou'])
            if 'val_loss' in metrics:
                new_row['val_loss'] = to_float(metrics['val_loss'])
            if 'val_iou' in metrics:
                new_row['val_iou'] = to_float(metrics['val_iou'])
            
            # Append new row using a temporary DataFrame
            new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            new_df.to_csv(self.metrics_file, index=False)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            logger.error(f"Metrics: {metrics}")

    def _safe_plot_metrics(self, window_size: int, save_path: str) -> None:
        """Thread-safe metric plotting."""
        try:
            with plt.ioff():  # Disable interactive mode
                self.visualizer.plot_metrics(window_size=window_size, save_path=save_path)
                plt.close('all')  # Clean up all figures
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
                          outputs: Dict[str, torch.Tensor], batch: Any, batch_idx: int) -> None:
        """Log training metrics after each batch."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            try:
                # Update metrics CSV
                self._update_metrics_csv(outputs, trainer.current_epoch)
                
                # Update plot every 10 steps
                if trainer.global_step % 10 == 0:
                    self._safe_plot_metrics(
                        window_size=200,
                        save_path=str(self.metrics_dir / 'metrics.png')
                    )
            except Exception as e:
                logger.error(f"Error in on_train_batch_end: {str(e)}")

    def on_validation_batch_start(
                                    self, 
                                    trainer: pl.Trainer, 
                                    pl_module: pl.LightningModule, 
                                    batch: Any, 
                                    batch_idx: int, 
                                    dataloader_idx: int = 0
                                ) -> None:
        """Store first validation batch for visualization."""
        if batch_idx == 0:
            self.val_batch = batch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handle end of validation epoch including visualizations."""
        try:
            # Get the validation metrics
            val_metrics = {
                'val_loss': trainer.callback_metrics.get('val_loss'),
                'val_iou': trainer.callback_metrics.get('val_iou')
            }
            
            # Update metrics CSV with validation results
            self._update_metrics_csv({**val_metrics}, trainer.current_epoch)
            
            # Plot updated metrics
            self._safe_plot_metrics(
                window_size=200,
                save_path=str(self.metrics_dir / f'metrics_epoch_{trainer.current_epoch}.png')
            )

            # Create visualizations using validation batch
            if self.val_batch is not None:
                images = self.val_batch['image']
                masks = self.val_batch['mask']

                # Select interesting slices
                indices = self.visualizer.find_interesting_slices(masks, self.num_samples)
                
                # Create and save visualizations
                with torch.no_grad(), plt.ioff():  # Disable gradients and interactive mode
                    for i, idx in enumerate(indices):
                        try:
                            # Move image to the same device as the model
                            image = images[idx].to(pl_module.device)
                            mask = masks[idx]
                            
                            # Make prediction
                            prediction = pl_module(image.unsqueeze(0))
                            if isinstance(prediction, tuple):
                                prediction = prediction[0]
                            
                            # Save visualization - pass class values if available from dataset
                            class_values = None
                            
                            # Get the validation dataloader safely
                            try:
                                # For PyTorch Lightning >= 2.0
                                if hasattr(trainer, 'val_dataloaders'):
                                    val_dataloader = trainer.val_dataloaders
                                # For PyTorch Lightning < 2.0
                                elif hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'val_dataloader'):
                                    val_dataloader = trainer.datamodule.val_dataloader()
                                # Direct access from module
                                elif hasattr(pl_module, '_val_dataloader'):
                                    val_dataloader = pl_module._val_dataloader
                                
                                # Access class_values if available
                                if hasattr(val_dataloader, 'dataset') and hasattr(val_dataloader.dataset, 'class_values'):
                                    class_values = val_dataloader.dataset.class_values
                                elif isinstance(val_dataloader, list) and len(val_dataloader) > 0:
                                    if hasattr(val_dataloader[0].dataset, 'class_values'):
                                        class_values = val_dataloader[0].dataset.class_values
                            except Exception as e:
                                logger.warning(f"Could not access dataset class values: {e}")
                                class_values = None
                            
                            save_path = str(self.plots_dir / f'val_epoch_{trainer.current_epoch}_sample_{i}.png')
                            self.visualizer.visualize_prediction(
                                image=image,
                                mask=mask,
                                prediction=prediction,
                                save_path=save_path,
                                class_values=class_values
                            )
                            plt.close('all')  # Clean up after each visualization
                            
                        except Exception as e:
                            logger.error(f"Error visualizing validation sample {i}: {str(e)}")
                            continue

                # Clear stored validation batch
                self.val_batch = None
                            
        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {str(e)}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create visualization at the beginning of each training epoch."""
        try:
            # Get first batch from train dataloader
            train_loader = trainer.train_dataloader
            batch = next(iter(train_loader))
            images = batch['image']
            masks = batch['mask']

            # Select interesting slices
            indices = self.visualizer.find_interesting_slices(masks, self.num_samples)
            
            # Create and save visualizations
            with torch.no_grad(), plt.ioff():  # Disable gradients and interactive mode
                for i, idx in enumerate(indices):
                    try:
                        # Move image to the same device as the model
                        image = images[idx].to(pl_module.device)
                        mask = masks[idx]
                        
                        # Make prediction
                        prediction = pl_module(image.unsqueeze(0))
                        if isinstance(prediction, tuple):
                            prediction = prediction[0]
                        
                        # Save visualization - pass class values if available from dataset
                        class_values = None
                        try:
                            # Check if dataloader has dataset attribute and class_values
                            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'class_values'):
                                class_values = train_loader.dataset.class_values
                        except Exception as e:
                            logger.warning(f"Could not access training dataset class values: {e}")
                            class_values = None
                        
                        save_path = str(self.plots_dir / f'epoch_{trainer.current_epoch}_sample_{i}.png')
                        self.visualizer.visualize_prediction(
                            image=image,
                            mask=mask,
                            prediction=prediction,
                            save_path=save_path,
                            class_values=class_values
                        )
                        plt.close('all')  # Clean up after each visualization
                        
                    except Exception as e:
                        logger.error(f"Error visualizing sample {i}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in on_train_epoch_start: {str(e)}")