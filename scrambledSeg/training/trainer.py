"""Training module for SegFormer model."""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from ..visualization.callbacks import VisualizationCallback
from ..visualization.core import SegmentationVisualizer

logger = logging.getLogger(__name__)

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
        threshold_params: Dict[str, Any] = None,
        vis_dir: str = 'visualizations',
        enable_memory_tracking: bool = False,
        enable_adaptive_batch_size: bool = False,
        target_gpu_util: float = 0.9,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        num_epochs: int = 100,
        gradient_clip_val: float = 1.0,
        visualization: Optional[Dict[str, Any]] = None,
        log_dir: str = 'logs',
        test_mode: bool = False
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
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_adaptive_batch_size = enable_adaptive_batch_size
        self.target_gpu_util = target_gpu_util
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.test_mode = test_mode
        self.validation_step_outputs = []
        
        # Store threshold parameters
        self.threshold_params = threshold_params or {
            'threshold': 0.5,  # Simplified threshold for BCE+Dice
            'min_hole_size_factor': 64
        }
        
        # Set up logging directory
        self.log_dir = Path(log_dir)
        if test_mode:
            self.log_dir = self.log_dir / 'test'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up visualization directories
        self.vis_dir = self.log_dir / 'plots'
        self.metrics_dir = self.log_dir / 'metrics'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure visualization callback
        vis_config = visualization or {}
        self.vis_callback = VisualizationCallback(
            output_dir=str(self.vis_dir),
            metrics_dir=str(self.metrics_dir),
            num_samples=vis_config.get('num_samples', 4),
            min_coverage=vis_config.get('min_coverage', 0.05),
            dpi=vis_config.get('dpi', 300),
            enable_memory_tracking=enable_memory_tracking,
            visualizer=SegmentationVisualizer(
                metrics_file=str(self.metrics_dir / 'metrics.csv'),
                min_coverage=vis_config.get('min_coverage', 0.05),
                dpi=vis_config.get('dpi', 300)
            )
        )
        
        # Set up metrics
        self.iou_metric = torchmetrics.JaccardIndex(task="binary", num_classes=1)
        
        # Save hyperparameters only if not in test mode
        if not test_mode:
            logger.info("Saving hyperparameters...")
            save_params = {
                'enable_memory_tracking': enable_memory_tracking,
                'enable_adaptive_batch_size': enable_adaptive_batch_size,
                'target_gpu_util': target_gpu_util,
                'min_batch_size': min_batch_size,
                'max_batch_size': max_batch_size,
                'num_epochs': num_epochs,
                'gradient_clip_val': gradient_clip_val,
                'test_mode': test_mode
            }
            self.save_hyperparameters(save_params)
        else:
            logger.info("Test mode: Skipping hyperparameter logging")

    def get_binary_predictions(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get binary predictions using thresholding and optional morphological cleanup.
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Target masks, used for device placement
            
        Returns:
            Binary predictions tensor (B, C, H, W)
        """
        # Input validation
        if predictions.ndim != 4:
            raise ValueError(f"Expected 4D predictions tensor, got shape {predictions.shape}")
        
        # Check if sigmoid needs to be applied
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Apply threshold
        threshold = self.threshold_params.get('threshold', 0.5)
        binary_pred = (predictions > threshold).float()
        
        # Apply morphological cleanup if enabled
        if self.threshold_params.get('enable_cleanup', True):
            kernel_size = self.threshold_params.get('cleanup_kernel_size', 3)
            cleanup_threshold = self.threshold_params.get('cleanup_threshold', 5)
            
            # Ensure kernel size is odd
            kernel_size = max(3, kernel_size + (kernel_size + 1) % 2)
            pad_size = kernel_size // 2
            
            # Create cleanup kernel
            cleanup_kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                                      device=predictions.device)
            
            # Apply morphological operation
            padded_binary = F.pad(binary_pred, 
                                (pad_size, pad_size, pad_size, pad_size),
                                mode='constant', 
                                value=0)
            connected = F.conv2d(padded_binary, cleanup_kernel, padding=0)
            
            # Calculate threshold based on kernel size
            auto_threshold = (kernel_size * kernel_size) / 2
            final_threshold = min(cleanup_threshold, auto_threshold)
            
            # Apply cleanup
            final_pred = torch.where(connected >= final_threshold, 
                                   binary_pred, 
                                   torch.zeros_like(binary_pred))
            
            return final_pred
        
        return binary_pred

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with enhanced monitoring."""
        images = batch['image']
        masks = batch['mask']
        
        # Input statistics
        img_stats = {
            'img_min': images.min().item(),
            'img_max': images.max().item(),
            'img_mean': images.mean().item(),
            'mask_coverage': masks.mean().item()
        }
        self.log_dict({f'input/{k}': v for k, v in img_stats.items()}, on_step=True)
        
        # Check for NaN in inputs
        if torch.isnan(images).any() or torch.isnan(masks).any():
            logger.error("NaN detected in input tensors")
            logger.error(f"Input stats: {img_stats}")
            self.trainer.should_stop = True
            return None
        
        # Forward pass
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Output statistics
        output_stats = {
            'out_min': outputs.min().item(),
            'out_max': outputs.max().item(),
            'out_mean': outputs.mean().item(),
            'out_std': outputs.std().item(),
            'unique_values': len(torch.unique(outputs)),
            'zeros_pct': (outputs == 0).float().mean().item() * 100
        }
        self.log_dict({f'output/{k}': v for k, v in output_stats.items()}, on_step=True)
        
        # Check for suspicious output patterns
        if output_stats['unique_values'] < 10:  # Very few unique values
            logger.warning(f"Low variance in outputs: {output_stats}")
        if output_stats['zeros_pct'] > 90:  # Mostly zeros
            logger.warning(f"Output mostly zeros: {output_stats}")
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any():
            logger.error("NaN detected in model outputs")
            logger.error(f"Output stats: {output_stats}")
            self.trainer.should_stop = True
            return None
        
        # Calculate combined BCE+Dice loss
        loss = self.criterion(outputs, masks)
        
        # Loss statistics
        loss_stats = {
            'loss_value': loss.item()
        }
        self.log_dict({f'loss/{k}': v for k, v in loss_stats.items()}, on_step=True)
        
        # Check for NaN in loss
        if torch.isnan(loss).any():
            logger.error("NaN detected in loss calculation")
            logger.error(f"Loss stats: {loss_stats}")
            self.trainer.should_stop = True
            return None
        
        # Get binary predictions and calculate IoU
        binary_preds = self.get_binary_predictions(outputs, masks)
        iou = self.iou_metric(binary_preds, masks)
        
        # Binary prediction statistics
        pred_stats = {
            'pred_coverage': binary_preds.mean().item(),
            'pred_zeros': (binary_preds == 0).float().mean().item() * 100,
            'pred_ones': (binary_preds == 1).float().mean().item() * 100
        }
        self.log_dict({f'pred/{k}': v for k, v in pred_stats.items()}, on_step=True)
        
        # Log main metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return {
            'loss': loss,
            'train_iou': iou,
            'output_stats': output_stats,
            'pred_stats': pred_stats
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Calculate loss
        loss = self.criterion(outputs, masks)
        
        # Calculate IoU
        iou = self.iou_metric(self.get_binary_predictions(outputs, masks), masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', iou, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store for epoch end processing
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_iou': iou
        })
        
        return {
            'val_loss': loss,
            'val_iou': iou
        }


    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        try:
            # Aggregate validation metrics
            val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            val_iou = torch.stack([x['val_iou'] for x in self.validation_step_outputs]).mean()
            
            # Log aggregated metrics
            self.log('val_loss_epoch', val_loss, prog_bar=True)
            self.log('val_iou_epoch', val_iou, prog_bar=True)
            
            # Clear the outputs list
            self.validation_step_outputs.clear()
            
        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {str(e)}")

    def on_validation_epoch_start(self) -> None:
        """Clear the validation outputs at the start of each validation epoch."""
        self.validation_step_outputs = []

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        if self._scheduler is None:
            return self._optimizer
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {
                "scheduler": self._scheduler,
                "interval": "step"
            }
        }
    
    def train_dataloader(self):
        """Return training dataloader."""
        return self._train_dataloader
        
    def val_dataloader(self):
        """Return validation dataloader."""
        return self._val_dataloader