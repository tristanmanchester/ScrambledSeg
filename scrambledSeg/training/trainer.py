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
        # Update to reflect the new configuration structure
        self.prediction_params = threshold_params or {
            'enable_cleanup': True,
            'cleanup_kernel_size': 3,
            'cleanup_threshold': 5,
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
        
        # Set up metrics for multi-class segmentation
        self.iou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=4)
        
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

    def get_predicted_labels(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels using argmax and optional cleanup.
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Target masks, used for device placement
            
        Returns:
            Class prediction tensor (B, H, W) with integer class labels
        """
        # Input validation
        if predictions.ndim != 4:
            raise ValueError(f"Expected 4D predictions tensor, got shape {predictions.shape}")
        
        # Apply softmax to get class probabilities
        if predictions.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            # Get predicted class indices using argmax of softmax probabilities
            pred_probs = F.softmax(predictions, dim=1)
            pred_classes = torch.argmax(pred_probs, dim=1)  # B, H, W with class indices
        else:
            # If predictions are already processed, just use argmax
            pred_classes = torch.argmax(predictions, dim=1)  # B, H, W with class indices
        
        # Ensure we have long type for class indices
        pred_classes = pred_classes.long()
        
        # Convert to one-hot encoding for potential cleanup
        num_classes = predictions.size(1)
        pred_one_hot = F.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()  # B, C, H, W
        
        # Apply morphological cleanup if enabled
        # Use prediction_params if available, otherwise fall back to threshold_params for backward compatibility
        prediction_params = getattr(self, 'prediction_params', None) or getattr(self, 'threshold_params', {}) or {}
        if prediction_params.get('enable_cleanup', True):
            kernel_size = prediction_params.get('cleanup_kernel_size', 3)
            cleanup_threshold = prediction_params.get('cleanup_threshold', 5)
            
            # Ensure kernel size is odd
            kernel_size = max(3, kernel_size + (kernel_size + 1) % 2)
            pad_size = kernel_size // 2
            
            # Create cleanup kernel
            cleanup_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=predictions.device)
            
            # Process each class separately
            cleaned_pred_one_hot = torch.zeros_like(pred_one_hot)
            
            for c in range(num_classes):
                class_mask = pred_one_hot[:, c:c+1, :, :]  # Keep dim for conv2d: B, 1, H, W
                
                # Apply morphological operation
                padded_mask = F.pad(class_mask, 
                                    (pad_size, pad_size, pad_size, pad_size),
                                    mode='constant', 
                                    value=0)
                connected = F.conv2d(padded_mask, cleanup_kernel, padding=0)
                
                # Calculate threshold based on kernel size
                auto_threshold = (kernel_size * kernel_size) / 2
                final_threshold = min(cleanup_threshold, auto_threshold)
                
                # Apply cleanup for this class
                cleaned_mask = torch.where(connected >= final_threshold, 
                                          class_mask, 
                                          torch.zeros_like(class_mask))
                
                cleaned_pred_one_hot[:, c:c+1, :, :] = cleaned_mask
            
            # Convert back to class indices
            # If a pixel has no class after cleanup, assign the original prediction
            cleaned_sum = torch.sum(cleaned_pred_one_hot, dim=1, keepdim=True)
            valid_mask = (cleaned_sum > 0).squeeze(1)
            
            # Where valid, get argmax of cleaned predictions
            cleaned_classes = torch.argmax(cleaned_pred_one_hot, dim=1)
            
            # Combine: use cleaned where valid, otherwise use original prediction
            # Handle boolean mask by ensuring both sides of where() have the same dtype
            cleaned_classes = cleaned_classes.to(pred_classes.dtype)
            final_pred = torch.where(valid_mask, cleaned_classes, pred_classes)
            
            return final_pred.long()  # Ensure long type for class indices
        
        return pred_classes

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step with enhanced monitoring."""
        images = batch['image']
        masks = batch['mask']
        
        # Input statistics
        # Convert masks to float for statistics calculation
        # For multi-class segmentation, mask coverage is the % of non-background pixels
        if masks.dtype == torch.long:
            # For integer class labels, calculate non-background coverage
            mask_coverage = (masks > 0).float().mean().item()
        elif masks.dtype == torch.bool:
            # For boolean masks, convert to float first
            mask_coverage = masks.float().mean().item()
        else:
            # For float masks, can call mean directly
            mask_coverage = masks.mean().item()
        
        img_stats = {
            'img_min': images.min().item(),
            'img_max': images.max().item(),
            'img_mean': images.mean().item(),
            'mask_coverage': mask_coverage
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
        # Handle different data types and multi-class outputs
        if outputs.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            output_stats = {
                'out_min': outputs.min().item(),
                'out_max': outputs.max().item(),
                'out_mean': outputs.mean().item(),
                'out_std': outputs.std().item(),
                'unique_values': len(torch.unique(outputs)),
                'zeros_pct': (outputs == 0).float().mean().item() * 100
            }
        else:
            # For non-float types, convert to float for statistics
            outputs_float = outputs.float()
            output_stats = {
                'out_min': outputs_float.min().item(),
                'out_max': outputs_float.max().item(),
                'out_mean': outputs_float.mean().item(),
                'out_std': outputs_float.std().item(),
                'unique_values': len(torch.unique(outputs)),
                'zeros_pct': (outputs == 0).float().mean().item() * 100
            }
        self.log_dict({f'output/{k}': v for k, v in output_stats.items()}, on_step=True)
        
        # Check for suspicious output patterns
        if output_stats.get('unique_values', 20) < 10:  # Very few unique values
            logger.warning(f"Low variance in outputs: {output_stats}")
        if output_stats.get('zeros_pct', 0) > 90:  # Mostly zeros
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
        
        # Get predicted class labels and calculate IoU
        pred_labels = self.get_predicted_labels(outputs, masks)
        
        # For multi-class IoU, we need to ensure targets are in the expected format
        # IoU metric expects class indices of type long, not one-hot or float
        masks_for_iou = masks.clone()  # Make a copy to avoid modifying original
        
        # The JaccardIndex expects class indices as long type
        if masks_for_iou.dim() == 4 and masks_for_iou.size(1) == 1:
            masks_for_iou = masks_for_iou.squeeze(1)  # B, H, W
        
        # Convert to long type for IoU calculation
        if masks_for_iou.dtype != torch.long:
            if masks_for_iou.dtype == torch.bool:
                # Boolean tensors need to be converted to long carefully
                masks_for_iou = masks_for_iou.long()
            else:
                # Float tensors should be rounded to nearest integer
                masks_for_iou = torch.round(masks_for_iou).long()
            
        iou = self.iou_metric(pred_labels, masks_for_iou)
        
        # Class prediction statistics (for multiple classes)
        num_classes = outputs.size(1)
        
        # Always convert to float for statistics
        pred_labels_float = pred_labels.float()
        
        # Make sure to convert boolean tensor to float before calling mean()
        pred_stats = {
            'pred_coverage': (pred_labels_float > 0).float().mean().item(),  # Non-background coverage
        }
        
        # Add per-class statistics
        for c in range(num_classes):
            pred_stats[f'class_{c}_pct'] = (pred_labels == c).float().mean().item() * 100
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
        
        # Calculate loss - ensure masks are in the correct format
        # For CrossEntropyLoss, masks should be (B, H, W) with class indices as Long type
        # For BCEDiceLoss, masks should be (B, 1, H, W) with values in [0,1] as Float type
        
        # Pre-process masks for CrossEntropyLoss
        masks_for_loss = masks.clone()  # Make a copy to avoid modifying original
        
        # For our multi-class segmentation with CrossEntropyDiceLoss
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or hasattr(self.criterion, 'ce_criterion'):
            # This is multi-class loss - masks should be class indices (B, H, W)
            if masks_for_loss.dim() == 4 and masks_for_loss.size(1) == 1:
                masks_for_loss = masks_for_loss.squeeze(1)  # Remove channel dimension
            
            # Explicitly convert float labels to long/integer type
            # Handle both binary (0/1) and multi-class cases
            if masks_for_loss.dtype != torch.long:
                masks_for_loss = masks_for_loss.long()
        
        # Calculate loss
        try:
            loss = self.criterion(outputs, masks_for_loss)
        except RuntimeError as e:
            # Provide more diagnostic information on error
            logger.error(f"Error in validation loss calculation: {e}")
            logger.error(f"Output shape: {outputs.shape}, dtype: {outputs.dtype}")
            logger.error(f"Masks shape: {masks_for_loss.shape}, dtype: {masks_for_loss.dtype}")
            logger.error(f"Unique mask values: {torch.unique(masks_for_loss)}")
            # Re-raise to stop training
            raise
        
        # Calculate IoU for multi-class
        pred_labels = self.get_predicted_labels(outputs, masks)
        
        # Ensure masks are in the expected format for IoU calculation
        masks_for_iou = masks.clone()  # Make a copy to avoid modifying original
        
        # JaccardIndex expects class indices as long type
        if masks_for_iou.dim() == 4 and masks_for_iou.size(1) == 1:
            masks_for_iou = masks_for_iou.squeeze(1)  # B, H, W
        
        # Convert to long type for IoU calculation
        if masks_for_iou.dtype != torch.long:
            if masks_for_iou.dtype == torch.bool:
                # Boolean tensors need to be converted to long carefully
                masks_for_iou = masks_for_iou.long()
            else:
                # Float tensors should be rounded to nearest integer
                masks_for_iou = torch.round(masks_for_iou).long()
            
        iou = self.iou_metric(pred_labels, masks_for_iou)
        
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