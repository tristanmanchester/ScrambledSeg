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
import time
import psutil
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
        test_mode: bool = False,
        num_classes: int = 2
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
            num_classes=num_classes,
            visualizer=SegmentationVisualizer(
                metrics_file=str(self.metrics_dir / 'metrics.csv'),
                min_coverage=vis_config.get('min_coverage', 0.05),
                dpi=vis_config.get('dpi', 300)
            )
        )
        
        # Set up comprehensive metrics for multi-class segmentation
        # Use num_classes from config instead of hardcoded value
        
        # Primary metrics
        self.iou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        # Use segmentation DiceScore for proper multi-class dice calculation
        from torchmetrics.segmentation import DiceScore
        self.dice_metric = DiceScore(num_classes=num_classes, average='macro', input_format='index')
        self.specificity_metric = torchmetrics.Specificity(task="multiclass", num_classes=num_classes, average='macro')
        
        # Per-class metrics for detailed analysis
        self.iou_per_class = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        self.precision_per_class = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None)
        self.recall_per_class = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None)
        self.f1_per_class = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=None)
        
        # Confusion matrix for detailed analysis
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        # Store generic class names for better reporting
        self.class_names = [f'Class_{i}' for i in range(num_classes)]  # Generate class names dynamically
        
        # Training monitoring variables
        self.batch_start_time = None
        self.epoch_start_time = None
        self.total_training_samples = 0
        
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
        # Start timing for this batch
        batch_start_time = time.time()
        
        images = batch['image']
        masks = batch['mask']
        batch_size = images.size(0)
        self.total_training_samples += batch_size
        
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
        
        # Get predicted class labels and calculate comprehensive metrics
        pred_labels = self.get_predicted_labels(outputs, masks)
        
        # For multi-class metrics, we need to ensure targets are in the expected format
        # All metrics expect class indices of type long, not one-hot or float
        masks_for_metrics = masks.clone()  # Make a copy to avoid modifying original
        
        # The metrics expect class indices as long type
        if masks_for_metrics.dim() == 4 and masks_for_metrics.size(1) == 1:
            masks_for_metrics = masks_for_metrics.squeeze(1)  # B, H, W
        
        # Convert to long type for metric calculations
        if masks_for_metrics.dtype != torch.long:
            if masks_for_metrics.dtype == torch.bool:
                # Boolean tensors need to be converted to long carefully
                masks_for_metrics = masks_for_metrics.long()
            else:
                # Float tensors should be rounded to nearest integer
                masks_for_metrics = torch.round(masks_for_metrics).long()
        
        # Calculate all metrics
        iou = self.iou_metric(pred_labels, masks_for_metrics)
        precision = self.precision_metric(pred_labels, masks_for_metrics)
        recall = self.recall_metric(pred_labels, masks_for_metrics)
        f1 = self.f1_metric(pred_labels, masks_for_metrics)
        dice = self.dice_metric(pred_labels, masks_for_metrics)
        specificity = self.specificity_metric(pred_labels, masks_for_metrics)
        
        # Calculate per-class metrics (for detailed analysis)
        iou_per_class = self.iou_per_class(pred_labels, masks_for_metrics)
        precision_per_class = self.precision_per_class(pred_labels, masks_for_metrics)
        recall_per_class = self.recall_per_class(pred_labels, masks_for_metrics)
        f1_per_class = self.f1_per_class(pred_labels, masks_for_metrics)
        
        # Update confusion matrix
        self.confusion_matrix.update(pred_labels, masks_for_metrics)
        
        # Calculate training efficiency metrics
        batch_time = time.time() - batch_start_time
        samples_per_second = batch_size / batch_time if batch_time > 0 else 0
        
        # Get current learning rate
        current_lr = self._optimizer.param_groups[0]['lr'] if self._optimizer else 0.0
        
        # Calculate gradient statistics
        total_grad_norm = 0.0
        param_count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                total_grad_norm += grad_norm.item() ** 2
                param_count += param.numel()
        
        total_grad_norm = (total_grad_norm ** 0.5) if total_grad_norm > 0 else 0.0
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        # Get GPU memory usage if available
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Training progress metrics
        training_metrics = {
            'batch_time': batch_time,
            'samples_per_second': samples_per_second,
            'learning_rate': current_lr,
            'gradient_norm': total_grad_norm,
            'avg_gradient_norm': avg_grad_norm,
            'gpu_memory_used_gb': gpu_memory_used,
            'gpu_memory_total_gb': gpu_memory_total,
            'cpu_percent': cpu_percent,
            'total_samples_seen': self.total_training_samples
        }
        
        # Log training progress metrics
        self.log_dict({f'progress/{k}': v for k, v in training_metrics.items()}, on_step=True)
        
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
        
        # Log comprehensive performance metrics
        self.log('train_precision', precision, on_step=True, on_epoch=True)
        self.log('train_recall', recall, on_step=True, on_epoch=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True)
        self.log('train_specificity', specificity, on_step=True, on_epoch=True)
        
        # Log per-class metrics for detailed analysis
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_per_class):
                self.log(f'train_iou_{class_name.lower()}', iou_per_class[i], on_step=False, on_epoch=True)
                self.log(f'train_precision_{class_name.lower()}', precision_per_class[i], on_step=False, on_epoch=True)
                self.log(f'train_recall_{class_name.lower()}', recall_per_class[i], on_step=False, on_epoch=True)
                self.log(f'train_f1_{class_name.lower()}', f1_per_class[i], on_step=False, on_epoch=True)
        
        return {
            'loss': loss,
            'train_iou': iou,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
            'train_dice': dice,
            'train_specificity': specificity,
            'output_stats': output_stats,
            'pred_stats': pred_stats,
            'training_metrics': training_metrics,
            'per_class_metrics': {
                'iou': iou_per_class,
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class
            }
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
        
        # Calculate comprehensive metrics for validation
        pred_labels = self.get_predicted_labels(outputs, masks)
        
        # Ensure masks are in the expected format for metric calculations
        masks_for_metrics = masks.clone()  # Make a copy to avoid modifying original
        
        # All metrics expect class indices as long type
        if masks_for_metrics.dim() == 4 and masks_for_metrics.size(1) == 1:
            masks_for_metrics = masks_for_metrics.squeeze(1)  # B, H, W
        
        # Convert to long type for metric calculations
        if masks_for_metrics.dtype != torch.long:
            if masks_for_metrics.dtype == torch.bool:
                # Boolean tensors need to be converted to long carefully
                masks_for_metrics = masks_for_metrics.long()
            else:
                # Float tensors should be rounded to nearest integer
                masks_for_metrics = torch.round(masks_for_metrics).long()
        
        # Calculate all validation metrics
        iou = self.iou_metric(pred_labels, masks_for_metrics)
        precision = self.precision_metric(pred_labels, masks_for_metrics)
        recall = self.recall_metric(pred_labels, masks_for_metrics)
        f1 = self.f1_metric(pred_labels, masks_for_metrics)
        dice = self.dice_metric(pred_labels, masks_for_metrics)
        specificity = self.specificity_metric(pred_labels, masks_for_metrics)
        
        # Calculate per-class metrics for validation
        iou_per_class = self.iou_per_class(pred_labels, masks_for_metrics)
        precision_per_class = self.precision_per_class(pred_labels, masks_for_metrics)
        recall_per_class = self.recall_per_class(pred_labels, masks_for_metrics)
        f1_per_class = self.f1_per_class(pred_labels, masks_for_metrics)
        
        # Log main validation metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', iou, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log comprehensive validation metrics
        self.log('val_precision', precision, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_recall', recall, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_f1', f1, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_dice', dice, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_specificity', specificity, on_step=True, on_epoch=True, sync_dist=True)
        
        # Log per-class validation metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_per_class):
                self.log(f'val_iou_{class_name.lower()}', iou_per_class[i], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'val_precision_{class_name.lower()}', precision_per_class[i], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'val_recall_{class_name.lower()}', recall_per_class[i], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'val_f1_{class_name.lower()}', f1_per_class[i], on_step=False, on_epoch=True, sync_dist=True)
        
        # Store for epoch end processing
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_iou': iou,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_dice': dice,
            'val_specificity': specificity
        })
        
        return {
            'val_loss': loss,
            'val_iou': iou,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_dice': dice,
            'val_specificity': specificity,
            'per_class_metrics': {
                'iou': iou_per_class,
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class
            }
        }


    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        try:
            # Aggregate validation metrics
            val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            val_iou = torch.stack([x['val_iou'] for x in self.validation_step_outputs]).mean()
            val_precision = torch.stack([x['val_precision'] for x in self.validation_step_outputs]).mean()
            val_recall = torch.stack([x['val_recall'] for x in self.validation_step_outputs]).mean()
            val_f1 = torch.stack([x['val_f1'] for x in self.validation_step_outputs]).mean()
            val_dice = torch.stack([x['val_dice'] for x in self.validation_step_outputs]).mean()
            val_specificity = torch.stack([x['val_specificity'] for x in self.validation_step_outputs]).mean()
            
            # Log aggregated metrics
            self.log('val_loss_epoch', val_loss, prog_bar=True)
            self.log('val_iou_epoch', val_iou, prog_bar=True)
            self.log('val_precision_epoch', val_precision, prog_bar=True)
            self.log('val_recall_epoch', val_recall, prog_bar=True)
            self.log('val_f1_epoch', val_f1, prog_bar=True)
            self.log('val_dice_epoch', val_dice, prog_bar=True)
            self.log('val_specificity_epoch', val_specificity, prog_bar=True)
            
            # Get current confusion matrix and log class-wise metrics
            if hasattr(self, 'confusion_matrix'):
                cm = self.confusion_matrix.compute()
                if cm is not None and cm.numel() > 0:
                    # Log confusion matrix statistics
                    self.log('confusion_matrix_trace', torch.trace(cm), prog_bar=False)
                    
                    # Calculate and log detailed confusion matrix metrics
                    self._log_confusion_matrix_analysis(cm)
                    
                    # Save confusion matrix for analysis
                    self._save_confusion_matrix(cm, self.current_epoch)
                    
                    # Reset confusion matrix for next epoch
                    self.confusion_matrix.reset()
            
            # Clear the outputs list
            self.validation_step_outputs.clear()
            
        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {str(e)}")
            logger.error(f"Available keys in validation outputs: {list(self.validation_step_outputs[0].keys()) if self.validation_step_outputs else 'None'}")

    def on_validation_epoch_start(self) -> None:
        """Clear the validation outputs at the start of each validation epoch."""
        self.validation_step_outputs = []
    
    def on_train_epoch_start(self) -> None:
        """Track epoch start time and reset counters."""
        self.epoch_start_time = time.time()
        self.total_training_samples = 0
        
    def on_train_epoch_end(self) -> None:
        """Log epoch-level training metrics."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            # Log epoch-level metrics
            self.log('epoch_time_minutes', epoch_time / 60.0, on_epoch=True)
            self.log('samples_per_epoch', self.total_training_samples, on_epoch=True)
            
            if epoch_time > 0:
                self.log('epoch_samples_per_second', self.total_training_samples / epoch_time, on_epoch=True)
    
    def _log_confusion_matrix_analysis(self, cm: torch.Tensor):
        """Log detailed confusion matrix analysis."""
        try:
            num_classes = cm.size(0)
            
            # Calculate per-class accuracy
            class_accuracies = torch.diag(cm) / torch.sum(cm, dim=1)
            
            # Calculate per-class precision, recall, and F1
            true_positives = torch.diag(cm)
            false_positives = torch.sum(cm, dim=0) - true_positives
            false_negatives = torch.sum(cm, dim=1) - true_positives
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Log per-class metrics from confusion matrix
            for i, class_name in enumerate(self.class_names):
                if i < num_classes:
                    self.log(f'cm_accuracy_{class_name.lower()}', class_accuracies[i], prog_bar=False)
                    self.log(f'cm_precision_{class_name.lower()}', precision[i], prog_bar=False)
                    self.log(f'cm_recall_{class_name.lower()}', recall[i], prog_bar=False)
                    self.log(f'cm_f1_{class_name.lower()}', f1[i], prog_bar=False)
            
            # Calculate overall metrics
            overall_accuracy = torch.trace(cm) / torch.sum(cm)
            macro_precision = torch.mean(precision)
            macro_recall = torch.mean(recall)
            macro_f1 = torch.mean(f1)
            
            # Calculate weighted metrics
            class_support = torch.sum(cm, dim=1)
            weighted_precision = torch.sum(precision * class_support) / torch.sum(class_support)
            weighted_recall = torch.sum(recall * class_support) / torch.sum(class_support)
            weighted_f1 = torch.sum(f1 * class_support) / torch.sum(class_support)
            
            # Log overall metrics
            self.log('cm_overall_accuracy', overall_accuracy, prog_bar=False)
            self.log('cm_macro_precision', macro_precision, prog_bar=False)
            self.log('cm_macro_recall', macro_recall, prog_bar=False)
            self.log('cm_macro_f1', macro_f1, prog_bar=False)
            self.log('cm_weighted_precision', weighted_precision, prog_bar=False)
            self.log('cm_weighted_recall', weighted_recall, prog_bar=False)
            self.log('cm_weighted_f1', weighted_f1, prog_bar=False)
            
            # Calculate and log class imbalance metrics
            class_distribution = class_support / torch.sum(class_support)
            entropy = -torch.sum(class_distribution * torch.log(class_distribution + 1e-8))
            max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float))
            normalized_entropy = entropy / max_entropy
            
            self.log('cm_class_entropy', entropy, prog_bar=False)
            self.log('cm_normalized_entropy', normalized_entropy, prog_bar=False)
            
        except Exception as e:
            logger.error(f"Error in confusion matrix analysis: {str(e)}")
    
    def _save_confusion_matrix(self, cm: torch.Tensor, epoch: int):
        """Save confusion matrix to file for later analysis."""
        try:
            import json
            
            # Create confusion matrix directory
            cm_dir = self.log_dir / 'confusion_matrices'
            cm_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy for JSON serialization
            cm_np = cm.detach().cpu().numpy()
            
            # Create confusion matrix data structure
            cm_data = {
                'epoch': epoch,
                'matrix': cm_np.tolist(),
                'class_names': self.class_names,
                'total_samples': int(torch.sum(cm).item()),
                'timestamp': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
            }
            
            # Save to JSON file
            cm_file = cm_dir / f'confusion_matrix_epoch_{epoch:03d}.json'
            with open(cm_file, 'w') as f:
                json.dump(cm_data, f, indent=2)
            
            # Also save latest confusion matrix
            latest_file = cm_dir / 'confusion_matrix_latest.json'
            with open(latest_file, 'w') as f:
                json.dump(cm_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving confusion matrix: {str(e)}")

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