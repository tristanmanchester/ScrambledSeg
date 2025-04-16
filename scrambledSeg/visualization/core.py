"""Core visualization module for segmentation results and training metrics."""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright')
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
import os
import pandas as pd
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SegmentationVisualizer:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        metrics_file: Optional[str] = None,
        min_coverage: float = 0.05,
        style: str = 'seaborn',
        dpi: int = 300,
        cmap: str = 'viridis'
    ):
        """Initialize visualizer."""
        self.dpi = dpi
        
        self.base_dir = Path(output_dir) if output_dir else None
        if self.base_dir:
            self.plots_dir = self.base_dir / 'plots'
            self.metrics_dir = self.base_dir / 'metrics'
            self.validation_dir = self.base_dir / 'validation'
            self.metrics_file = Path(metrics_file) if metrics_file else (self.metrics_dir / 'metrics.csv')
            
            # Create directories
            for dir_path in [self.plots_dir, self.metrics_dir, self.validation_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        else:
            self.plots_dir = None
            self.metrics_dir = None
            self.validation_dir = None
            self.metrics_file = Path(metrics_file) if metrics_file else None
        
        # Store other parameters
        self.min_coverage = min_coverage
        self.cmap = cmap
        self.mask_cmap = cmap
        
        # Set style with seaborn
        if style == 'seaborn':
            sns.set_theme()
        else:
            plt.style.use(style)
        
        # Initialize metrics history
        self.metrics_history = {}

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize tensor/array to [0, 1] range."""
        is_tensor = torch.is_tensor(tensor)
        
        if is_tensor:
            tensor = tensor.detach().cpu()
        
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]
        
        if is_tensor:
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
        else:
            min_val = np.min(tensor)
            max_val = np.max(tensor)
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
                
        return tensor

    def find_interesting_slices(self, masks, k: int = 4) -> List[int]:
        """Find k most interesting slices based on mask coverage."""
        coverages = []
        for i in range(len(masks)):
            # Get the mask
            mask = masks[i]
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()
                
            # For multi-class segmentation (integer labels), check for non-background
            # For binary segmentation (float values), use threshold
            if mask.dtype in [np.uint8, np.int32, np.int64]:
                # Multi-class case - calculate percent of non-background pixels
                coverage = float(np.mean(mask > 0))
            else:
                # Binary case - normalize and threshold
                mask = self._normalize_tensor(mask)
                coverage = float(np.mean(mask > 0.5))
                
            if coverage >= self.min_coverage:
                coverages.append((i, coverage))
        
        coverages.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in coverages[:k]]
        
        if len(indices) < k:
            logger.warning(
                f"Only found {len(indices)} slices with coverage >= {self.min_coverage}. "
                "Taking best available slices."
            )
            all_coverages = []
            for i in range(len(masks)):
                # Get the mask
                mask = masks[i]
                if torch.is_tensor(mask):
                    mask = mask.detach().cpu().numpy()
                    
                # For multi-class segmentation (integer labels), check for non-background
                # For binary segmentation (float values), use threshold
                if mask.dtype in [np.uint8, np.int32, np.int64]:
                    # Multi-class case - calculate percent of non-background pixels
                    coverage = float(np.mean(mask > 0))
                else:
                    # Binary case - normalize and threshold
                    mask = self._normalize_tensor(mask)
                    coverage = float(np.mean(mask > 0.5))
                    
                all_coverages.append((i, coverage))
                
            all_coverages.sort(key=lambda x: x[1], reverse=True)
            additional_indices = [i for i, _ in all_coverages if i not in indices]
            indices.extend(additional_indices[:k - len(indices)])
            
        return indices[:k]

    def visualize_prediction(
            self,
            image: Union[torch.Tensor, np.ndarray],
            mask: Union[torch.Tensor, np.ndarray],
            prediction: Union[torch.Tensor, np.ndarray],
            save_path: Optional[Path] = None,
            show_colorbar: bool = True,
            threshold: float = 0.5,
            class_values: Optional[np.ndarray] = None
        ) -> Optional[plt.Figure]:
            """Create visualization grid showing input, target, continuous predictions, sigmoid outputs, binary predictions, and difference map."""
            try:
                if torch.is_tensor(image):
                    image = image.detach().cpu().numpy()
                if torch.is_tensor(mask):
                    mask = mask.detach().cpu().numpy()
                # Handle multi-class prediction
                if torch.is_tensor(prediction):
                    # Apply softmax instead of sigmoid for multi-class
                    prediction_softmax = F.softmax(prediction, dim=1)
                    prediction = prediction.detach().cpu().numpy()
                    prediction_softmax = prediction_softmax.detach().cpu().numpy()
                    
                    # Get predicted class using argmax
                    pred_class = np.argmax(prediction_softmax, axis=1)
                    
                    # Create a one-hot encoded version of the ground truth mask
                    if torch.is_tensor(mask):
                        mask_np = mask.detach().cpu().numpy()
                    else:
                        mask_np = mask
                        
                    # Handle the case where mask has a channel dimension
                    if mask_np.ndim == 4 and mask_np.shape[1] == 1:
                        mask_np = mask_np.squeeze(1)  # Remove channel dimension
                    elif mask_np.ndim == 3 and mask_np.shape[0] == 1:
                        mask_np = mask_np.squeeze(0)  # Remove channel dimension
                else:
                    # Fallback for non-tensor inputs
                    prediction_softmax = prediction
                    pred_class = np.argmax(prediction, axis=1)
                    mask_np = mask

                image = self._normalize_tensor(image)
                
                # Create visual representations for multi-class segmentation
                # Display class indices directly (each class gets a different color)
                mask_display = mask_np if mask_np.ndim == 2 else mask_np[0]
                pred_class_display = pred_class[0] if pred_class.ndim > 2 else pred_class
                
                # Create difference map (1 where prediction differs from ground truth, 0 otherwise)
                diff_map = (pred_class_display != mask_display).astype(np.float32)
                
                # Create figure with GridSpec
                fig = plt.figure(figsize=(18, 12))
                gs = fig.add_gridspec(2, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05])
                
                axes = []
                cbar_axes = []
                for row in range(2):
                    for col in range(3):
                        ax = fig.add_subplot(gs[row, col * 2])
                        axes.append(ax)
                        if show_colorbar and (row > 0 or col > 0):
                            cbar_ax = fig.add_subplot(gs[row, col * 2 + 1])
                            cbar_axes.append(cbar_ax)

                # Use dataset-level class values if provided (ensures consistency across visualizations)
                try:
                    if class_values is not None:
                        all_classes = class_values
                        min_class = np.min(class_values)
                        max_class = np.max(class_values)
                        num_classes = len(class_values)
                        logger.info(f"Using dataset class values for visualization: {class_values}")
                    else:
                        # Find the actual unique classes in both mask and prediction for proper colormap limits
                        mask_unique = np.unique(mask_display)
                        pred_unique = np.unique(pred_class_display)
                        
                        # Combine all unique classes to ensure colorbar covers all possible values
                        all_classes = np.unique(np.concatenate([mask_unique, pred_unique]))
                        min_class = all_classes.min()
                        max_class = all_classes.max()
                        num_classes = len(all_classes)
                except Exception as e:
                    logger.warning(f"Error determining class values: {e}, using default range")
                    # Fallback to default range if there's any issue
                    min_class = 0
                    # Ensure we have at least 2 classes to avoid colormap issues
                    max_class = max(np.max(mask_display), np.max(pred_class_display), 1)
                    num_classes = int(max_class) + 1
                    all_classes = np.arange(num_classes)
                
                # Create a discrete colormap with exactly the number of classes we have
                # This ensures the colorbar doesn't have extra colors
                class_cmap = plt.cm.get_cmap('tab10', num_classes)
                
                # Get the number of classes from the prediction tensor for probability maps
                n_output_classes = prediction.shape[1] if torch.is_tensor(prediction) else prediction.shape[1]
                
                # Select which class probabilities to show (first two non-background classes if possible)
                prob_classes = []
                for c in range(1, min(n_output_classes, 3)):  # Skip background (0), show up to 2 classes
                    prob_classes.append(c)
                # If we don't have non-background classes, show background and first class
                if not prob_classes:
                    prob_classes = [0, min(1, n_output_classes-1)]
                
                # Define fixed ranges for different plot types
                plot_configs = [
                    (image, 'Input Image', 'gray', None),  # Input image can auto-scale
                    (mask_display, 'Ground Truth Classes', class_cmap, (min_class, max_class)),  # Class indices
                    (prediction_softmax[0, prob_classes[0]], f'Class {prob_classes[0]} Probability', 'viridis', (0, 1)),
                    (prediction_softmax[0, prob_classes[-1]], f'Class {prob_classes[-1]} Probability', 'viridis', (0, 1)),
                    (pred_class_display, 'Predicted Classes', class_cmap, (min_class, max_class)),  # Class indices
                    (diff_map, 'Difference Map', 'viridis', (0, 1))  # Binary difference
                ]
                
                for idx, (img_data, title, cmap, vlim) in enumerate(plot_configs):
                    # Force aspect ratio to be equal
                    im = axes[idx].imshow(img_data, cmap=cmap, aspect='equal')
                    if vlim is not None:
                        im.set_clim(*vlim)
                    
                    axes[idx].set_title(title)
                    # Ensure image fills subplot
                    axes[idx].set_position(axes[idx].get_position())
                    
                    if show_colorbar and idx > 0:
                        plt.colorbar(im, cax=cbar_axes[idx-1])
                
                # Turn off axes
                for ax in axes:
                    ax.axis('off')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close(fig)
                    return None
                return fig
                
            except Exception as e:
                logger.error(f"Error in visualize_prediction: {e}")
                if save_path:
                    plt.close('all')
                raise



    def plot_metrics(
        self,
        window_size: int = 200,
        save_path: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> Optional[plt.Figure]:
        """Plot training and validation metrics over time with loss and IoU on the same plot."""
        if not os.path.exists(self.metrics_file):
            logger.warning("No metrics file found yet. Skipping plot generation.")
            return None
            
        df = pd.read_csv(self.metrics_file)
        
        for col in df.columns:
            if col in ['step', 'epoch']:
                continue
            df[col] = pd.to_numeric(df[col].replace('', float('nan')), errors='coerce')
        
        if len(df) < 2:
            logger.warning("Not enough data points to plot metrics")
            return None
        
        # Create figure with single plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot loss on left y-axis
        loss_color = 'tab:red'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss', color=loss_color)
        
        # Initialize lines and labels for legend
        all_lines = []
        all_labels = []
        
        if not df['train_loss'].isna().all():
            # Plot training loss as a smoothed line
            line = ax1.plot(df['step'], 
                    df['train_loss'].rolling(window=min(window_size, len(df)), min_periods=1).mean(),
                    color=loss_color, label='Train Loss', alpha=0.5, linewidth=1)
            all_lines.extend(line)
            all_labels.append('Train Loss')
            
            # Add validation loss if available as scatter points
            if 'val_loss' in df.columns and not df['val_loss'].isna().all():
                # Filter out NaN values for scatter plot
                val_data = df[['step', 'val_loss']].dropna()
                scatter = ax1.scatter(val_data['step'], val_data['val_loss'],
                                    color=loss_color, label='Val Loss', alpha=0.9,
                                    marker='o', facecolors='none', s=30, linewidth=1)
                all_lines.append(scatter)
                all_labels.append('Val Loss')
        
        ax1.tick_params(axis='y', labelcolor=loss_color)
        
        # Create ax2 only if we have IoU data
        if 'train_iou' in df.columns and not df['train_iou'].isna().all():
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            iou_color = 'tab:blue'
            ax2.set_ylabel('IoU', color=iou_color)
            
            # Plot training IoU as a smoothed line
            line = ax2.plot(df['step'], 
                    df['train_iou'].rolling(window=min(window_size, len(df)), min_periods=1).mean(),
                    color=iou_color, label='Train IoU', alpha=0.5, linewidth=1)
            all_lines.extend(line)
            all_labels.append('Train IoU')
            
            # Add validation IoU if available as scatter points
            if 'val_iou' in df.columns and not df['val_iou'].isna().all():
                # Filter out NaN values for scatter plot
                val_data = df[['step', 'val_iou']].dropna()
                scatter = ax2.scatter(val_data['step'], val_data['val_iou'],
                                    color=iou_color, label='Val IoU', alpha=0.9,
                                    marker='o', facecolors='none', s=30, linewidth=1)
                all_lines.append(scatter)
                all_labels.append('Val IoU')
            
            ax2.tick_params(axis='y', labelcolor=iou_color)
            ax2.grid(False)
        
        # Add grid
        ax1.grid(True, alpha=0.15)
        
        # Add legend using collected lines and labels
        if all_lines:
            ax1.legend(all_lines, all_labels, loc='center right')
        
        plt.title('Training Metrics', pad=20)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig