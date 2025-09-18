"""Loss functions for segmentation tasks."""

import logging
from typing import Dict, List, Optional, Union, Tuple, Callable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Lovász loss implementation based on PyTorch version
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor, 
    labels: torch.Tensor, 
    classes: Union[List, str] = 'present',
    ignore: Optional[int] = None
) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    ignore: void class labels
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovasz-Softmax loss.
    Based on Lovasz-Softmax and Jaccard hinge loss in "Lovasz-Softmax: A Tractable Surrogate 
    for the Optimization of the Intersection-Over-Union Measure in Neural Networks"
    https://arxiv.org/abs/1705.08790
    
    Adapted to work with multi-class segmentation.
    """
    def __init__(self, classes: Union[List, str] = 'present', per_image: bool = False, ignore: Optional[int] = None):
        """
        Args:
            classes: 'all' for all, 'present' for classes present in labels, 
                     or a list of classes to average.
            per_image: compute the loss per image instead of per batch
            ignore: void class label
        """
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions [B, C, H, W], logits (before softmax)
            target: Ground truth class indices [B, H, W] or [B, 1, H, W]
            
        Returns:
            Lovasz loss value
        """
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            
        if pred.dim() == 4:
            # Apply softmax along class dimension
            pred_softmax = F.softmax(pred, dim=1)
            
            # Flatten both predictions and targets
            B, C, H, W = pred.shape
            pred_flat = pred_softmax.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            target_flat = target.reshape(-1)  # [B*H*W]
            
            if self.per_image:
                losses = []
                for i in range(B):
                    img_pred = pred_softmax[i].permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
                    img_target = target[i].reshape(-1)  # [H*W]
                    losses.append(lovasz_softmax_flat(img_pred, img_target, self.classes, self.ignore))
                return torch.mean(torch.stack(losses))
            else:
                return lovasz_softmax_flat(pred_flat, target_flat, self.classes, self.ignore)
        else:
            # Single image case or already flattened
            return lovasz_softmax_flat(pred, target, self.classes, self.ignore)


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0
    ):
        """
        Combined BCE and Dice loss for binary segmentation tasks.
        
        Args:
            bce_weight: Weight for BCE loss component
            dice_weight: Weight for Dice loss component
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce_criterion = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss component."""
        pred = torch.sigmoid(pred)
        
        # Calculate Dice coefficient
        dims = (2, 3) if pred.dim() == 4 else (1, 2)
        intersection = (pred * target).sum(dims)
        cardinality = (pred + target).sum(dims)
        
        dice = 1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        return dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining BCE and Dice loss.
        
        Args:
            pred: Model predictions (before sigmoid)
            target: Ground truth binary masks
            
        Returns:
            Combined weighted loss
        """
        bce = self.bce_criterion(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance.
    
    Reference:
    - Lin et al., Focal Loss for Dense Object Detection, ICCV 2017
    """
    def __init__(
        self, 
        alpha: Union[float, List[float]] = 0.25, 
        gamma: float = 2.0, 
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Args:
            alpha: Weighting factor for the rare class, can be a list for multi-class.
                  If a list, should be of length num_classes.
            gamma: Focusing parameter. Higher gamma gives more weight to hard examples.
            reduction: 'none', 'mean', 'sum'
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (B, C, H, W) - raw logits
            targets: Ground truth class indices (B, H, W) or (B, 1, H, W)
            
        Returns:
            Focal loss value
        """
        # For CrossEntropyLoss, target should be class indices (B, H, W)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # Remove channel dimension
        
        # Convert to the correct data type
        if targets.dtype != torch.long:
            targets = targets.long()
            
        # Get number of classes from inputs
        num_classes = inputs.size(1)
        
        # Apply softmax to get class probabilities
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Get probabilities for the target classes
        pt = (inputs_softmax * targets_one_hot).sum(dim=1)
        
        # Cast ignore index to a mask
        mask = (targets != self.ignore_index).float()
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if isinstance(self.alpha, (list, tuple, np.ndarray)):
            # Alpha is a list for each class
            assert len(self.alpha) == num_classes, "Alpha should have same length as num_classes"
            alpha_t = torch.tensor(self.alpha, device=inputs.device)
            # Apply alpha for each class in the target
            alpha_t = alpha_t.gather(0, targets.clamp(0))  # Clamp to handle -100
        else:
            # Alpha is a single value - convert to tensor
            alpha_t = torch.ones_like(pt) * self.alpha
            
        # Calculate loss with focal weighting
        loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)
        
        # Apply mask for ignore index
        loss = loss * mask
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # mean
            return loss.sum() / (mask.sum() + 1e-8)


class TverskyLoss(nn.Module):
    """
    Tversky loss for handling imbalanced data by giving different weights 
    to false positives and false negatives.
    
    Reference:
    - Salehi et al., Tversky loss function for image segmentation using 3D fully convolutional deep networks, 2017
    """
    def __init__(
        self, 
        alpha: float = 0.5, 
        beta: float = 0.5, 
        smooth: float = 1.0,
        ignore_index: int = -100
    ):
        """
        Args:
            alpha: Weight of false positives
            beta: Weight of false negatives
            smooth: Smoothing constant
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (B, C, H, W) - raw logits
            targets: Ground truth class indices (B, H, W) or (B, 1, H, W)
            
        Returns:
            Tversky loss value
        """
        # For multi-class, target should be class indices (B, H, W)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # Remove channel dimension
        
        # Convert to the correct data type
        if targets.dtype != torch.long:
            targets = targets.long()
            
        # Get number of classes from inputs
        num_classes = inputs.size(1)
        
        # Apply softmax to get class probabilities
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Cast ignore index to a mask (1 for valid pixels, 0 for ignored)
        mask = (targets != self.ignore_index).float().unsqueeze(1).expand_as(inputs_softmax)
        
        # Calculate Tversky index for each class
        losses = []
        for class_idx in range(num_classes):
            # Get probabilities and targets for this class
            inputs_class = inputs_softmax[:, class_idx]
            targets_class = targets_one_hot[:, class_idx]
            mask_class = mask[:, class_idx]
            
            # True positives, false positives, false negatives
            tp = (inputs_class * targets_class * mask_class).sum()
            fp = (inputs_class * (1 - targets_class) * mask_class).sum()
            fn = ((1 - inputs_class) * targets_class * mask_class).sum()
            
            # Tversky index with alpha/beta weighting
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss = 1 - tversky
            losses.append(loss)
            
        # Average loss across all classes
        return torch.stack(losses).mean()


class CompoundLoss(nn.Module):
    """
    Compound loss combining Lovász, Focal, and Tversky losses for robust 
    multi-class segmentation, particularly effective for imbalanced datasets
    with small structures like solid state battery materials.
    """
    def __init__(
        self,
        lovasz_weight: float = 0.4,
        focal_weight: float = 0.3,
        tversky_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: Union[float, List[float]] = 0.25,
        tversky_alpha: float = 0.3,  # Lower alpha means lower false positive penalty
        tversky_beta: float = 0.7,   # Higher beta means higher false negative penalty
        smooth: float = 1.0,
        ignore_index: int = -100,
        classes: str = 'present'
    ):
        """
        Args:
            lovasz_weight: Weight for Lovász loss component
            focal_weight: Weight for Focal loss component
            tversky_weight: Weight for Tversky loss component
            focal_gamma: Focusing parameter for Focal loss
            focal_alpha: Class weight for Focal loss
            tversky_alpha: Weight of false positives in Tversky loss
            tversky_beta: Weight of false negatives in Tversky loss
            smooth: Smoothing factor
            ignore_index: Index to ignore in loss calculation
            classes: 'all' for all, 'present' for classes present in labels
        """
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # Initialize component losses
        self.lovasz_loss = LovaszSoftmaxLoss(classes=classes, per_image=False, ignore=ignore_index)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, 
                                       smooth=smooth, ignore_index=ignore_index)
        
        # Validation
        weights_sum = lovasz_weight + focal_weight + tversky_weight
        if abs(weights_sum - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {weights_sum} instead of 1.0. "
                          f"Normalizing weights.")
            self.lovasz_weight /= weights_sum
            self.focal_weight /= weights_sum
            self.tversky_weight /= weights_sum
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining all loss components.
        
        Args:
            pred: Model predictions (B, C, H, W) - raw logits
            target: Ground truth class indices (B, H, W) or (B, 1, H, W)
            
        Returns:
            Combined weighted loss
        """
        # Calculate individual losses
        lovasz = self.lovasz_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        # Log individual loss components for debugging
        if torch.isnan(lovasz) or torch.isnan(focal) or torch.isnan(tversky):
            logger.error(f"NaN in losses: Lovász={lovasz.item()}, "
                        f"Focal={focal.item()}, Tversky={tversky.item()}")
            
        # Combine losses with weights
        return (self.lovasz_weight * lovasz + 
                self.focal_weight * focal + 
                self.tversky_weight * tversky)


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        ignore_index: int = -100
    ):
        """
        Combined Cross Entropy and Dice loss for multi-class segmentation tasks.
        
        Args:
            ce_weight: Weight for Cross Entropy loss component
            dice_weight: Weight for Dice loss component
            smooth: Smoothing factor for Dice loss
            ignore_index: Index to ignore in CE loss calculation
        """
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def multi_class_dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-class Dice loss component.
        
        Args:
            pred: Model predictions (B, C, H, W) after softmax
            target: One-hot encoded ground truth masks (B, C, H, W)
            
        Returns:
            Average Dice loss across all classes
        """
        # Calculate Dice coefficient for each class
        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (pred * target).sum(dims)
        cardinality = (pred + target).sum(dims)
        
        dice_per_class = 1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        
        # Return mean across all classes except background if needed
        return dice_per_class.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining Cross Entropy and Dice loss.
        
        Args:
            pred: Model predictions (B, C, H, W) - raw logits
            target: Ground truth class indices (B, 1, H, W) or (B, H, W)
            
        Returns:
            Combined weighted loss
        """
        # For CrossEntropyLoss, target should be class indices (B, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # Remove channel dimension
        
        # Ensure target is long type for CrossEntropyLoss
        if target.dtype != torch.long:
            # Convert any float targets to long by rounding
            if target.dtype == torch.float32 or target.dtype == torch.float64:
                target = torch.round(target).long()
            else:
                target = target.long()
        
        # Calculate Cross Entropy loss
        ce_loss = self.ce_criterion(pred, target)
        
        # For Dice loss, we need softmax probabilities and one-hot encoded target
        pred_softmax = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoded format
        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice loss
        dice_loss = self.multi_class_dice_loss(pred_softmax, target_one_hot)
        
        # Combine losses
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def create_loss(loss_type: str = "crossentropy_dice", **kwargs) -> nn.Module:
    """Create a loss function based on type.
    
    Args:
        loss_type: Type of loss function to create:
            - "bcedice": Binary Cross Entropy + Dice Loss
            - "crossentropy_dice": Cross Entropy + Dice Loss for multi-class
            - "crossentropy": Pure Cross Entropy Loss
            - "lovasz": Lovász Softmax Loss for multi-class
            - "focal": Focal Loss for handling class imbalance
            - "tversky": Tversky Loss for balancing precision and recall
            - "compound": Combined Lovász + Focal + Tversky for battery segmentation
        **kwargs: Additional arguments to pass to the loss function
        
    Returns:
        The instantiated loss function
    """
    if loss_type.lower() == "bcedice":
        return BCEDiceLoss(**kwargs)
    elif loss_type.lower() == "crossentropy_dice":
        return CrossEntropyDiceLoss(**kwargs)
    elif loss_type.lower() == "crossentropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type.lower() == "lovasz":
        return LovaszSoftmaxLoss(**kwargs)
    elif loss_type.lower() == "focal":
        return FocalLoss(**kwargs)
    elif loss_type.lower() == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type.lower() == "compound":
        return CompoundLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
    # Usage example in config:
    # loss:
    #   type: "compound"  # Use compound loss for battery segmentation
    #   params:
    #     lovasz_weight: 0.4
    #     focal_weight: 0.3
    #     tversky_weight: 0.3
    #     focal_gamma: 2.0
    #     tversky_alpha: 0.3
    #     tversky_beta: 0.7
