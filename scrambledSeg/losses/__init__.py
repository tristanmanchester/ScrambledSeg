"""Loss functions for segmentation tasks."""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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

def create_loss(loss_type: str = "bcedice", **kwargs) -> nn.Module:
    """Create a loss function based on type."""
    if loss_type.lower() == "bcedice":
        return BCEDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")