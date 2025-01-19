"""SegFormer model implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import logging
from dataclasses import dataclass
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

@dataclass
class MiTConfig:
    """Configuration for Mix Transformer (MiT) variants."""
    depths: List[int]
    hidden_sizes: List[int]
    decoder_hidden_size: int
    params_m: float
    imagenet_top1: float

class MiTVariants:
    """Mix Transformer (MiT) model variants and their configurations."""
    
    VARIANTS = {
        'b0': MiTConfig([2, 2, 2, 2], [32, 64, 160, 256], 256, 3.7, 70.5),
        'b1': MiTConfig([2, 2, 2, 2], [64, 128, 320, 512], 256, 14.0, 78.7),
        'b2': MiTConfig([3, 4, 6, 3], [64, 128, 320, 512], 768, 25.4, 81.6),
        'b3': MiTConfig([3, 4, 18, 3], [64, 128, 320, 512], 768, 45.2, 83.1),
        'b4': MiTConfig([3, 8, 27, 3], [64, 128, 320, 512], 768, 62.6, 83.6),
        'b5': MiTConfig([3, 6, 40, 3], [64, 128, 320, 512], 768, 82.0, 83.8),
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> MiTConfig:
        """Get configuration for a specific model variant."""
        # Extract variant from model name (e.g., 'nvidia/segformer-b0-finetuned-ade-512-512' -> 'b0')
        for variant in cls.VARIANTS:
            if f'-{variant}-' in model_name or model_name.endswith(f'-{variant}'):
                return cls.VARIANTS[variant]
        raise ValueError(f"Could not determine MiT variant from model name: {model_name}")

# class RefinementModule(nn.Module):
#     @staticmethod
#     def _calculate_optimal_groups(channels: int) -> int:
#         """Calculate optimal number of groups for GroupNorm.
        
#         Args:
#             channels: Number of input channels
            
#         Returns:
#             Optimal number of groups ensuring 8-16 channels per group
#         """
#         # Get all factors of the channel count
#         factors = [i for i in range(1, channels + 1) if channels % i == 0]
        
#         # Calculate channels per group for each possible group count
#         channels_per_group = [(f, channels // f) for f in factors]
        
#         # Filter for options with 8-16 channels per group
#         valid_options = [(groups, cpg) for groups, cpg in channels_per_group if 8 <= cpg <= 16]
        
#         if not valid_options:
#             # If no perfect options, find closest to 8 channels per group
#             return max(1, channels // 8)
        
#         # Return the option with the most groups (more fine-grained normalization)
#         return max(valid_options, key=lambda x: x[0])[0]

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         mid_channels = in_channels // 2
#         num_groups = self._calculate_optimal_groups(mid_channels)
        
#         logger.info(f"Initializing RefinementModule with:")
#         logger.info(f"  in_channels={in_channels}")
#         logger.info(f"  mid_channels={mid_channels}")
#         logger.info(f"  num_groups={num_groups}")
#         logger.info(f"  channels_per_group={mid_channels/num_groups}")
        
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
#         self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels)
#         self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
#         self.dropout = nn.Dropout2d(p=0.1)
    
#     def forward(self, x):
#         x = F.relu(self.gn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = self.conv2(x)
#         return x

class CustomSegformer(nn.Module):
    def __init__(self, encoder_name="nvidia/mit-b0", num_classes=1, pretrained=True, 
                 ignore_mismatched_sizes=True, dropout_rate=0.1, enable_intermediate_supervision=True):
        super().__init__()
        
        logger.info(f"Initializing SegFormer with encoder: {encoder_name}")
        
        # Initialize the SegFormer backbone
        cache_dir = ".model_cache"  # Local cache directory
        logger.info(f"Using cache directory: {cache_dir}")
        
        # Try to load local model first
        try:
           self.backbone = transformers.SegformerModel.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=True # Added local files only to try local cache first
            )
           logger.info(f"Successfully loaded model from local cache.")
        except Exception as e:
            logger.warning(f"Could not load local model: {e}")
            logger.info(f"Attempting to download model from Hugging Face Hub")
            self.backbone = transformers.SegformerModel.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                output_hidden_states=True,
                cache_dir=cache_dir # Remove local_files_only to download
            )
            logger.info(f"Successfully downloaded and saved model to local cache.")
            
        
        # Get model configuration and actual encoder channels
        try:
            self.mit_config = MiTVariants.get_config(encoder_name)
            # Get actual encoder channels from the model
            encoder_hidden_states = self.backbone.encoder.layer_norm
            actual_channels = [layer.normalized_shape[0] for layer in encoder_hidden_states]
            logger.info(f"Using MiT variant with:")
            logger.info(f"  Expected hidden sizes: {self.mit_config.hidden_sizes}")
            logger.info(f"  Actual encoder channels: {actual_channels}")
            logger.info(f"  Decoder hidden size: {self.mit_config.decoder_hidden_size}")
            logger.info(f"  Model size: {self.mit_config.params_m}M parameters")
            self.encoder_channels = actual_channels
        except ValueError as e:
            logger.warning(f"Could not determine MiT variant: {e}")
            logger.warning("Using default B0 configuration")
            self.mit_config = MiTVariants.VARIANTS['b0']
            self.encoder_channels = self.mit_config.hidden_sizes
        
        
        # Create decoder components
        hidden_size = self.mit_config.decoder_hidden_size
        self.linear_c = nn.Linear(sum(self.encoder_channels), hidden_size)
        self.linear_fuse = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        
        # # Add refinement module using the decoder hidden size
        # self.refinement = RefinementModule(hidden_size, hidden_size)
        
        # Final classifier
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        
        # Add intermediate classifiers if enabled
        self.enable_intermediate_supervision = enable_intermediate_supervision
        if enable_intermediate_supervision:
            self.intermediate_classifiers = nn.ModuleList([
                nn.Conv2d(channels, num_classes, kernel_size=1)
                for channels in self.encoder_channels
            ])
        
        # Modify input conv layer for grayscale
        if pretrained:
            logger.info("Modifying input conv layer for 1 channel")
            old_conv = self.backbone.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride, padding=old_conv.padding)
            
            # Average the weights across RGB channels
            logger.info("Initializing input conv from pre-trained weights (averaging RGB channels)")
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                new_conv.bias.data = old_conv.bias.data
                
            self.backbone.encoder.patch_embeddings[0].proj = new_conv
            logger.info(f"Successfully modified input conv layer and added refinement modules")
    
    def _get_features(self, hidden_states):
        batch_size = hidden_states[0].shape[0]
        
        # Hidden states are already in [B, C, H, W] format
        hidden_states = [
            F.instance_norm(state)  # Normalize each channel independently
            for state in hidden_states
        ]
        
        # Resize to the same size
        target_size = hidden_states[0].shape[2:]
        hidden_states = [
            F.interpolate(
                feature, size=target_size, mode='bilinear', align_corners=False
            ) if feature.shape[2:] != target_size else feature
            for feature in hidden_states
        ]
        
        # Flatten and concatenate
        hidden_states = [
            state.flatten(2).transpose(1, 2) 
            for state in hidden_states
        ]
        hidden_states = torch.cat(hidden_states, dim=-1)
        
        # Apply linear projection
        hidden_states = self.linear_c(hidden_states)
        
        # Reshape back to spatial features
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, target_size[0], target_size[1]
        )
        
        return hidden_states
    
    def forward(self, x):
        """Forward pass."""
        # Get encoder features
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        encoder_features = outputs.hidden_states
        
        # Get decoder features
        decoder_features = self._get_features(encoder_features)
        # Remove this dropout
        # decoder_features = self.dropout(decoder_features)
        decoder_features = self.linear_fuse(decoder_features)
        
        # Get final logits through classifier
        logits = self.classifier(decoder_features)
        
        # Handle intermediate supervision if enabled
        intermediate_preds = []
        if self.enable_intermediate_supervision and self.training:
            for i, (feat, classifier) in enumerate(zip(encoder_features, self.intermediate_classifiers)):
                # Remove dropout here
                # feat = dropout(feat)
                pred = classifier(feat)
                pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
                intermediate_preds.append(pred)
        
        # Remove final dropout
        # logits = self.dropouts[-1](logits)
        
        # Upsample to input size
        h, w = x.shape[-2:]
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        
        if self.enable_intermediate_supervision and self.training:
            return logits, intermediate_preds
        return logits