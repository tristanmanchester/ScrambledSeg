"""SegFormer model implementation."""

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

logger = logging.getLogger(__name__)

DEFAULT_SEGFORMER_REVISIONS = {
    "nvidia/mit-b0": "80983a413c30d36a39c20203974ae7807835e2b4",
    "nvidia/segformer-b4-finetuned-ade-512-512": "2641fd1e2893964d8d473d8cf65a906cb0bff071",
}


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

    VARIANTS: Mapping[str, MiTConfig] = MappingProxyType(
        {
            "b0": MiTConfig([2, 2, 2, 2], [32, 64, 160, 256], 256, 3.7, 70.5),
            "b1": MiTConfig([2, 2, 2, 2], [64, 128, 320, 512], 256, 14.0, 78.7),
            "b2": MiTConfig([3, 4, 6, 3], [64, 128, 320, 512], 768, 25.4, 81.6),
            "b3": MiTConfig([3, 4, 18, 3], [64, 128, 320, 512], 768, 45.2, 83.1),
            "b4": MiTConfig([3, 8, 27, 3], [64, 128, 320, 512], 768, 62.6, 83.6),
            "b5": MiTConfig([3, 6, 40, 3], [64, 128, 320, 512], 768, 82.0, 83.8),
        }
    )

    @classmethod
    def get_config(cls, model_name: str) -> MiTConfig:
        """Get configuration for a specific model variant."""
        # Extract variant from model name (e.g., 'nvidia/segformer-b0-finetuned-ade-512-512' -> 'b0')
        for variant in cls.VARIANTS:
            if f"-{variant}-" in model_name or model_name.endswith(f"-{variant}"):
                return cls.VARIANTS[variant]
        raise ValueError(f"Could not determine MiT variant from model name: {model_name}")


class CustomSegformer(nn.Module):
    def __init__(
        self,
        encoder_name="nvidia/mit-b0",
        encoder_revision: str | None = None,
        num_classes=1,
        pretrained=True,
        ignore_mismatched_sizes=True,
    ):
        super().__init__()

        logger.debug("Initializing SegFormer with encoder %s", encoder_name)

        cache_dir = ".model_cache"
        logger.debug("Using cache directory %s", cache_dir)

        self.backbone = self._create_backbone(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            num_classes=num_classes,
            pretrained=pretrained,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            cache_dir=cache_dir,
        )

        try:
            self.mit_config = MiTVariants.get_config(encoder_name)
            encoder_hidden_states = self.backbone.encoder.layer_norm
            actual_channels = [layer.normalized_shape[0] for layer in encoder_hidden_states]
            logger.debug(
                "MiT variant hidden sizes expected=%s actual=%s decoder=%s params=%sM",
                self.mit_config.hidden_sizes,
                actual_channels,
                self.mit_config.decoder_hidden_size,
                self.mit_config.params_m,
            )
            self.encoder_channels = actual_channels
        except ValueError as e:
            logger.warning(f"Could not determine MiT variant: {e}")
            logger.warning("Using default B0 configuration")
            self.mit_config = MiTVariants.VARIANTS["b0"]
            self.encoder_channels = self.mit_config.hidden_sizes

        hidden_size = self.mit_config.decoder_hidden_size
        self.linear_c = nn.Linear(sum(self.encoder_channels), hidden_size)
        self.linear_fuse = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

        self._adapt_input_projection()

    def _create_backbone(
        self,
        *,
        encoder_name: str,
        encoder_revision: str | None,
        num_classes: int,
        pretrained: bool,
        ignore_mismatched_sizes: bool,
        cache_dir: str,
    ) -> nn.Module:
        """Construct the backbone with or without pretrained weights."""
        if pretrained:
            return self._load_pretrained_backbone(
                encoder_name=encoder_name,
                encoder_revision=encoder_revision,
                num_classes=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                cache_dir=cache_dir,
            )

        config = self._load_backbone_config(
            encoder_name=encoder_name,
            encoder_revision=encoder_revision,
            num_classes=num_classes,
            cache_dir=cache_dir,
        )
        logger.debug("Building SegFormer backbone from config without pretrained weights")
        return transformers.SegformerModel(config)

    @staticmethod
    def _resolve_encoder_revision(
        encoder_name: str,
        encoder_revision: str | None,
    ) -> str | None:
        """Return the immutable revision used for Hugging Face cache lookups."""
        if encoder_revision is not None:
            return encoder_revision
        return DEFAULT_SEGFORMER_REVISIONS.get(encoder_name)

    @classmethod
    def _require_remote_revision(
        cls,
        *,
        encoder_name: str,
        encoder_revision: str | None,
    ) -> str:
        """Require a pinned revision before downloading model artifacts."""
        resolved_revision = cls._resolve_encoder_revision(encoder_name, encoder_revision)
        if resolved_revision is None:
            raise ValueError(
                "Downloading Hugging Face checkpoints requires a pinned encoder_revision. "
                f"Provide encoder_revision explicitly for '{encoder_name}' or pre-populate the "
                "local cache."
            )
        return resolved_revision

    @classmethod
    def _load_pretrained_backbone(
        cls,
        *,
        encoder_name: str,
        encoder_revision: str | None,
        num_classes: int,
        ignore_mismatched_sizes: bool,
        cache_dir: str,
    ) -> nn.Module:
        """Load pretrained weights, preferring the local cache when possible."""
        resolved_revision = cls._resolve_encoder_revision(encoder_name, encoder_revision)
        try:
            backbone = transformers.SegformerModel.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                output_hidden_states=True,
                cache_dir=cache_dir,
                local_files_only=True,
                revision=resolved_revision,
                use_safetensors=True,
            )
            logger.debug("Loaded model weights from local cache")
            return backbone
        except Exception as exc:
            logger.warning(f"Could not load local model: {exc}")
            logger.info("Downloading model weights from Hugging Face Hub")
            remote_revision = cls._require_remote_revision(
                encoder_name=encoder_name,
                encoder_revision=encoder_revision,
            )
            backbone = transformers.SegformerModel.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                output_hidden_states=True,
                cache_dir=cache_dir,
                revision=remote_revision,
                use_safetensors=True,
            )
            logger.debug("Downloaded model weights and populated the local cache")
            return backbone

    @classmethod
    def _load_backbone_config(
        cls,
        *,
        encoder_name: str,
        encoder_revision: str | None,
        num_classes: int,
        cache_dir: str,
    ) -> transformers.SegformerConfig:
        """Load a backbone config without pulling pretrained weights."""
        resolved_revision = cls._resolve_encoder_revision(encoder_name, encoder_revision)
        try:
            config = transformers.SegformerConfig.from_pretrained(
                encoder_name,
                cache_dir=cache_dir,
                local_files_only=True,
                revision=resolved_revision,
            )
        except Exception as exc:
            logger.warning(f"Could not load local model config: {exc}")
            logger.info("Downloading model config from Hugging Face Hub")
            remote_revision = cls._require_remote_revision(
                encoder_name=encoder_name,
                encoder_revision=encoder_revision,
            )
            config = transformers.SegformerConfig.from_pretrained(
                encoder_name,
                cache_dir=cache_dir,
                revision=remote_revision,
            )

        config.num_labels = num_classes
        config.output_hidden_states = True
        return config

    def _adapt_input_projection(self) -> None:
        """Adapt the first patch embedding to grayscale input."""
        old_conv = self.backbone.encoder.patch_embeddings[0].proj
        if old_conv.in_channels == 1:
            return

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data

        self.backbone.encoder.patch_embeddings[0].proj = new_conv
        logger.debug("Adapted the encoder input convolution for grayscale input")

    def _get_features(self, hidden_states):
        batch_size = hidden_states[0].shape[0]
        target_size = hidden_states[0].shape[2:]

        hidden_states = [F.instance_norm(state) for state in hidden_states]

        hidden_states = [
            F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            if feature.shape[2:] != target_size
            else feature
            for feature in hidden_states
        ]

        hidden_states = [state.flatten(2).transpose(1, 2) for state in hidden_states]
        hidden_states = torch.cat(hidden_states, dim=-1)
        hidden_states = self.linear_c(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, target_size[0], target_size[1]
        )

        return hidden_states

    def forward(self, x):
        """Forward pass."""
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        encoder_features = outputs.hidden_states

        decoder_features = self._get_features(encoder_features)
        decoder_features = self.linear_fuse(decoder_features)
        logits = self.classifier(decoder_features)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return logits
