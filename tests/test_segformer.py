"""Direct tests for the SegFormer model wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import scrambledSeg.models.segformer as segformer_module
from scrambledSeg.models.segformer import CustomSegformer, MiTVariants


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SimpleNamespace(
            layer_norm=[
                SimpleNamespace(normalized_shape=(32,)),
                SimpleNamespace(normalized_shape=(64,)),
                SimpleNamespace(normalized_shape=(160,)),
                SimpleNamespace(normalized_shape=(256,)),
            ],
            patch_embeddings=[
                SimpleNamespace(
                    proj=torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
                )
            ],
        )


def test_mit_variants_detects_variant_from_model_name() -> None:
    """Variant lookup should match common Hugging Face model names."""

    config = MiTVariants.get_config("nvidia/segformer-b2-finetuned-ade-512-512")

    assert config.hidden_sizes == [64, 128, 320, 512]


def test_custom_segformer_adapts_pretrained_backbone_for_grayscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretrained initialization should replace the RGB input conv with a grayscale version."""

    fake_backbone = _FakeBackbone()
    captured_calls: list[dict[str, object]] = []

    def _fake_from_pretrained(*args, **kwargs):
        captured_calls.append(kwargs)
        return fake_backbone

    monkeypatch.setattr(
        segformer_module.transformers.SegformerModel,
        "from_pretrained",
        _fake_from_pretrained,
    )

    model = CustomSegformer(encoder_name="nvidia/mit-b0", pretrained=True)

    assert model.encoder_channels == [32, 64, 160, 256]
    assert model.backbone.encoder.patch_embeddings[0].proj.in_channels == 1
    assert captured_calls == [
        {
            "num_labels": 1,
            "ignore_mismatched_sizes": True,
            "output_hidden_states": True,
            "cache_dir": ".model_cache",
            "local_files_only": True,
            "revision": segformer_module.DEFAULT_SEGFORMER_REVISIONS["nvidia/mit-b0"],
            "use_safetensors": True,
        }
    ]


def test_custom_segformer_requires_pinned_revision_for_uncached_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown remote models should fail closed instead of downloading an unpinned revision."""

    def _missing_cache(*args, **kwargs):
        raise OSError("model not cached")

    monkeypatch.setattr(
        segformer_module.transformers.SegformerModel,
        "from_pretrained",
        _missing_cache,
    )

    with pytest.raises(ValueError, match="requires a pinned encoder_revision"):
        CustomSegformer(encoder_name="custom/segformer", pretrained=True)
