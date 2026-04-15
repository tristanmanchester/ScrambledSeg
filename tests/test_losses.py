"""Unit tests for ScrambledSeg loss functions."""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("numpy")
from scrambledSeg.losses import (
    BCEDiceLoss,
    CompoundLoss,
    CrossEntropyDiceLoss,
    FocalLoss,
    LovaszSoftmaxLoss,
    TverskyLoss,
    create_loss,
    list_available_losses,
    lovasz_softmax_flat,
)

torch = pytest.importorskip("torch")


def test_bce_dice_loss_backward() -> None:
    """BCEDiceLoss should produce finite values and gradients."""

    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16), dtype=torch.float32)

    loss = BCEDiceLoss()(logits, target)
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None


def test_cross_entropy_dice_loss_backward() -> None:
    """CrossEntropyDiceLoss should support multi-class targets."""

    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randint(0, 3, (2, 16, 16), dtype=torch.long)

    loss = CrossEntropyDiceLoss()(logits, target)
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None


def test_compound_loss_combination() -> None:
    """CompoundLoss should combine component losses without NaNs."""

    logits = torch.randn(1, 4, 32, 32, requires_grad=True)
    target = torch.randint(0, 4, (1, 32, 32), dtype=torch.long)

    loss = CompoundLoss()(logits, target)
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None


def test_create_loss_factory_dispatch() -> None:
    """The ``create_loss`` factory should return the expected classes."""

    assert isinstance(create_loss("bcedice"), BCEDiceLoss)
    assert isinstance(create_loss("crossentropy_dice"), CrossEntropyDiceLoss)
    assert isinstance(create_loss("compound"), CompoundLoss)


def test_create_loss_accepts_aliases() -> None:
    """Hyphenated aliases should resolve via the loss factory."""

    loss = create_loss("crossentropy-dice")
    assert isinstance(loss, CrossEntropyDiceLoss)


def test_list_available_losses_includes_registered_entries() -> None:
    """The registry helper should expose normalized loss names."""

    registry = list_available_losses()
    assert "crossentropy_dice" in registry
    assert "compound" in registry


@pytest.mark.parametrize(
    "loss_factory",
    [
        lambda: FocalLoss(ignore_index=-100),
        lambda: TverskyLoss(ignore_index=-100),
        lambda: CrossEntropyDiceLoss(ignore_index=-100),
    ],
)
def test_multiclass_losses_ignore_index_matches_valid_only_pixels(loss_factory) -> None:
    """Ignore-index aware losses should match the valid-pixel-only computation."""

    logits = torch.tensor(
        [[[[2.0, -1.0, 0.5]], [[0.2, 3.0, -0.5]], [[-1.5, 0.1, 1.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1, -100]]], dtype=torch.long)

    full_loss = loss_factory()(logits, target)
    assert torch.isfinite(full_loss)
    full_loss.backward()
    assert logits.grad is not None

    valid_logits = logits.detach().clone()[..., :2].requires_grad_(True)
    valid_target = target[..., :2]
    valid_loss = loss_factory()(valid_logits, valid_target)

    assert torch.isfinite(valid_loss)
    assert full_loss.item() == pytest.approx(valid_loss.item(), rel=1e-6, abs=1e-6)


def test_lovasz_softmax_flat_ignore_matches_filtered_labels() -> None:
    """The flat Lovasz helper should drop ignored labels before computing the loss."""

    probas = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.3, 0.4],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, -100, 2], dtype=torch.long)
    valid = labels != -100

    ignored_loss = lovasz_softmax_flat(probas, labels, ignore=-100)
    filtered_loss = lovasz_softmax_flat(probas[valid], labels[valid], ignore=None)

    assert ignored_loss.item() == pytest.approx(filtered_loss.item(), rel=1e-6, abs=1e-6)


def test_lovasz_softmax_loss_ignore_matches_valid_only_pixels() -> None:
    """The module wrapper should exclude ignored pixels from the Lovasz loss."""

    logits = torch.tensor(
        [[[[2.0, -1.0, 0.5, 0.1]], [[0.2, 3.0, -0.5, 0.3]], [[-1.5, 0.1, 1.0, 1.8]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1, -100, 2]]], dtype=torch.long)

    ignored_loss = LovaszSoftmaxLoss(ignore=-100)(logits, target)
    assert torch.isfinite(ignored_loss)

    valid_logits = logits.detach().clone()[..., [0, 1, 3]].requires_grad_(True)
    valid_target = target[..., [0, 1, 3]]
    filtered_loss = LovaszSoftmaxLoss(ignore=-100)(valid_logits, valid_target)

    assert filtered_loss.item() == pytest.approx(ignored_loss.item(), rel=1e-6, abs=1e-6)
