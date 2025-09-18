"""Unit tests for ScrambledSeg loss functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("numpy")
from scrambledSeg.losses import (
    BCEDiceLoss,
    CompoundLoss,
    CrossEntropyDiceLoss,
    create_loss,
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

