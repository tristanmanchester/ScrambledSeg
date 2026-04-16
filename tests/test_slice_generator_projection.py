"""Tests for synthetic slice generator label conventions."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from scrambledSeg.generation import slice_generator_projection as sgp


def test_slice_generator_uses_distinct_documented_label_values() -> None:
    """The exported label constants should match the documented 0/1/2/3 mapping."""

    assert sgp.LABEL_OUT_OF_RECONSTRUCTION == 0
    assert sgp.LABEL_BACKGROUND == 1
    assert sgp.LABEL_ELECTROLYTE == 2
    assert sgp.LABEL_CATHODE == 3


def test_reconstruction_mask_includes_background_ring() -> None:
    """Background pixels should be inside the reconstruction mask."""

    label_image = np.array(
        [
            [sgp.LABEL_OUT_OF_RECONSTRUCTION, sgp.LABEL_BACKGROUND],
            [sgp.LABEL_ELECTROLYTE, sgp.LABEL_CATHODE],
        ],
        dtype=np.uint8,
    )

    recon_mask = label_image > sgp.LABEL_OUT_OF_RECONSTRUCTION

    expected = np.array([[False, True], [True, True]])
    assert np.array_equal(recon_mask, expected)
