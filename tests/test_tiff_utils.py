"""Direct tests for TIFF tiling helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from scrambledSeg.prediction.tiff_utils import TiffHandler, TiffInputKind, TiffOutputKind


def test_tiff_handler_merges_multiclass_tiles_without_changing_labels() -> None:
    """Class-index tiles should merge back to the original labels."""

    handler = TiffHandler(tile_size=2, overlap=0)
    tile = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    tiles = [(tile, (slice(0, 2), slice(0, 2), True, True))]

    merged = handler.merge_tiles(
        tiles,
        output_shape=(2, 2),
        output_kind=TiffOutputKind.LABELS,
    )

    assert np.array_equal(merged, tile)


def test_tiff_handler_iter_tiles_pads_edge_tiles() -> None:
    """Edge tiles should be padded back to the configured tile size."""

    handler = TiffHandler(tile_size=4, overlap=0)
    image = np.arange(9, dtype=np.uint8).reshape(3, 3)

    tiles = list(handler.iter_tiles(image, input_kind=TiffInputKind.IMAGE))

    assert len(tiles) == 1
    tile, _ = tiles[0]
    assert tile.shape == (4, 4)
    assert np.array_equal(tile[:3, :3], image)


def test_tiff_handler_loads_multipage_tiffs(tmp_path: Path) -> None:
    """Direct TIFF loads should preserve multipage stack shapes."""

    handler = TiffHandler(tile_size=2, overlap=0)
    image = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
    image_path = tmp_path / "stack.tif"
    tifffile.imwrite(image_path, image)

    loaded = handler.load_tiff(image_path)

    assert np.array_equal(loaded, image)


def test_tiff_handler_requires_explicit_kind_for_ambiguous_3d_shapes() -> None:
    """Auto TIFF inference should reject 3D arrays that could be images or stacks."""

    handler = TiffHandler(tile_size=2, overlap=0)
    ambiguous = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)

    with pytest.raises(ValueError, match="Ambiguous 3D TIFF shape"):
        handler.resolve_input_kind(ambiguous, TiffInputKind.AUTO)


def test_tiff_handler_saves_probability_outputs_without_thresholding(tmp_path: Path) -> None:
    """Probability TIFF output should preserve floating-point scores."""

    handler = TiffHandler(tile_size=2, overlap=0)
    data = np.full((4, 4), 0.75, dtype=np.float32)
    output_path = tmp_path / "probabilities.tif"

    handler.save_tiff(data, output_path, output_kind=TiffOutputKind.PROBABILITIES)

    loaded = tifffile.imread(output_path)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, data)
