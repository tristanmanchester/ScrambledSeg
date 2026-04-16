"""Utilities for handling large TIFF images with tiling."""

import logging
import math
from enum import Enum
from typing import Iterator, List, Tuple

import numpy as np
import tifffile
from PIL import Image

from .errors import PredictionDataAccessError, PredictionInputError
from .types import PathLike, TiledPrediction, TileLocation

logger = logging.getLogger(__name__)
DEFAULT_TILE_OVERLAP = 32


class TiffInputKind(Enum):
    """How a TIFF array should be interpreted at the prediction boundary."""

    AUTO = "auto"
    IMAGE = "image"
    STACK = "stack"

    @classmethod
    def parse(cls, value: "TiffInputKind | str") -> "TiffInputKind":
        """Normalize string input into a TIFF input-kind enum."""

        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        aliases = {
            "auto": cls.AUTO,
            "image": cls.IMAGE,
            "stack": cls.STACK,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise PredictionInputError(
                f"Invalid TIFF input kind '{value}'. Choose from {[kind.value for kind in cls]}."
            ) from exc


class TiffOutputKind(Enum):
    """How TIFF predictions should be merged and persisted."""

    LABELS = "labels"
    PROBABILITIES = "probabilities"

    @classmethod
    def parse(cls, value: "TiffOutputKind | str") -> "TiffOutputKind":
        """Normalize string input into a TIFF output-kind enum."""

        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        aliases = {
            "labels": cls.LABELS,
            "probabilities": cls.PROBABILITIES,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise PredictionInputError(
                f"Invalid TIFF output kind '{value}'. Choose from {[kind.value for kind in cls]}."
            ) from exc


class TiffHandler:
    """Load, save, and tile TIFF images."""

    def __init__(self, tile_size: int = 512, overlap: int = DEFAULT_TILE_OVERLAP):
        """Create a TIFF helper with fixed tile geometry."""
        if overlap < 0 or overlap >= tile_size:
            raise PredictionInputError(f"overlap must be in [0, {tile_size}), got {overlap}")
        self.tile_size = tile_size
        self.overlap = overlap
        self.trim_size = overlap // 2

    def load_tiff(self, path: PathLike) -> np.ndarray:
        """Load a TIFF image as a NumPy array."""
        try:
            data = tifffile.imread(path)
        except Exception as e:
            logger.warning(f"Failed to load with tifffile, falling back to PIL: {e}")
            try:
                with Image.open(path) as img:
                    data = np.array(img)
            except Exception as pil_exc:
                raise PredictionDataAccessError(f"Could not load TIFF input {path}") from pil_exc

        logger.debug("Loaded image with shape %s and dtype %s", data.shape, data.dtype)
        return data

    def resolve_input_kind(
        self, image: np.ndarray, input_kind: TiffInputKind | str = TiffInputKind.AUTO
    ) -> TiffInputKind:
        """Validate and resolve how a TIFF array should be interpreted."""

        resolved_kind = TiffInputKind.parse(input_kind)
        if resolved_kind is TiffInputKind.IMAGE:
            self._validate_image_array(image)
            return resolved_kind
        if resolved_kind is TiffInputKind.STACK:
            self._validate_stack_array(image)
            return resolved_kind

        if image.ndim == 2:
            return TiffInputKind.IMAGE
        if image.ndim != 3:
            raise PredictionInputError(
                f"Unsupported TIFF shape {image.shape}. Expected a 2D image, a channel-last "
                "3D image, or a 3D stack."
            )
        if image.shape[2] <= 4:
            raise PredictionInputError(
                "Ambiguous 3D TIFF shape "
                f"{image.shape}. Pass input_kind='image' for channel-last TIFF images or "
                "input_kind='stack' for multi-slice stacks."
            )

        return TiffInputKind.STACK

    def _validate_image_array(self, image: np.ndarray) -> None:
        """Require a 2D image or a channel-last image with a small channel count."""

        if image.ndim == 2:
            return
        if image.ndim == 3 and image.shape[2] <= 4:
            return
        raise PredictionInputError(
            f"TIFF image input must be 2D or channel-last with <=4 channels, got shape "
            f"{image.shape}."
        )

    def _validate_stack_array(self, image: np.ndarray) -> None:
        """Require a 3D stack shaped like ``(z, height, width)``."""

        if image.ndim != 3:
            raise PredictionInputError(
                f"TIFF stack input must be a 3D array shaped (z, h, w), got shape {image.shape}."
            )

    def save_tiff(
        self,
        data: np.ndarray,
        path: PathLike,
        output_kind: TiffOutputKind | str = TiffOutputKind.LABELS,
    ) -> None:
        """Persist a prediction array as a TIFF image."""

        resolved_output_kind = TiffOutputKind.parse(output_kind)

        if resolved_output_kind is TiffOutputKind.LABELS:
            if np.issubdtype(data.dtype, np.floating):
                rounded = np.rint(data)
                if not np.allclose(data, rounded):
                    raise PredictionInputError(
                        "Label TIFF output must contain integer-like class indices. "
                        "Use output_kind='probabilities' to save floating-point scores."
                    )
                data = rounded

            max_label = int(np.max(data)) if data.size else 0
            label_dtype = np.uint8 if max_label <= np.iinfo(np.uint8).max else np.uint16
            data = data.astype(label_dtype, copy=False)
        else:
            data = data.astype(np.float32, copy=False)

        if resolved_output_kind is TiffOutputKind.PROBABILITIES or (
            data.ndim in [3, 4] and (data.shape[0] > 1 or data.ndim == 4)
        ):
            tifffile.imwrite(path, data)
        else:
            img = Image.fromarray(data)
            img.save(path)

        logger.debug(
            "Saved image to %s with shape %s and range [%s, %s]",
            path,
            data.shape,
            np.min(data),
            np.max(data),
        )

    def get_tile_locations(self, image_shape: Tuple[int, ...]) -> List[TileLocation]:
        """Return tile locations for a given image shape."""
        height, width = image_shape[:2]
        effective_tile_size = self.tile_size - self.overlap
        n_tiles_h = math.ceil(height / effective_tile_size)
        n_tiles_w = math.ceil(width / effective_tile_size)

        tile_locations = []
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                row_start = i * effective_tile_size
                col_start = j * effective_tile_size

                is_edge_row = i == 0 or i == n_tiles_h - 1
                is_edge_col = j == 0 or j == n_tiles_w - 1

                if i == n_tiles_h - 1:
                    row_start = max(0, height - self.tile_size)
                if j == n_tiles_w - 1:
                    col_start = max(0, width - self.tile_size)

                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)

                tile_locations.append(
                    (slice(row_start, row_end), slice(col_start, col_end), is_edge_row, is_edge_col)
                )

        return tile_locations

    def merge_tiles(
        self,
        tiles: List[TiledPrediction],
        output_shape: Tuple[int, ...],
        output_kind: TiffOutputKind | str = TiffOutputKind.LABELS,
    ) -> np.ndarray:
        """Merge predicted tiles back into a single image."""

        resolved_output_kind = TiffOutputKind.parse(output_kind)
        if resolved_output_kind is TiffOutputKind.LABELS:
            logger.debug("Merging label tiles")
            return self._merge_tiles_labels(tiles, output_shape)

        logger.debug("Merging probability tiles")
        return self._merge_tiles_probabilities(tiles, output_shape)

    def _merge_tiles_labels(
        self, tiles: List[TiledPrediction], output_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Merge class-index tiles with overlap weighting."""
        max_class = 0
        for tile_data, _ in tiles:
            max_class = max(max_class, np.max(tile_data))
        num_classes = int(max_class) + 1

        logger.debug("Merging multi-class tiles with %s classes", num_classes)
        class_confidences = np.zeros((num_classes,) + output_shape[:2], dtype=np.float32)

        for tile_data, (row_slice, col_slice, is_edge_row, is_edge_col) in tiles:
            weight_mask = np.ones(
                (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)
            )

            border_size = int(self.trim_size * 0.7)
            center_weight = 3.0

            if not is_edge_row:
                weight_mask[:border_size, :] = np.linspace(0.1, 1, border_size)[:, np.newaxis]
                weight_mask[-border_size:, :] = np.linspace(1, 0.1, border_size)[:, np.newaxis]
                center_start = border_size
                center_end = weight_mask.shape[0] - border_size
                if center_end > center_start:
                    weight_mask[center_start:center_end, :] *= center_weight

            if not is_edge_col:
                weight_mask[:, :border_size] *= np.linspace(0.1, 1, border_size)[np.newaxis, :]
                weight_mask[:, -border_size:] *= np.linspace(1, 0.1, border_size)[np.newaxis, :]
                center_start = border_size
                center_end = weight_mask.shape[1] - border_size
                if center_end > center_start:
                    weight_mask[:, center_start:center_end] *= center_weight

            for class_idx in range(num_classes):
                class_mask = (tile_data == class_idx).astype(np.float32)
                class_confidences[class_idx, row_slice, col_slice] += class_mask * weight_mask

        return np.argmax(class_confidences, axis=0).astype(np.uint8)

    def _merge_tiles_probabilities(
        self, tiles: List[TiledPrediction], output_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Merge continuous-valued tiles with linear overlap blending."""
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape[:2], dtype=np.float32)

        for tile_data, (row_slice, col_slice, is_edge_row, is_edge_col) in tiles:
            weight_mask = np.ones(
                (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)
            )

            if not is_edge_row:
                weight_mask[: self.trim_size, :] = np.linspace(0, 1, self.trim_size)[:, np.newaxis]
                weight_mask[-self.trim_size :, :] = np.linspace(1, 0, self.trim_size)[:, np.newaxis]

            if not is_edge_col:
                weight_mask[:, : self.trim_size] *= np.linspace(0, 1, self.trim_size)[np.newaxis, :]
                weight_mask[:, -self.trim_size :] *= np.linspace(1, 0, self.trim_size)[
                    np.newaxis, :
                ]

            output[row_slice, col_slice] += (
                tile_data * weight_mask[..., np.newaxis]
                if tile_data.ndim == 3
                else tile_data * weight_mask
            )
            weights[row_slice, col_slice] += weight_mask

        valid_mask = weights > 0
        output[valid_mask] /= (
            weights[valid_mask][..., np.newaxis] if output.ndim == 3 else weights[valid_mask]
        )

        return output

    def iter_tiles(
        self, image: np.ndarray, input_kind: TiffInputKind | str = TiffInputKind.IMAGE
    ) -> Iterator[TiledPrediction]:
        """Yield padded tiles and their output locations."""

        resolved_input_kind = TiffInputKind.parse(input_kind)
        if resolved_input_kind is not TiffInputKind.IMAGE:
            raise PredictionInputError(
                "iter_tiles only supports single-image TIFF inputs. Split stacks before tiling."
            )
        self._validate_image_array(image)

        tile_locations = self.get_tile_locations(image.shape)

        for row_slice, col_slice, is_edge_row, is_edge_col in tile_locations:
            tile = image[row_slice, col_slice]

            if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                padded_tile = np.zeros(
                    (self.tile_size, self.tile_size) + tile.shape[2:], dtype=tile.dtype
                )
                padded_tile[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded_tile

            yield tile, (row_slice, col_slice, is_edge_row, is_edge_col)
