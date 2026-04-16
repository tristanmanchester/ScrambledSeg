"""Multi-axis prediction for tomographic segmentation and 2D image segmentation."""

import logging
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..axis import Axis, AxisPredictor
from .data import DEFAULT_DATASET_PATH, TomoDataset
from .errors import PredictionInputError
from .tiff_utils import (
    DEFAULT_TILE_OVERLAP,
    TiffHandler,
    TiffInputKind,
    TiffOutputKind,
)
from .types import (
    NormalizationRange,
    PathLike,
    PredictionAccumulator,
    TiledPrediction,
    TileLocation,
)

logger = logging.getLogger(__name__)


class PredictionMode(Enum):
    """Available prediction modes."""

    SINGLE_AXIS = "SINGLE_AXIS"
    THREE_AXIS = "THREE_AXIS"
    TWELVE_AXIS = "TWELVE_AXIS"

    @classmethod
    def parse(cls, value: "PredictionMode | str") -> "PredictionMode":
        """Normalize legacy and current prediction mode spellings."""
        if isinstance(value, cls):
            return value

        normalized = value.strip().upper()
        aliases = {
            "SINGLE": cls.SINGLE_AXIS,
            "SINGLE_AXIS": cls.SINGLE_AXIS,
            "THREE": cls.THREE_AXIS,
            "THREE_AXIS": cls.THREE_AXIS,
            "TWELVE": cls.TWELVE_AXIS,
            "TWELVE_AXIS": cls.TWELVE_AXIS,
        }

        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = [mode.value for mode in cls]
            raise PredictionInputError(
                f"Invalid prediction mode: {value}. Must be one of {choices}."
            ) from exc


class Predictor:
    """Handles both 3D volume and 2D image prediction."""

    def __init__(
        self,
        model: torch.nn.Module,
        prediction_mode: Union[PredictionMode, str] = PredictionMode.SINGLE_AXIS,
        batch_size: int = 8,
        device: Optional[str] = None,
        normalize_range: Optional[NormalizationRange] = None,
        tile_size: int = 512,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        precision: str = "bf16",
    ):
        """Initialize a predictor for H5 volumes and TIFF images."""
        self.model = model

        self.prediction_mode = PredictionMode.parse(prediction_mode)

        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.precision = precision
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            logger.debug("Using BF16 precision for inference")
        elif precision == "16" and torch.cuda.is_available():
            logger.debug("Using FP16 precision for inference")
        else:
            logger.debug("Using FP32 precision for inference")
            self.precision = "32"

        self.data_handler = TomoDataset(normalize_range=normalize_range)
        self.axis_handler = AxisPredictor()
        self.tiff_handler = TiffHandler(tile_size=tile_size, overlap=tile_overlap)
        self.rotations = [0, 90, 180, 270]

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a tensor to a NumPy array without relying on PyTorch's NumPy bridge."""
        detached = tensor.detach().cpu()
        if detached.is_floating_point():
            return np.asarray(detached.tolist(), dtype=np.float32)
        if detached.dtype == torch.bool:
            return np.asarray(detached.tolist(), dtype=np.bool_)
        return np.asarray(detached.tolist(), dtype=np.int64)

    def _get_inference_dtype(self) -> torch.dtype:
        """Return the configured inference dtype for model execution."""
        if self.precision == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.precision == "16" and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def predict_volume(
        self,
        input_path: PathLike,
        output_path: PathLike,
        dataset_path: str = DEFAULT_DATASET_PATH,
    ) -> None:
        """Predict segmentation for an H5 volume."""
        volume = self.data_handler.load_h5(input_path, dataset_path)

        output = None
        counts = None
        axes = self._get_prediction_axes()

        for axis in axes:
            logger.debug("Processing axis %s", axis.name)
            rotations = (
                self.rotations if self.prediction_mode == PredictionMode.TWELVE_AXIS else [0]
            )

            for angle in rotations:
                if angle != 0:
                    logger.debug("Rotating by %s degrees", angle)
                axis_pred, axis_counts = self._predict_axis(volume, axis, angle)

                if output is None:
                    output = np.zeros_like(axis_pred, dtype=np.float32)
                    counts = np.zeros_like(axis_counts, dtype=np.float32)
                output += axis_pred
                counts += axis_counts

        output = np.divide(output, counts, out=np.zeros_like(output), where=counts > 0)
        self.data_handler.save_h5(output, output_path, dataset_path)
        logger.info("Prediction complete!")

    def predict_tiff(
        self,
        input_path: PathLike,
        output_path: PathLike,
        input_kind: TiffInputKind | str = TiffInputKind.AUTO,
        output_kind: TiffOutputKind | str = TiffOutputKind.LABELS,
    ) -> None:
        """Predict segmentation for a TIFF image or stack."""

        image = self.tiff_handler.load_tiff(input_path)
        resolved_input_kind = self.tiff_handler.resolve_input_kind(image, input_kind)
        resolved_output_kind = TiffOutputKind.parse(output_kind)

        if resolved_input_kind is TiffInputKind.STACK:
            output = self._predict_tiff_stack_array(image, resolved_output_kind)
        else:
            output = self._predict_tiff_image_array(image, resolved_output_kind)

        self.tiff_handler.save_tiff(output, output_path, output_kind=resolved_output_kind)
        logger.info("Prediction complete!")

    def _predict_tiff_image_array(
        self, image: np.ndarray, output_kind: TiffOutputKind
    ) -> np.ndarray:
        """Predict a single TIFF image that may be grayscale or channel-last."""

        return self._process_2d_image(image, output_kind)

    def _predict_tiff_stack_array(
        self, image: np.ndarray, output_kind: TiffOutputKind
    ) -> np.ndarray:
        """Predict each slice of a TIFF stack independently and re-stack the output."""

        logger.debug("Processing %s TIFF slices independently", image.shape[0])
        all_results = []
        for z in tqdm(range(image.shape[0])):
            all_results.append(self._process_2d_image(image[z], output_kind))
        return np.stack(all_results, axis=0)

    def _process_2d_image(
        self, image: np.ndarray, output_kind: TiffOutputKind = TiffOutputKind.LABELS
    ) -> np.ndarray:
        """Run tiled prediction on a single 2D image."""
        tiles: List[TiledPrediction] = []
        batch_tiles: List[torch.Tensor] = []
        batch_locations: List[TileLocation] = []

        logger.debug("Processing image tiles")
        for tile_data, location in tqdm(self.tiff_handler.iter_tiles(image, TiffInputKind.IMAGE)):
            tile_tensor = self._normalize_tile(tile_data)
            batch_tiles.append(tile_tensor)
            batch_locations.append(location)

            if len(batch_tiles) >= self.batch_size:
                pred_tiles = self._predict_batch(batch_tiles, output_kind=output_kind)
                tiles.extend(zip(pred_tiles, batch_locations))
                batch_tiles = []
                batch_locations = []

        if batch_tiles:
            pred_tiles = self._predict_batch(batch_tiles, output_kind=output_kind)
            tiles.extend(zip(pred_tiles, batch_locations))

        logger.debug("Merging image tiles")
        merge_shape: tuple[int, ...] = image.shape[:2]
        if (
            output_kind is TiffOutputKind.PROBABILITIES
            and tiles
            and tiles[0][0].ndim == 3
        ):
            merge_shape = image.shape[:2] + (tiles[0][0].shape[2],)

        return self.tiff_handler.merge_tiles(
            tiles,
            merge_shape,
            output_kind=output_kind,
        )

    @torch.no_grad()
    def _predict_axis(
        self, volume: np.ndarray, axis: Axis, rotation_angle: float = 0
    ) -> PredictionAccumulator:
        """Predict all slices for a single axis and rotation."""
        shape = self.axis_handler.get_volume_shape(axis, volume.shape)
        depth = shape[0]

        output = None
        counts = None

        for start_idx in tqdm(range(0, depth, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, depth)
            batch_slices = []

            for idx in range(start_idx, end_idx):
                slice_data = self.axis_handler.get_slice(volume, axis, idx)
                if rotation_angle != 0:
                    slice_data = self.axis_handler.rotate_slice(slice_data, rotation_angle)

                batch_slices.append(self.data_handler.normalize_slice(slice_data))

            batch_tensor = torch.cat(batch_slices, dim=0).to(
                device=self.device,
                dtype=self._get_inference_dtype(),
            )
            pred_batch = self.model(batch_tensor)

            if pred_batch.size(1) > 1:
                pred_batch = torch.softmax(pred_batch, dim=1)

                if output is None:
                    output = np.zeros(
                        (volume.shape[0], pred_batch.size(1), volume.shape[1], volume.shape[2]),
                        dtype=np.float32,
                    )
                    counts = np.zeros_like(output, dtype=np.float32)

                for batch_idx, idx in enumerate(range(start_idx, end_idx)):
                    pred_slice = pred_batch[batch_idx]
                    pred_np = self._tensor_to_numpy(pred_slice)
                    if rotation_angle != 0:
                        for c in range(pred_np.shape[0]):
                            pred_np[c] = self.axis_handler.rotate_slice(pred_np[c], -rotation_angle)

                    self.axis_handler.set_slice(output, axis, idx, pred_np, out=output)
                    self.axis_handler.set_slice(
                        counts, axis, idx, np.ones_like(pred_np), out=counts
                    )
            else:
                pred_batch = torch.sigmoid(pred_batch)

                if output is None:
                    output = np.zeros_like(volume, dtype=np.float32)
                    counts = np.zeros_like(volume, dtype=np.float32)

                for batch_idx, idx in enumerate(range(start_idx, end_idx)):
                    pred_slice = pred_batch[batch_idx]
                    pred_np = self._tensor_to_numpy(pred_slice)
                    if rotation_angle != 0:
                        pred_np = self.axis_handler.rotate_slice(pred_np, -rotation_angle)

                    self.axis_handler.set_slice(output, axis, idx, pred_np, out=output)
                    self.axis_handler.set_slice(
                        counts, axis, idx, np.ones_like(pred_np), out=counts
                    )

        return output, counts

    def _normalize_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Convert a TIFF tile into a normalized BCHW tensor."""
        tile = tile.astype(np.float32)

        if tile.max() > 1:
            tile = tile / 255.0

        if len(tile.shape) == 2:
            tile = tile[None, None, :, :]
        elif len(tile.shape) == 3:
            if tile.shape[2] <= 4:
                tile = tile.transpose(2, 0, 1)[None, :, :, :]
            else:
                raise PredictionInputError(
                    f"Unexpected tile shape: {tile.shape}. Expected a 2D image tile or "
                    "a channel-last tile with at most four channels."
                )

        return torch.from_numpy(tile)

    @torch.no_grad()
    def _predict_batch(
        self,
        batch_tiles: List[torch.Tensor],
        output_kind: TiffOutputKind | str = TiffOutputKind.LABELS,
    ) -> List[np.ndarray]:
        """Predict a batch of normalized TIFF tiles."""
        batch_tensor = torch.cat(batch_tiles, dim=0).to(
            device=self.device,
            dtype=self._get_inference_dtype(),
        )
        pred_batch = self.model(batch_tensor)
        resolved_output_kind = TiffOutputKind.parse(output_kind)

        if pred_batch.size(1) > 1:
            pred_batch = torch.softmax(pred_batch, dim=1)
            if resolved_output_kind is TiffOutputKind.PROBABILITIES:
                return [
                    np.transpose(self._tensor_to_numpy(pred), (1, 2, 0))
                    for pred in pred_batch
                ]

            pred_class = torch.argmax(pred_batch, dim=1)
            return [self._tensor_to_numpy(pred) for pred in pred_class]

        pred_batch = torch.sigmoid(pred_batch)
        if resolved_output_kind is TiffOutputKind.PROBABILITIES:
            return [self._tensor_to_numpy(pred.squeeze()) for pred in pred_batch]

        pred_class = (pred_batch > 0.5).to(torch.uint8)
        return [self._tensor_to_numpy(pred.squeeze()) for pred in pred_class]

    def _get_prediction_axes(self) -> List[Axis]:
        """Get list of axes to process based on prediction mode."""
        if self.prediction_mode == PredictionMode.SINGLE_AXIS:
            return [Axis.XY]
        else:
            return list(Axis)
