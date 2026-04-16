"""Data handling utilities for tomographic predictions."""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from .errors import PredictionDataAccessError
from .types import NormalizationRange, PathLike

logger = logging.getLogger(__name__)
DEFAULT_DATASET_PATH = "/data"


class TomoDataset:
    """Load, save, and normalize tomographic prediction data."""

    def __init__(self, normalize_range: Optional[NormalizationRange] = None):
        """Create a data handler with an optional fixed normalization range."""
        self.normalize_range = normalize_range

    def load_h5(self, path: PathLike, dataset_path: str = DEFAULT_DATASET_PATH) -> np.ndarray:
        """Load a volume from an H5 dataset."""
        logger.debug("Loading h5 file %s from dataset %s", path, dataset_path)
        with h5py.File(path, "r") as f:
            if dataset_path not in f:
                raise PredictionDataAccessError(f"Dataset {dataset_path} not found in {path}")
            data = f[dataset_path][:]

        if data.dtype != np.uint16:
            logger.warning("Expected uint16 data, got %s. Converting.", data.dtype)
            data = data.astype(np.uint16)

        logger.debug(
            "Loaded volume with shape %s, dtype %s, range [%s, %s]",
            data.shape,
            data.dtype,
            data.min(),
            data.max(),
        )
        return data

    def save_h5(
        self,
        data: np.ndarray,
        path: PathLike,
        dataset_path: str = DEFAULT_DATASET_PATH,
    ) -> None:
        """Save prediction output to an H5 dataset."""
        logger.debug("Saving h5 file %s to dataset %s", path, dataset_path)
        output, dtype = self._prepare_h5_output(data)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset(dataset_path, data=output, dtype=dtype)

        logger.debug(
            "Saved volume with shape %s, dtype %s, range [%s, %s]",
            output.shape,
            output.dtype,
            output.min(),
            output.max(),
        )

    def _prepare_h5_output(self, data: np.ndarray) -> tuple[np.ndarray, np.dtype[np.generic]]:
        """Normalize output arrays for H5 persistence."""
        if data.ndim == 4 and data.shape[1] > 1:
            if data.shape[0] == 1:
                output = np.argmax(data[0], axis=0)
            else:
                output = np.argmax(data, axis=1)
            return output.astype(np.uint8), np.dtype(np.uint8)

        if data.dtype != np.uint16:
            logger.warning("Converting %s predictions to uint16.", data.dtype)
            if np.issubdtype(data.dtype, np.floating):
                data = (data * 65535).astype(np.uint16)
            else:
                data = data.astype(np.uint16)

        return data, np.dtype(np.uint16)

    def normalize_slice(self, slice_data: np.ndarray) -> torch.Tensor:
        """Convert a slice into a normalized BCHW tensor."""
        slice_data = slice_data.astype(np.float32)

        if self.normalize_range is None:
            min_value = float(slice_data.min())
            max_value = float(slice_data.max())
        else:
            min_value, max_value = self.normalize_range

        scale = max(max_value - min_value, 1e-6)
        slice_data = np.clip((slice_data - min_value) / scale, 0.0, 1.0)

        slice_tensor = (
            torch.tensor(slice_data.tolist(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        return slice_tensor

    def denormalize_prediction(self, pred: torch.Tensor) -> np.ndarray:
        """Convert a prediction tensor into its persisted array form."""
        if pred.size(1) > 1:
            pred_np = torch.argmax(pred, dim=1).cpu().numpy()
            return pred_np.astype(np.uint8)

        pred_np = pred.squeeze().cpu().numpy()
        return (pred_np > 0.5).astype(np.uint16) * 65535

    def get_slice_stats(self, slice_data: np.ndarray) -> dict[str, float]:
        """Return basic statistics for a slice."""
        return {
            "min": float(slice_data.min()),
            "max": float(slice_data.max()),
            "mean": float(slice_data.mean()),
        }
