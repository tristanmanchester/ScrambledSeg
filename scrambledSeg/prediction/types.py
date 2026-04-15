"""Shared type aliases for prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np

PathLike: TypeAlias = str | Path
NormalizationRange: TypeAlias = tuple[float, float]
TileLocation: TypeAlias = tuple[slice, slice, bool, bool]
TiledPrediction: TypeAlias = tuple[np.ndarray, TileLocation]
PredictionAccumulator: TypeAlias = tuple[np.ndarray, np.ndarray]
