"""Shared exception types for prediction and inference workflows."""


class PredictionError(RuntimeError):
    """Base class for prediction-facing failures."""


class PredictionInputError(PredictionError, ValueError):
    """Invalid caller input or unsupported prediction input shape."""


class PredictionDataAccessError(PredictionError):
    """Prediction input could not be loaded from disk or storage."""


class ModelLoadError(PredictionError):
    """A checkpoint could not be translated into a runnable model."""
