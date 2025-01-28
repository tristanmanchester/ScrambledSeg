"""Multi-axis prediction for tomographic segmentation."""
import numpy as np
import torch
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
import logging
from tqdm import tqdm

from .data import TomoDataset
from .axis import AxisPredictor, Axis

logger = logging.getLogger(__name__)

class PredictionMode(Enum):
    """Available prediction modes."""
    SINGLE_AXIS = "single"  # Just XY predictions
    THREE_AXIS = "three"    # XY, YZ, XZ
    TWELVE_AXIS = "twelve"  # 3 axes × 4 rotations (0°, 90°, 180°, 270°)

class EnsembleMethod(Enum):
    """Methods for combining multiple predictions."""
    MEAN = "mean"
    VOTING = "voting"

class TomoPredictor:
    """Handles multi-axis prediction for tomographic data."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        prediction_mode: Union[PredictionMode, str] = PredictionMode.SINGLE_AXIS,
        ensemble_method: Union[EnsembleMethod, str] = EnsembleMethod.MEAN,
        batch_size: int = 8,
        device: Optional[str] = None,
        normalize_range: Optional[tuple] = None
    ):
        """Initialize predictor.
        
        Args:
            model: PyTorch model for prediction
            prediction_mode: Single axis, three axis, or twelve axis prediction
            ensemble_method: Method for combining multiple predictions
            batch_size: Number of slices to process at once
            device: Device to run prediction on
            normalize_range: Optional range for input normalization
            
        Raises:
            ValueError: If prediction_mode or ensemble_method is invalid
        """
        self.model = model
        
        # Validate and convert prediction mode
        if isinstance(prediction_mode, str):
            try:
                prediction_mode = PredictionMode(prediction_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid prediction mode: {prediction_mode}. "
                    f"Must be one of {[mode.value for mode in PredictionMode]}"
                )
        elif not isinstance(prediction_mode, PredictionMode):
            raise ValueError("prediction_mode must be a string or PredictionMode enum")
        self.prediction_mode = prediction_mode
        
        # Validate and convert ensemble method
        if isinstance(ensemble_method, str):
            try:
                ensemble_method = EnsembleMethod(ensemble_method)
            except ValueError:
                raise ValueError(
                    f"Invalid ensemble method: {ensemble_method}. "
                    f"Must be one of {[method.value for method in EnsembleMethod]}"
                )
        elif not isinstance(ensemble_method, EnsembleMethod):
            raise ValueError("ensemble_method must be a string or EnsembleMethod enum")
        self.ensemble_method = ensemble_method
        
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize handlers
        self.data_handler = TomoDataset(normalize_range=normalize_range)
        self.axis_handler = AxisPredictor()
        
        # Define rotations for 12-axis prediction
        self.rotations = [0, 90, 180, 270]
    
    def predict_volume(
        self, 
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        dataset_path: str = "/data"
    ) -> None:
        """Predict segmentation for entire volume.
        
        Args:
            input_path: Path to input h5 file
            output_path: Path to output h5 file
            dataset_path: Path to dataset within h5 files
        """
        # Load volume
        volume = self.data_handler.load_h5(input_path, dataset_path)
        
        # Create output volume
        output = np.zeros_like(volume, dtype=np.float32)
        counts = np.zeros_like(volume, dtype=np.float32)
        
        # Get axes to process
        axes = self._get_prediction_axes()
        
        # Process each axis
        for axis in axes:
            logger.info(f"Processing axis: {axis.name}")
            
            # Get rotations for this axis
            rotations = self.rotations if self.prediction_mode == PredictionMode.TWELVE_AXIS else [0]
            
            for angle in rotations:
                if angle != 0:
                    logger.info(f"Rotating by {angle} degrees")
                
                # Process all slices for this axis/rotation
                axis_pred, axis_counts = self._predict_axis(volume, axis, angle)
                
                # Add to ensemble
                output += axis_pred
                counts += axis_counts
        
        # Average predictions
        output = np.divide(output, counts, where=counts > 0)
        
        # Convert to uint16 and save
        self.data_handler.save_h5(output, output_path, dataset_path)
        
        logger.info("Prediction complete!")
    
    @torch.no_grad()
    def _predict_axis(
        self, 
        volume: np.ndarray,
        axis: Axis,
        rotation_angle: float = 0
    ) -> tuple:
        """Predict all slices along an axis.
        
        Args:
            volume: Input volume
            axis: Axis to predict along
            rotation_angle: Optional rotation to apply
            
        Returns:
            Tuple of (predictions, count_mask)
        """
        # Get shape for this axis view
        shape = self.axis_handler.get_volume_shape(axis, volume.shape)
        depth = shape[0]
        
        # Create output arrays
        output = np.zeros_like(volume, dtype=np.float32)
        counts = np.zeros_like(volume, dtype=np.float32)
        
        # Process in batches
        for start_idx in tqdm(range(0, depth, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, depth)
            batch_slices = []
            
            # Get batch of slices
            for idx in range(start_idx, end_idx):
                # Extract and rotate slice if needed
                slice_data = self.axis_handler.get_slice(volume, axis, idx)
                if rotation_angle != 0:
                    slice_data = self.axis_handler.rotate_slice(slice_data, rotation_angle)
                
                # Normalize
                slice_tensor = self.data_handler.normalize_slice(slice_data)
                batch_slices.append(slice_tensor)
            
            # Stack batch and predict
            batch_tensor = torch.cat(batch_slices, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            
            # Get predictions
            pred_batch = self.model(batch_tensor)
            pred_batch = torch.sigmoid(pred_batch)
            
            # Process each slice in batch
            for batch_idx, idx in enumerate(range(start_idx, end_idx)):
                pred_slice = pred_batch[batch_idx]
                
                # Rotate back if needed
                pred_np = pred_slice.cpu().numpy()
                if rotation_angle != 0:
                    pred_np = self.axis_handler.rotate_slice(pred_np, -rotation_angle)
                
                # Add to output
                self.axis_handler.set_slice(output, axis, idx, pred_np, out=output)
                self.axis_handler.set_slice(counts, axis, idx, np.ones_like(pred_np), out=counts)
        
        return output, counts
    
    def _get_prediction_axes(self) -> List[Axis]:
        """Get list of axes to process based on prediction mode."""
        if self.prediction_mode == PredictionMode.SINGLE_AXIS:
            return [Axis.XY]
        else:
            return list(Axis)
