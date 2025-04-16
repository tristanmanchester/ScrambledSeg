"""Multi-axis prediction for tomographic segmentation and 2D image segmentation."""
import numpy as np
import torch
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple
import logging
from tqdm import tqdm

from .data import TomoDataset
from .axis import AxisPredictor, Axis
from .tiff_utils import TiffHandler

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

class Predictor:
    """Handles both 3D volume and 2D image prediction."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        prediction_mode: Union[PredictionMode, str] = PredictionMode.SINGLE_AXIS,
        ensemble_method: Union[EnsembleMethod, str] = EnsembleMethod.MEAN,
        batch_size: int = 8,
        device: Optional[str] = None,
        normalize_range: Optional[tuple] = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        precision: str = 'bf16'
    ):
        """Initialize predictor.
        
        Args:
            model: PyTorch model for prediction
            prediction_mode: Single axis, three axis, or twelve axis prediction
            ensemble_method: Method for combining multiple predictions
            batch_size: Number of slices/tiles to process at once
            device: Device to run prediction on
            normalize_range: Optional range for input normalization
            tile_size: Size of tiles for processing large images
            tile_overlap: Overlap between tiles
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
        
        # Set precision for inference
        self.precision = precision
        if precision == 'bf16' and torch.cuda.is_bf16_supported():
            logger.info("Using BF16 precision for inference")
        elif precision == '16' and torch.cuda.is_available():
            logger.info("Using FP16 precision for inference")
        else:
            logger.info("Using FP32 precision for inference")
            self.precision = '32'
        
        # Initialize handlers
        self.data_handler = TomoDataset(normalize_range=normalize_range)
        self.axis_handler = AxisPredictor()
        # Use larger default overlap for better edge handling
        self.tiff_handler = TiffHandler(tile_size=tile_size, overlap=tile_overlap or 128)
        
        # Define rotations for 12-axis prediction
        self.rotations = [0, 90, 180, 270]

    def predict(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        dataset_path: str = "/data"
    ) -> None:
        """Predict segmentation for input file.
        
        Automatically detects file type (H5 or TIFF) and processes accordingly.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            dataset_path: Path to dataset within H5 files (only used for H5)
        """
        input_path = Path(input_path)
        if input_path.suffix.lower() in ['.h5']:
            self.predict_volume(input_path, output_path, dataset_path)
        elif input_path.suffix.lower() in ['.tif', '.tiff']:
            self.predict_image(input_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")

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

    def predict_image(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> None:
        """Predict segmentation for image using tiling.
        
        Args:
            input_path: Path to input TIFF file
            output_path: Path to output TIFF file
        """
        # Load image
        image = self.tiff_handler.load_tiff(input_path)
        
        # Check if image is 3D (multipage TIFF)
        if len(image.shape) == 3 and image.shape[0] > 1:
            logger.info(f"Detected 3D volume with {image.shape[0]} slices, processing slice by slice")
            # Process each slice separately and stack results
            all_results = []
            
            for z in tqdm(range(image.shape[0])):
                # Extract 2D slice
                slice_2d = image[z]
                
                # Process this slice
                slice_result = self._process_2d_image(slice_2d)
                all_results.append(slice_result)
            
            # Stack results along Z axis
            output = np.stack(all_results, axis=0)
            
            # Save result
            self.tiff_handler.save_tiff(output, output_path)
            logger.info("Prediction complete!")
        else:
            # Process as 2D image
            output = self._process_2d_image(image)
            
            # Save result
            self.tiff_handler.save_tiff(output, output_path)
            logger.info("Prediction complete!")
    
    def _process_2d_image(self, image: np.ndarray) -> np.ndarray:
        """Process a 2D image using tiling.
        
        Args:
            image: 2D input image of shape (H, W) or (H, W, C)
            
        Returns:
            Processed image
        """
        # Process tiles
        tiles = []
        batch_tiles = []
        batch_locations = []
        
        logger.info("Processing tiles...")
        for tile_data, location in tqdm(self.tiff_handler.iter_tiles(image)):
            # Normalize tile
            tile_tensor = self._normalize_tile(tile_data)
            
            batch_tiles.append(tile_tensor)
            batch_locations.append(location)
            
            # Process batch if full
            if len(batch_tiles) >= self.batch_size:
                pred_tiles = self._predict_batch(batch_tiles)
                tiles.extend(zip(pred_tiles, batch_locations))
                batch_tiles = []
                batch_locations = []
        
        # Process remaining tiles
        if batch_tiles:
            pred_tiles = self._predict_batch(batch_tiles)
            tiles.extend(zip(pred_tiles, batch_locations))
        
        # Merge tiles
        logger.info("Merging tiles...")
        output = self.tiff_handler.merge_tiles(tiles, image.shape[:2])
        
        return output

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
            
            # Stack batch, convert to appropriate precision and predict
            dtype = torch.float32
            if self.precision == 'bf16' and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.precision == '16' and torch.cuda.is_available():
                dtype = torch.float16
                
            batch_tensor = torch.cat(batch_slices, dim=0).to(device=self.device, dtype=dtype)
            
            # Get predictions
            pred_batch = self.model(batch_tensor)
            
            # For multi-class, use softmax instead of sigmoid
            if pred_batch.size(1) > 1:  # Multi-class case
                # Apply softmax to get class probabilities
                pred_batch = torch.softmax(pred_batch, dim=1)
                
                # Process each slice in batch
                for batch_idx, idx in enumerate(range(start_idx, end_idx)):
                    pred_slice = pred_batch[batch_idx]
                    
                    # Rotate back if needed
                    pred_np = pred_slice.cpu().numpy()
                    if rotation_angle != 0:
                        # For multi-class, rotate each channel separately
                        for c in range(pred_np.shape[0]):
                            pred_np[c] = self.axis_handler.rotate_slice(pred_np[c:c+1], -rotation_angle)[0]
                    
                    # Add to output - for each class
                    self.axis_handler.set_slice(output, axis, idx, pred_np, out=output)
                    self.axis_handler.set_slice(counts, axis, idx, np.ones_like(pred_np), out=counts)
            else:
                # Binary case - use sigmoid
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

    def _normalize_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Normalize tile data to tensor.
        
        Args:
            tile: Input tile array
            
        Returns:
            Normalized tensor of shape (1, C, H, W)
        """
        # Convert to float32
        tile = tile.astype(np.float32)
        
        # Normalize to [0,1]
        if tile.max() > 1:
            tile = tile / 255.0
            
        # Add batch and channel dimensions
        if len(tile.shape) == 2:  # Grayscale image: H x W
            tile = tile[None, None, :, :]  # B x C x H x W (where C=1)
        elif len(tile.shape) == 3:
            # Check if this is a color image (H,W,3) or a multichannel volume slice
            if tile.shape[2] <= 4:  # Color image with channels as last dimension
                tile = tile.transpose(2, 0, 1)[None, :, :, :]  # B x C x H x W
            else:
                # This should never happen with our fixed code, but just in case
                raise ValueError(f"Unexpected tile shape: {tile.shape}. Expected 2D image.")
            
        return torch.from_numpy(tile)

    @torch.no_grad()
    def _predict_batch(self, batch_tiles: list) -> list:
        """Predict batch of tiles.
        
        Args:
            batch_tiles: List of normalized tile tensors
            
        Returns:
            List of prediction arrays
        """
        # Apply precision
        dtype = torch.float32
        if self.precision == 'bf16' and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif self.precision == '16' and torch.cuda.is_available():
            dtype = torch.float16
            
        # Stack batch and convert to appropriate precision
        batch_tensor = torch.cat(batch_tiles, dim=0).to(device=self.device, dtype=dtype)
        
        # Get predictions
        pred_batch = self.model(batch_tensor)
        
        # For multi-class, use softmax instead of sigmoid
        if pred_batch.size(1) > 1:  # Multi-class case
            # Apply softmax to get class probabilities
            pred_batch = torch.softmax(pred_batch, dim=1)
            
            # Get class indices using argmax
            pred_class = torch.argmax(pred_batch, dim=1)
            
            # Convert to numpy
            pred_tiles = []
            for pred in pred_class:
                pred_np = pred.cpu().numpy()
                pred_tiles.append(pred_np)
        else:
            # Binary case - use sigmoid
            pred_batch = torch.sigmoid(pred_batch)
            
            # Convert to numpy
            pred_tiles = []
            for pred in pred_batch:
                pred_np = pred.squeeze().cpu().numpy()
                pred_tiles.append(pred_np)
            
        return pred_tiles

    def _get_prediction_axes(self) -> List[Axis]:
        """Get list of axes to process based on prediction mode."""
        if self.prediction_mode == PredictionMode.SINGLE_AXIS:
            return [Axis.XY]
        else:
            return list(Axis)