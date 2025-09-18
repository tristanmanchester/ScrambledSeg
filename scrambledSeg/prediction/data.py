"""Data handling utilities for tomographic predictions."""
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TomoDataset:
    """Handles loading, saving, and preprocessing of tomographic data."""
    
    def __init__(self, normalize_range: Optional[Tuple[float, float]] = None):
        """Initialize dataset handler.
        
        Args:
            normalize_range: Optional tuple of (min, max) values to use for normalization.
                           If None, will use the min/max of each slice.
        """
        self.normalize_range = normalize_range
        
    def load_h5(self, path: Union[str, Path], dataset_path: str = "/data") -> np.ndarray:
        """Load uint16 data from h5 file.
        
        Args:
            path: Path to h5 file
            dataset_path: Path to dataset within h5 file
            
        Returns:
            numpy array of shape (D, H, W) with uint16 dtype
        """
        logger.info(f"Loading h5 file: {path}")
        with h5py.File(path, "r") as f:
            if dataset_path not in f:
                raise ValueError(f"Dataset {dataset_path} not found in {path}")
            data = f[dataset_path][:]
        
        if data.dtype != np.uint16:
            logger.warning(f"Expected uint16 data, got {data.dtype}. Converting...")
            data = data.astype(np.uint16)
        
        logger.info(f"Loaded volume of shape {data.shape} and dtype {data.dtype}")
        logger.info(f"Value range: [{data.min()}, {data.max()}]")
        return data
    
    def save_h5(self, data: np.ndarray, path: Union[str, Path], dataset_path: str = "/data"):
        """Save predictions to h5 file, handling both binary and multi-class data.
        
        Args:
            data: numpy array to save
            path: Path to h5 file
            dataset_path: Path to dataset within h5 file
        """
        logger.info(f"Saving predictions to {path}")
        
        # Check if this is multi-class data (multiple channels in axis 1)
        is_multi_channel = data.ndim > 3 and data.shape[1] > 1
        
        if is_multi_channel:
            # Multi-class data - get class indices using argmax
            logger.info(f"Detected multi-class data with {data.shape[1]} classes")
            # If shape is (B, C, H, W), first check if B is 1, then apply argmax along C
            if data.ndim == 4 and data.shape[0] == 1:
                # Remove batch dimension first
                data = data[0]
                # Apply argmax along channel dimension (axis 0)
                data = np.argmax(data, axis=0).astype(np.uint8)
            elif data.ndim == 4:
                # Apply argmax along channel dimension (axis 1)
                data = np.argmax(data, axis=1).astype(np.uint8)
            
            logger.info(f"Converted to class indices with shape {data.shape}")
            # Create parent directory if needed
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(path, "w") as f:
                f.create_dataset(dataset_path, data=data, dtype=np.uint8)
        else:
            parent_dir = Path(path).parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            if (
                np.issubdtype(data.dtype, np.integer)
                and data.ndim == 3
                and data.max() <= 255
            ):
                logger.info("Detected integer class map; saving as uint8")
                data = data.astype(np.uint8, copy=False)
                with h5py.File(path, "w") as f:
                    f.create_dataset(dataset_path, data=data, dtype=np.uint8)
            else:
                if data.dtype != np.uint16:
                    logger.warning(f"Converting {data.dtype} to uint16...")
                    if data.dtype in (np.float32, np.float64):
                        # Assume normalized [0, 1] data
                        data = (data * 65535).astype(np.uint16)
                    else:
                        data = data.astype(np.uint16)

                with h5py.File(path, "w") as f:
                    f.create_dataset(dataset_path, data=data, dtype=np.uint16)
        
        logger.info(f"Saved volume of shape {data.shape}")
        logger.info(f"Value range: [{data.min()}, {data.max()}]")
    
    def normalize_slice(self, slice_data: np.ndarray) -> torch.Tensor:
        """Convert slice to normalized tensor.
        
        Args:
            slice_data: Numpy array of shape (H, W)
            
        Returns:
            Normalized tensor of shape (1, 1, H, W)
        """
        # Convert to float32
        slice_data = slice_data.astype(np.float32)
        
        # If uint16 input or large values, normalize to [0,1] first
        if slice_data.max() > 255:
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        # Divide by 255 as in training (this is required for the model to work correctly)
        slice_data = slice_data / 255.0
        
        # Add batch and channel dimensions
        slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0)
        return slice_tensor
    
    def denormalize_prediction(self, pred: torch.Tensor) -> np.ndarray:
        """Convert prediction tensor to appropriate output format.
        
        Args:
            pred: Prediction tensor from model, shape (1, C, H, W)
            
        Returns:
            For binary segmentation: Binary prediction as uint16 array
            For multi-class segmentation: Class indices as uint8 array
        """
        # Check if multi-class segmentation (more than 1 channel)
        if pred.size(1) > 1:
            # Multi-class - get argmax
            pred_np = torch.argmax(pred, dim=1).cpu().numpy()
            return pred_np.astype(np.uint8)
        else:
            # Binary segmentation
            # Remove batch and channel dimensions
            pred_np = pred.squeeze().cpu().numpy()
            
            # Convert to binary prediction
            pred_binary = (pred_np > 0.5).astype(np.uint16) * 65535
            
            return pred_binary
    
    def get_slice_stats(self, slice_data: np.ndarray) -> dict:
        """Get statistics for a slice.
        
        Args:
            slice_data: numpy array of any dtype
            
        Returns:
            dict with min, max, mean values
        """
        return {
            "min": float(slice_data.min()),
            "max": float(slice_data.max()),
            "mean": float(slice_data.mean())
        }
