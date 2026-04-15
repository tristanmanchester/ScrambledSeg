"""Axis handling utilities for tomographic predictions."""
import numpy as np
from enum import Enum, auto
from typing import Tuple, Optional
import logging
from scipy.ndimage import rotate

logger = logging.getLogger(__name__)

class Axis(Enum):
    """Enumeration of possible prediction axes."""
    XY = auto()  # Slice along depth, plane is (H, W)
    YZ = auto()  # Slice along width, plane is (D, H)
    XZ = auto()  # Slice along height, plane is (D, W)

class AxisPredictor:
    """Handles 3D volume slicing and reconstruction for multi-axis prediction."""
    
    def __init__(self):
        """Initialize axis predictor."""
        # Define axis permutations for each view
        self._axis_permutations = {
            Axis.XY: (0, 1, 2),  # No permutation needed
            Axis.YZ: (2, 0, 1),  # Slice along width
            Axis.XZ: (1, 0, 2),  # Slice along height
        }

    def _get_view_permutation(self, axis: Axis, ndim: int) -> Tuple[int, ...]:
        """Return the permutation needed for a given array dimensionality."""
        if ndim == 3:
            return self._axis_permutations[axis]
        if ndim == 4:
            spatial_axes = [0, 2, 3]
            spatial_permutation = [spatial_axes[idx] for idx in self._axis_permutations[axis]]
            return (spatial_permutation[0], 1, spatial_permutation[1], spatial_permutation[2])
        raise ValueError(f"Unsupported volume ndim {ndim}. Expected 3 or 4.")

    def _get_view(self, volume: np.ndarray, axis: Axis) -> np.ndarray:
        """Return the axis-aligned view for 3D or 4D volumes."""
        return np.transpose(volume, self._get_view_permutation(axis, volume.ndim))
    
    def get_slice(self, volume: np.ndarray, axis: Axis, idx: int) -> np.ndarray:
        """Extract a 2D slice from the volume along specified axis.
        
        Args:
            volume: Input volume of shape (D, H, W)
            axis: Axis to extract slice from
            idx: Index of slice to extract
            
        Returns:
            2D numpy array representing the slice
        """
        if not isinstance(axis, Axis):
            raise ValueError(f"Invalid axis: {axis}. Must be an Axis enum.")

        volume_view = self._get_view(volume, axis)
        if idx < 0 or idx >= volume_view.shape[0]:
            raise ValueError(f"Invalid index {idx} for axis {axis}. Must be between 0 and {volume_view.shape[0] - 1}")
        
        # Extract slice
        slice_data = volume_view[idx]
        
        return slice_data
    
    def set_slice(self, volume: np.ndarray, axis: Axis, idx: int, slice_data: np.ndarray, 
                 out: Optional[np.ndarray] = None) -> np.ndarray:
        """Set a 2D slice in the volume along specified axis.
        
        Args:
            volume: Input volume of shape (D, H, W)
            axis: Axis to set slice in
            idx: Index to set slice at
            slice_data: 2D array to insert
            out: Optional output array to store result
            
        Returns:
            Updated volume with new slice
        """
        if not isinstance(axis, Axis):
            raise ValueError(f"Invalid axis: {axis}. Must be an Axis enum.")
        
        # Create output array if needed
        if out is None:
            out = volume.copy()
        
        # Permute axes to get desired view
        out_view = self._get_view(out, axis)
        if idx < 0 or idx >= out_view.shape[0]:
            raise ValueError(f"Invalid index {idx} for axis {axis}. Must be between 0 and {out_view.shape[0] - 1}")
        
        # Set slice
        out_view[idx] = slice_data

        return out
    
    def rotate_slice(self, slice_data: np.ndarray, angle: float, 
                    order: int = 1) -> np.ndarray:
        """Rotate a 2D slice by specified angle clockwise.
        
        Args:
            slice_data: Input 2D array
            angle: Rotation angle in degrees (clockwise)
            order: Interpolation order (0=nearest, 1=linear, etc.)
            
        Returns:
            Rotated slice
        """
        if angle == 0:
            return slice_data
            
        # Convert to counter-clockwise for scipy.rotate
        angle = -angle
            
        # Rotate using scipy's rotate function
        rotated = rotate(slice_data, angle, reshape=False, order=order)
        
        return rotated
    
    def get_volume_shape(self, axis: Axis, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get the shape of the volume when viewed from specified axis.
        
        Args:
            axis: Viewing axis
            shape: Original shape tuple
            
        Returns:
            Shape tuple for the specified axis view
        """
        if len(shape) == 3:
            return tuple(np.array(shape)[list(self._axis_permutations[axis])])
        if len(shape) == 4:
            permutation = self._get_view_permutation(axis, len(shape))
            return tuple(np.array(shape)[list(permutation)])
        raise ValueError(f"Unsupported shape {shape}. Expected 3D or 4D volume.")
    
    @staticmethod
    def create_empty_volume(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create an empty volume for storing predictions.
        
        Args:
            shape: Shape of volume to create
            dtype: Data type of volume
            
        Returns:
            Empty volume initialized to zeros
        """
        return np.zeros(shape, dtype=dtype)
