"""Utilities for handling large TIFF images with tiling."""
import numpy as np
from PIL import Image
import math
from typing import Tuple, List, Iterator, Union
import logging
import tifffile

logger = logging.getLogger(__name__)

class TiffHandler:
    """Handles loading, saving and tiling of TIFF images."""
    
    def __init__(self, tile_size: int = 512, overlap: int = 128):
        """Initialize TIFF handler.
        
        Args:
            tile_size: Size of tiles to split image into
            overlap: Overlap between tiles (should be double the desired trim size)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.trim_size = overlap // 2  # Amount to trim from each edge
        
    def load_tiff(self, path: str) -> Union[np.ndarray, List[np.ndarray]]:
        """Load TIFF image as numpy array. Supports both single and multipage TIFFs.
        
        Args:
            path: Path to TIFF file
            
        Returns:
            For single page: numpy array of shape (H, W) or (H, W, C)
            For multipage: numpy array of shape (Z, H, W) or (Z, H, W, C)
        """
        try:
            # First try tifffile for multipage support
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) > 1:
                    # Handle multipage TIFF
                    data = tifffile.imread(path)
                    if data.ndim == 3:  # Z,H,W
                        data = data.transpose(0, 1, 2)  # Ensure Z,H,W order
                    elif data.ndim == 4:  # Z,H,W,C
                        data = data.transpose(0, 1, 2, 3)  # Ensure Z,H,W,C order
                else:
                    # Single page TIFF
                    data = tifffile.imread(path)
        except Exception as e:
            logger.warning(f"Failed to load with tifffile, falling back to PIL: {e}")
            # Fallback to PIL for simple TIFFs
            with Image.open(path) as img:
                data = np.array(img)
            
        logger.info(f"Loaded image of shape {data.shape} and dtype {data.dtype}")
        return data
    
    def save_tiff(self, data: np.ndarray, path: str, threshold: float = 0.5):
        """Save numpy array as TIFF image. Supports both single and multipage TIFFs.
        
        Args:
            data: For single page: numpy array of shape (H, W) or (H, W, C)
                 For multipage: numpy array of shape (Z, H, W) or (Z, H, W, C)
            path: Output path
            threshold: Threshold for converting probabilities to binary (default: 0.5)
        """
        # Convert probabilities to class indices if needed
        if data.dtype in [np.float32, np.float64]:
            # For multi-class segmentation, the data is already in class indices (0, 1, 2, etc.)
            # We don't want to threshold it, as that would convert to binary (0, 1)
            # Instead, just convert to uint8 as-is
            if np.max(data) <= 10:  # Likely class indices
                logger.info(f"Saving multi-class segmentation with classes 0-{int(np.max(data))}")
                data = data.astype(np.uint8)
            else:
                # For probability maps (e.g., for binary segmentation)
                logger.info("Saving binary segmentation with threshold")
                data = (data > threshold).astype(np.uint8) * 255
            
        # Handle multipage TIFF
        if data.ndim in [3, 4] and (data.shape[0] > 1 or data.ndim == 4):
            tifffile.imwrite(path, data)
        else:
            # Single page TIFF - use PIL for compatibility
            img = Image.fromarray(data)
            img.save(path)
            
        logger.info(f"Saved image to {path} with shape {data.shape} and values in range [{np.min(data)}-{np.max(data)}]")
    
    def get_tile_locations(self, image_shape: Tuple[int, ...]) -> List[Tuple[slice, slice, bool, bool]]:
        """Get list of tile locations as slice tuples.
        
        Args:
            image_shape: Shape of input image (H,W) or (H,W,C)
            
        Returns:
            List of (row_slice, col_slice, is_edge_row, is_edge_col) tuples for each tile
        """
        # Use only the height and width for tiling
        height, width = image_shape[:2]
        effective_tile_size = self.tile_size - self.overlap  # Size after trimming
        
        # Calculate number of tiles needed
        n_tiles_h = math.ceil(height / effective_tile_size)
        n_tiles_w = math.ceil(width / effective_tile_size)
        
        tile_locations = []
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate base positions
                row_start = i * effective_tile_size
                col_start = j * effective_tile_size
                
                # Determine if this is an edge tile
                is_edge_row = (i == 0 or i == n_tiles_h - 1)
                is_edge_col = (j == 0 or j == n_tiles_w - 1)
                
                # For edge tiles, adjust to maintain tile_size
                if i == n_tiles_h - 1:  # Last row
                    row_start = max(0, height - self.tile_size)
                if j == n_tiles_w - 1:  # Last column
                    col_start = max(0, width - self.tile_size)
                
                # Calculate end positions
                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)
                
                tile_locations.append(
                    (slice(row_start, row_end),
                     slice(col_start, col_end),
                     is_edge_row,
                     is_edge_col)
                )
        
        return tile_locations
    
    def merge_tiles(self, tiles: List[Tuple[np.ndarray, Tuple[slice, slice, bool, bool]]], output_shape: Tuple[int, ...]) -> np.ndarray:
        """Merge tiles back into a single image, handling overlaps.
        
        Args:
            tiles: List of (tile_data, (row_slice, col_slice, is_edge_row, is_edge_col)) tuples
            output_shape: Shape of the output image
            
        Returns:
            Merged image array
        """
        # For multi-class segmentation, we need to handle class indices differently
        # Check if we're dealing with class indices (small ints)
        is_multiclass = False
        max_value = 0
        for tile_data, _ in tiles:
            max_value = max(max_value, np.max(tile_data))
            if max_value <= 10 and tile_data.dtype in [np.int64, np.int32, np.uint8, np.int8]:
                is_multiclass = True
                break
        
        if is_multiclass:
            logger.info("Detected multi-class segmentation, using special merging for class indices")
            return self._merge_tiles_multiclass(tiles, output_shape)
        else:
            logger.info("Using standard weighted merging for continuous values")
            return self._merge_tiles_standard(tiles, output_shape)
    
    def _merge_tiles_multiclass(self, tiles: List[Tuple[np.ndarray, Tuple[slice, slice, bool, bool]]], output_shape: Tuple[int, ...]) -> np.ndarray:
        """Merge tiles for multi-class segmentation.
        
        For class indices, we can't simply blend values as that would create non-integer classes.
        Instead, we use a voting mechanism with increased confidence in the central regions.
        """
        # Determine number of classes from max value
        max_class = 0
        for tile_data, _ in tiles:
            max_class = max(max_class, np.max(tile_data))
        num_classes = int(max_class) + 1  # Add one for class 0
        
        logger.info(f"Merging multi-class tiles with {num_classes} classes")
        
        # Initialize confidence maps for each class
        class_confidences = np.zeros((num_classes,) + output_shape[:2], dtype=np.float32)
        
        for tile_data, (row_slice, col_slice, is_edge_row, is_edge_col) in tiles:
            # Create weight mask for this tile (higher weights in center)
            weight_mask = np.ones((row_slice.stop - row_slice.start, 
                                  col_slice.stop - col_slice.start))
            
            # Apply stronger weight falloff at edges to reduce border artifacts
            border_ratio = 0.7  # How much of the overlap to consider as border
            border_size = int(self.trim_size * border_ratio) if self.trim_size > 0 else 0
            if border_size == 0 and self.trim_size > 0:
                border_size = 1
            center_weight = 3.0  # Center regions have higher confidence

            # Apply weight profile
            if not is_edge_row and border_size > 0:
                # Top edge falloff
                weight_mask[:border_size, :] = np.linspace(0.1, 1, border_size)[:, np.newaxis]
                # Bottom edge falloff
                weight_mask[-border_size:, :] = np.linspace(1, 0.1, border_size)[:, np.newaxis]
                # Center boost
                center_start = border_size
                center_end = weight_mask.shape[0] - border_size
                if center_end > center_start:
                    weight_mask[center_start:center_end, :] *= center_weight

            if not is_edge_col and border_size > 0:
                # Left edge falloff
                weight_mask[:, :border_size] *= np.linspace(0.1, 1, border_size)[np.newaxis, :]
                # Right edge falloff
                weight_mask[:, -border_size:] *= np.linspace(1, 0.1, border_size)[np.newaxis, :]
                # Center boost for columns
                center_start = border_size
                center_end = weight_mask.shape[1] - border_size
                if center_end > center_start:
                    weight_mask[:, center_start:center_end] *= center_weight
            
            valid_height, valid_width = weight_mask.shape
            tile_region = tile_data[:valid_height, :valid_width]

            # Add votes for each class
            for class_idx in range(num_classes):
                class_mask = (tile_region == class_idx).astype(np.float32)
                class_confidences[class_idx, row_slice, col_slice] += class_mask * weight_mask
        
        # Get class with highest confidence at each pixel
        output = np.argmax(class_confidences, axis=0).astype(np.uint8)
        
        return output
    
    def _merge_tiles_standard(self, tiles: List[Tuple[np.ndarray, Tuple[slice, slice, bool, bool]]], output_shape: Tuple[int, ...]) -> np.ndarray:
        """Merge tiles with standard weighted blending for continuous values.
        
        This is used for probability maps and other continuous value predictions.
        """
        # Initialize output array and weight map for blending
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape[:2], dtype=np.float32)
        
        for tile_data, (row_slice, col_slice, is_edge_row, is_edge_col) in tiles:
            # Create weight mask for this tile
            weight_mask = np.ones((row_slice.stop - row_slice.start, 
                                 col_slice.stop - col_slice.start))
            
            # Apply linear falloff in overlap regions
            if not is_edge_row:
                # Top edge falloff
                weight_mask[:self.trim_size, :] = np.linspace(0, 1, self.trim_size)[:, np.newaxis]
                # Bottom edge falloff
                weight_mask[-self.trim_size:, :] = np.linspace(1, 0, self.trim_size)[:, np.newaxis]
            
            if not is_edge_col:
                # Left edge falloff
                weight_mask[:, :self.trim_size] *= np.linspace(0, 1, self.trim_size)[np.newaxis, :]
                # Right edge falloff
                weight_mask[:, -self.trim_size:] *= np.linspace(1, 0, self.trim_size)[np.newaxis, :]
            
            # Add weighted tile to output
            output[row_slice, col_slice] += tile_data * weight_mask[..., np.newaxis] if tile_data.ndim == 3 else tile_data * weight_mask
            weights[row_slice, col_slice] += weight_mask
        
        # Normalize by weights
        valid_mask = weights > 0
        output[valid_mask] /= weights[valid_mask][..., np.newaxis] if output.ndim == 3 else weights[valid_mask]
        
        return output
    
    def iter_tiles(self, image: np.ndarray) -> Iterator[Tuple[np.ndarray, Tuple[slice, slice, bool, bool]]]:
        """Iterate over tiles in an image.
        
        Args:
            image: Input image array
            
        Yields:
            Tuple of (tile_data, (row_slice, col_slice, is_edge_row, is_edge_col))
        """
        # Ensure we're only processing 2D images at this point
        if len(image.shape) == 3 and image.shape[2] > 10:  # Likely a 3D volume incorrectly passed
            raise ValueError(f"Received 3D volume with shape {image.shape}. Use predict_image with 3D handling instead.")
            
        tile_locations = self.get_tile_locations(image.shape)
        
        for row_slice, col_slice, is_edge_row, is_edge_col in tile_locations:
            tile = image[row_slice, col_slice]
            
            # Pad if necessary to maintain tile size
            if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                padded_tile = np.zeros((self.tile_size, self.tile_size) + tile.shape[2:], dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            yield tile, (row_slice, col_slice, is_edge_row, is_edge_col)
