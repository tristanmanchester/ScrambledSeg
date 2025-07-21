"""Module for preprocessing 3D volumes into 2D slices for training."""
import h5py
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, NamedTuple, Union
from tqdm import tqdm
import tifffile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class SliceInfo(NamedTuple):
    """Information about a slice's origin."""
    volume_id: str
    orientation: str
    slice_idx: int

def load_volume(file_path: str) -> np.ndarray:
    """Load volume data from either H5 or TIFF file.
    
    Args:
        file_path: Path to H5 or TIFF file
        
    Returns:
        numpy array of shape (D, H, W)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.h5':
        with h5py.File(file_path, 'r') as f:
            volume = f['data'][:]
    elif file_ext in ['.tif', '.tiff']:
        volume = tifffile.imread(file_path)
        # Ensure 3D array
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4:
            # Assume last dimension is channel, take first channel
            volume = volume[..., 0]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
        
    return volume

def extract_tiles_from_slice(
    data_path: str,
    label_path: str,
    output_dir: Path,
    tile_size: int = 256,
    overlap: int = 64,
) -> List[SliceInfo]:
    """Extract overlapping tiles from 3D volumes by extracting all 2D slices from all axes.
    
    Args:
        data_path: Path to data volume file (H5 or TIFF)
        label_path: Path to label volume file (H5 or TIFF)
        output_dir: Directory to save extracted tiles
        tile_size: Size of tiles (square)
        overlap: Overlap between adjacent tiles
        
    Returns:
        List of SliceInfo objects for each extracted tile
    """
    import sys
    from pathlib import Path
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from scrambledSeg.prediction.axis import AxisPredictor, Axis
    
    base_name = Path(data_path).stem
    logger.info(f"Processing volume: {base_name}")
    
    tile_info = []
    axis_handler = AxisPredictor()
    
    try:
        # Load 3D volumes
        data_volume = load_volume(data_path)
        label_volume = load_volume(label_path)
        
        # Ensure both are 3D volumes
        if data_volume.ndim != 3 or label_volume.ndim != 3:
            raise ValueError(f"Expected 3D volumes, got data: {data_volume.ndim}D, label: {label_volume.ndim}D")
        
        # Verify shapes match
        assert data_volume.shape == label_volume.shape, f"Data shape {data_volume.shape} != Label shape {label_volume.shape}"
        
        # Extract slices from all three axes
        for axis in [Axis.XY, Axis.YZ, Axis.XZ]:
            axis_name = axis.name.lower()
            
            # Get the number of slices for this axis
            volume_shape = axis_handler.get_volume_shape(axis, data_volume.shape)
            num_slices = volume_shape[0]
            
            for slice_idx in range(num_slices):
                # Extract 2D slices from the volume
                data_slice = axis_handler.get_slice(data_volume, axis, slice_idx)
                label_slice = axis_handler.get_slice(label_volume, axis, slice_idx)
                
                # Verify shapes match
                assert data_slice.shape == label_slice.shape, f"Data shape {data_slice.shape} != Label shape {label_slice.shape}"
                
                # Normalize data if needed
                if data_slice.max() > 1.0:
                    data_slice = data_slice.astype(np.float32) / 65535.0
                
                # Convert labels to uint8
                label_slice = label_slice.astype(np.uint8)
                
                # Get dimensions
                height, width = data_slice.shape
                
                # Calculate tile positions with overlap
                stride = tile_size - overlap
                tile_positions = []
                
                for y in range(0, height - tile_size + 1, stride):
                    for x in range(0, width - tile_size + 1, stride):
                        tile_positions.append((y, x))
                
                # Add edge tiles if needed
                if height % stride != 0:
                    y = max(0, height - tile_size)
                    for x in range(0, width - tile_size + 1, stride):
                        if (y, x) not in tile_positions:
                            tile_positions.append((y, x))
                            
                if width % stride != 0:
                    x = max(0, width - tile_size)
                    for y in range(0, height - tile_size + 1, stride):
                        if (y, x) not in tile_positions:
                            tile_positions.append((y, x))
                
                # Add the bottom-right corner tile if needed
                if height > tile_size and width > tile_size:
                    corner = (max(0, height - tile_size), max(0, width - tile_size))
                    if corner not in tile_positions:
                        tile_positions.append(corner)
                
                # Extract and save tiles
                for tile_idx, (y, x) in enumerate(tile_positions):
                    # Extract tile
                    data_tile = data_slice[y:y+tile_size, x:x+tile_size]
                    label_tile = label_slice[y:y+tile_size, x:x+tile_size]
                    
                    # Skip tiles that are too small (shouldn't happen with our logic above)
                    if data_tile.shape[0] < tile_size or data_tile.shape[1] < tile_size:
                        continue
                        
                    # Add channel dimension
                    data_tile = np.expand_dims(data_tile, 0)
                    label_tile = np.expand_dims(label_tile, 0)
                    
                    # Create unique filename including axis and slice info
                    global_tile_idx = len(tile_info)
                    tile_name = f"{base_name}_{axis_name}_slice{slice_idx:03d}_tile{tile_idx:03d}"
                    data_filename = output_dir / 'data' / f"{tile_name}.tif"
                    label_filename = output_dir / 'labels' / f"{tile_name}.tif"

                    os.makedirs(data_filename.parent, exist_ok=True)
                    os.makedirs(label_filename.parent, exist_ok=True)
                    
                    tifffile.imwrite(str(data_filename), data_tile)
                    tifffile.imwrite(str(label_filename), label_tile)
                    
                    tile_info.append(SliceInfo(base_name, f"{axis_name}_slice{slice_idx:03d}", tile_idx))
                
    except Exception as e:
        logger.error(f"Error processing slice {base_name}: {str(e)}")
        raise
        
    return tile_info

def calculate_total_tiles(data_paths: List[str], tile_size: int = 256, overlap: int = 64) -> int:
    """Calculate approximate total number of tiles across all slices from all axes."""
    import sys
    from pathlib import Path
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from scrambledSeg.prediction.axis import AxisPredictor, Axis
    
    total = 0
    axis_handler = AxisPredictor()
    
    for data_path in data_paths:
        # Load volume
        volume_data = load_volume(data_path)
        
        if volume_data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume_data.ndim}D")
        
        # Calculate tiles for all three axes
        for axis in [Axis.XY, Axis.YZ, Axis.XZ]:
            # Get the number of slices for this axis
            volume_shape = axis_handler.get_volume_shape(axis, volume_data.shape)
            num_slices = volume_shape[0]
            slice_height, slice_width = volume_shape[1], volume_shape[2]
            
            # Calculate tiles per slice
            stride = tile_size - overlap
            tiles_y = max(1, (slice_height - overlap) // stride)
            tiles_x = max(1, (slice_width - overlap) // stride)
            
            # Add extra tiles for edges if needed
            if slice_height % stride != 0:
                tiles_y += 1
            if slice_width % stride != 0:
                tiles_x += 1
                
            # Add tiles for all slices in this axis
            total += num_slices * tiles_y * tiles_x
        
    return total

def _process_slice(args):
    data_path, label_path, output_dir, tile_size, overlap = args
    slice_info = extract_tiles_from_slice(data_path, label_path, output_dir, tile_size=tile_size, overlap=overlap)
    return (data_path, label_path, slice_info)

def create_datasets(
    data_paths: List[str],
    label_paths: List[str],
    output_dir: Path,
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    random_seed: int = 42,
    tile_size: int = 256,
    overlap: int = 64,
) -> None:
    """Create datasets with correct pre-allocated sizes, using 24 threads in parallel."""
    # Validate split ratios
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(random_seed)
    
    # Calculate total tiles
    total_tiles = calculate_total_tiles(data_paths)
    logger.info(f"Total number of tiles across all slices: {total_tiles}")
    
    # Add a safety margin of 5%
    total_tiles = int(total_tiles * 1.05)
    logger.info(f"Allocating space for {total_tiles} tiles (including 5% safety margin)")
    
    # Create output directories
    for split_name in split_ratios.keys():
        (output_dir / split_name / 'data').mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)

    # Parallel processing of slices
    tasks = [(data_path, label_path, output_dir, args.tile_size if 'args' in locals() else tile_size, args.overlap if 'args' in locals() else overlap) 
             for data_path, label_path in zip(data_paths, label_paths)]
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(_process_slice, task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing slices in parallel"):
            results.append(f.result())

    # Assignment and file moving (sequential, as file moving may not be thread-safe)
    for data_path, label_path, slice_info in results:
        n_slices = len(slice_info)
        split_probs = np.array([split_ratios[split] for split in split_ratios.keys()])
        split_probs /= split_probs.sum()  # Normalize
        assignments = np.random.choice(list(split_ratios.keys()), size=n_slices, p=split_probs)

        for i, (split_name, info) in enumerate(zip(assignments, slice_info)):
            base_name = info.volume_id
            orient_name = info.orientation
            tile_idx = info.slice_idx

            # The actual filename format we create
            tile_name = f"{base_name}_{orient_name}_tile{tile_idx:03d}"
            data_filename = output_dir / 'data' / f"{tile_name}.tif"
            label_filename = output_dir / 'labels' / f"{tile_name}.tif"

            new_data_filename = output_dir / split_name / 'data' / f"{tile_name}.tif"
            new_label_filename = output_dir / split_name / 'labels' / f"{tile_name}.tif"

            # Check if source files exist before trying to move them
            if data_filename.exists() and label_filename.exists():
                os.rename(data_filename, new_data_filename)
                os.rename(label_filename, new_label_filename)
            else:
                logger.warning(f"Missing files: {data_filename} or {label_filename}")

    logger.info("Dataset creation complete!")


def find_matching_volume_pairs(data_dir: Path, label_dir: Path) -> List[Tuple[str, str]]:
    """Find matching volume files (H5, TIFF, TIF) in data and label directories.
    
    Handles both:
    1. Files with identical stems (e.g., 'volume1.tif' and 'volume1.tif')
    2. Files with different naming patterns but matching indices
       (e.g., 'synthetic_slice_000.tif' and 'label_image_000.tif')
    """
    # Get all tiff and h5 files
    data_files = []
    for ext in ['*.h5', '*.tif', '*.tiff']:
        data_files.extend(list(data_dir.glob(ext)))
        
    label_files = []
    for ext in ['*.h5', '*.tif', '*.tiff']:
        label_files.extend(list(label_dir.glob(ext)))
    
    # First try: Match by identical stems
    data_by_stem = {f.stem: f for f in data_files}
    label_by_stem = {f.stem: f for f in label_files}
    common_stems = set(data_by_stem.keys()) & set(label_by_stem.keys())
    
    if common_stems:
        # Return sorted pairs of full paths with matching stems
        return sorted([
            (str(data_by_stem[stem]), str(label_by_stem[stem]))
            for stem in common_stems
        ])
    
    # Second try: Match by numerical indices
    # Extract indices from filenames using regex
    import re
    
    data_by_index = {}
    for f in data_files:
        # Try to find a number in the filename
        match = re.search(r'(\d+)', f.stem)
        if match:
            index = match.group(1)
            data_by_index[index] = f
    
    label_by_index = {}
    for f in label_files:
        match = re.search(r'(\d+)', f.stem)
        if match:
            index = match.group(1)
            label_by_index[index] = f
    
    # Find common indices
    common_indices = set(data_by_index.keys()) & set(label_by_index.keys())
    
    if common_indices:
        # Return sorted pairs of full paths with matching indices
        return sorted([
            (str(data_by_index[idx]), str(label_by_index[idx]))
            for idx in common_indices
        ])
    
    # If no matches found by either method
    raise ValueError(f"No matching volume files found in {data_dir} and {label_dir}. " +
                     "Files should either have identical names or contain matching numerical indices.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess 2D slices into overlapping tiles")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing data slice files (TIFF)")
    parser.add_argument("--label-dir", type=str, required=True,
                      help="Directory containing label slice files (TIFF)")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save tile datasets")
    parser.add_argument("--tile-size", type=int, default=512,
                      help="Size of tiles (square)")
    parser.add_argument("--overlap", type=int, default=32,
                      help="Overlap between adjacent tiles")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                      help="Ratio of validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                      help="Ratio of test set")
    parser.add_argument("--seed", type=int, default=69,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path(args.data_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    
    volume_pairs = find_matching_volume_pairs(data_dir, label_dir)
    data_paths = [pair[0] for pair in volume_pairs]
    label_paths = [pair[1] for pair in volume_pairs]
    
    create_datasets(
        data_paths=data_paths,
        label_paths=label_paths,
        output_dir=output_dir,
        split_ratios={"train": 1 - args.val_ratio - args.test_ratio, 
                     "val": args.val_ratio, 
                     "test": args.test_ratio},
        random_seed=args.seed,
        tile_size=args.tile_size,
        overlap=args.overlap
    )