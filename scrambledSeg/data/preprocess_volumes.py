"""Module for preprocessing 3D volumes into 2D slices for training."""
import h5py
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, NamedTuple
from tqdm import tqdm
import tifffile
import os

logger = logging.getLogger(__name__)

class SliceInfo(NamedTuple):
    """Information about a slice's origin."""
    volume_id: str
    orientation: str
    slice_idx: int

def extract_volume_slices(
    data_path: str,
    label_path: str,
    output_dir: Path,
) -> List[SliceInfo]:
    """Extract slices from volume in all orientations and save as TIFF."""
    base_name = Path(data_path).stem
    logger.info(f"Processing volume: {base_name}")
    
    slice_info = []
    
    try:
        with h5py.File(data_path, 'r') as f_data, h5py.File(label_path, 'r') as f_label:
            data_vol = f_data['data'][:]
            label_vol = f_label['data'][:]
            
            assert data_vol.shape == (512, 512, 512), f"Data volume shape {data_vol.shape} != (512, 512, 512)"
            assert label_vol.shape == (512, 512, 512), f"Label volume shape {label_vol.shape} != (512, 512, 512)"
            
            # Process each orientation
            orientations = [
                ('XY', lambda v: v),                     # Original orientation
                ('YZ', lambda v: np.swapaxes(v, 0, 2)),  # YZ plane
                ('XZ', lambda v: np.swapaxes(v, 0, 1))   # XZ plane
            ]
            
            for orient_name, transform_fn in orientations:
                # Transform whole volume for this orientation
                data_oriented = transform_fn(data_vol)
                label_oriented = transform_fn(label_vol)
                
                # Process slices in chunks
                for slice_idx in range(data_oriented.shape[0]):
                    # Extract full slices
                    data_slice = data_oriented[slice_idx].astype(np.float32) / 65535.0
                    label_slice = (label_oriented[slice_idx] > 0).astype(np.float32)
                    
                    # Add channel dimension
                    data_slice = np.expand_dims(data_slice, 0)
                    label_slice = np.expand_dims(label_slice, 0)
                    
                    # Ensure contiguous memory layout
                    # data_slice = np.ascontiguousarray(data_slice)
                    # label_slice = np.ascontiguousarray(label_slice)
                    
                    # Save slices
                    data_filename = output_dir / 'data' / f"{base_name}_{orient_name}_{slice_idx}.tif"
                    label_filename = output_dir / 'labels' / f"{base_name}_{orient_name}_{slice_idx}.tif"

                    os.makedirs(data_filename.parent, exist_ok=True)
                    os.makedirs(label_filename.parent, exist_ok=True)
                    
                    tifffile.imwrite(str(data_filename), data_slice, compression=None)
                    tifffile.imwrite(str(label_filename), label_slice, compression=None)
                    
                    slice_info.append(SliceInfo(base_name, orient_name, slice_idx))
                    
    except Exception as e:
        logger.error(f"Error processing volume {base_name}: {str(e)}")
        raise
        
    return slice_info

def calculate_total_slices(data_paths: List[str]) -> int:
    """Calculate exact total number of slices across all volumes and orientations."""
    total = 0
    for data_path in data_paths:
        with h5py.File(data_path, 'r') as f:
            shape = f['data'].shape
            # For a 512x512x512 volume, we get:
            # XY: 512 slices
            # YZ: 512 slices
            # XZ: 512 slices
            # Total per volume: 512 * 3 = 1536 slices
            total += shape[0] * 3  # multiply by 3 for the three orientations
    return total

def create_datasets(
    data_paths: List[str],
    label_paths: List[str],
    output_dir: Path,
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    random_seed: int = 42,
) -> None:
    """Create datasets with correct pre-allocated sizes."""
    # Validate split ratios
    if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(random_seed)
    
    # Calculate exact total number of slices
    total_slices = calculate_total_slices(data_paths)
    logger.info(f"Total number of slices across all volumes: {total_slices}")
    
    # Add a safety margin of 5%
    total_slices = int(total_slices * 1.05)
    logger.info(f"Allocating space for {total_slices} slices (including 5% safety margin)")
    
    # Create output directories
    for split_name in split_ratios.keys():
         (output_dir / split_name / 'data').mkdir(parents=True, exist_ok=True)
         (output_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)

    # Process each volume
    current_positions = {split: 0 for split in split_ratios.keys()}
    
    for data_path, label_path in tqdm(zip(data_paths, label_paths), total=len(data_paths)):
        # Process single volume
        slice_info = extract_volume_slices(data_path, label_path, output_dir)
        
        # Randomly assign slices to splits
        n_slices = len(slice_info)
        split_probs = np.array([split_ratios[split] for split in split_ratios.keys()])
        split_probs /= split_probs.sum()  # Normalize
        assignments = np.random.choice(list(split_ratios.keys()), size=n_slices, p=split_probs)
        
         # Assign slices
        for i, (split_name, info) in enumerate(zip(assignments, slice_info)):
           
            base_name = info.volume_id
            orient_name = info.orientation
            slice_idx = info.slice_idx
            
            # Create the path
            data_filename = output_dir / 'data' / f"{base_name}_{orient_name}_{slice_idx}.tif"
            label_filename = output_dir / 'labels' / f"{base_name}_{orient_name}_{slice_idx}.tif"
                
            # create the new path
            new_data_filename = output_dir / split_name / 'data' / f"{base_name}_{orient_name}_{slice_idx}.tif"
            new_label_filename = output_dir / split_name / 'labels' / f"{base_name}_{orient_name}_{slice_idx}.tif"
            
            # Move the files
            os.rename(data_filename, new_data_filename)
            os.rename(label_filename, new_label_filename)
                
    
    logger.info("Dataset creation complete!")
           
    
    logger.info("Dataset creation complete!")

def find_matching_h5_pairs(data_dir: Path, label_dir: Path) -> List[Tuple[str, str]]:
    """Find matching H5 files in data and label directories."""
    data_files = {f.stem: f for f in data_dir.glob('*.h5')}
    label_files = {f.stem: f for f in label_dir.glob('*.h5')}
    
    # Find common stems
    common_stems = set(data_files.keys()) & set(label_files.keys())
    if not common_stems:
        raise ValueError(f"No matching H5 files found in {data_dir} and {label_dir}")
    
    # Return sorted pairs of full paths
    return sorted([
        (str(data_files[stem]), str(label_files[stem]))
        for stem in common_stems
    ])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess volume H5 files into H5 datasets")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing data H5 files")
    parser.add_argument("--label-dir", type=str, required=True,
                      help="Directory containing label H5 files")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save H5 datasets")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                      help="Ratio of validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                      help="Ratio of test set")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path(args.data_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    
    h5_pairs = find_matching_h5_pairs(data_dir, label_dir)
    data_paths = [pair[0] for pair in h5_pairs]
    label_paths = [pair[1] for pair in h5_pairs]
    
    create_datasets(
        data_paths=data_paths,
        label_paths=label_paths,
        output_dir=output_dir,
        split_ratios={"train": 1 - args.val_ratio - args.test_ratio, 
                     "val": args.val_ratio, 
                     "test": args.test_ratio},
        random_seed=args.seed
    )