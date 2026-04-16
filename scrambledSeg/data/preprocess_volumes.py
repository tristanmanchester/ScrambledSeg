"""Preprocess paired 3D volumes into tiled 2D datasets for training."""

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, TypeAlias

import h5py
import numpy as np
import tifffile
from tqdm import tqdm

from ..axis import Axis, AxisPredictor

logger = logging.getLogger(__name__)

VolumePaths: TypeAlias = List[str]
VolumePathPair: TypeAlias = Tuple[str, str]
SplitRatios: TypeAlias = Dict[str, float]
DEFAULT_SPLIT_RATIOS: SplitRatios = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_TILE_OVERLAP = 32


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

    if file_ext == ".h5":
        with h5py.File(file_path, "r") as f:
            volume = f["data"][:]
    elif file_ext in [".tif", ".tiff"]:
        volume = tifffile.imread(file_path)
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4:
            volume = volume[..., 0]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    return volume


def extract_tiles_from_slice(
    data_path: str,
    label_path: str,
    output_dir: Path,
    tile_size: int = 256,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> List[SliceInfo]:
    """Extract overlapping tiles from every orthogonal slice of a paired volume.

    Args:
        data_path: Path to data volume file (H5 or TIFF)
        label_path: Path to label volume file (H5 or TIFF)
        output_dir: Directory to save extracted tiles
        tile_size: Size of tiles (square)
        overlap: Overlap between adjacent tiles

    Returns:
        List of SliceInfo objects for each extracted tile
    """
    base_name = Path(data_path).stem
    logger.info(f"Processing volume: {base_name}")

    tile_info = []
    axis_handler = AxisPredictor()
    data_volume = load_volume(data_path)
    label_volume = load_volume(label_path)

    if data_volume.ndim != 3 or label_volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volumes, got data: {data_volume.ndim}D, label: {label_volume.ndim}D"
        )

    if data_volume.shape != label_volume.shape:
        raise ValueError(f"Data shape {data_volume.shape} != Label shape {label_volume.shape}")

    for axis in [Axis.XY, Axis.YZ, Axis.XZ]:
        axis_name = axis.name.lower()
        volume_shape = axis_handler.get_volume_shape(axis, data_volume.shape)
        num_slices = volume_shape[0]

        for slice_idx in range(num_slices):
            data_slice = axis_handler.get_slice(data_volume, axis, slice_idx)
            label_slice = axis_handler.get_slice(label_volume, axis, slice_idx)

            if data_slice.shape != label_slice.shape:
                raise ValueError(
                    f"Data shape {data_slice.shape} != Label shape {label_slice.shape}"
                )

            if data_slice.max() > 1.0:
                data_slice = data_slice.astype(np.float32) / 65535.0

            label_slice = label_slice.astype(np.uint8)
            height, width = data_slice.shape

            stride = tile_size - overlap
            tile_positions = []

            for y in range(0, height - tile_size + 1, stride):
                for x in range(0, width - tile_size + 1, stride):
                    tile_positions.append((y, x))

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

            if height > tile_size and width > tile_size:
                corner = (max(0, height - tile_size), max(0, width - tile_size))
                if corner not in tile_positions:
                    tile_positions.append(corner)

            for tile_idx, (y, x) in enumerate(tile_positions):
                data_tile = data_slice[y : y + tile_size, x : x + tile_size]
                label_tile = label_slice[y : y + tile_size, x : x + tile_size]

                if data_tile.shape[0] < tile_size or data_tile.shape[1] < tile_size:
                    continue

                data_tile = np.expand_dims(data_tile, 0)
                label_tile = np.expand_dims(label_tile, 0)

                tile_name = f"{base_name}_{axis_name}_slice{slice_idx:03d}_tile{tile_idx:03d}"
                data_filename = output_dir / "data" / f"{tile_name}.tif"
                label_filename = output_dir / "labels" / f"{tile_name}.tif"

                data_filename.parent.mkdir(parents=True, exist_ok=True)
                label_filename.parent.mkdir(parents=True, exist_ok=True)

                tifffile.imwrite(str(data_filename), data_tile)
                tifffile.imwrite(str(label_filename), label_tile)

                tile_info.append(
                    SliceInfo(base_name, f"{axis_name}_slice{slice_idx:03d}", tile_idx)
                )

    return tile_info


def calculate_total_tiles(
    data_paths: VolumePaths,
    tile_size: int = 256,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> int:
    """Calculate approximate total number of tiles across all slices from all axes."""
    total = 0
    axis_handler = AxisPredictor()

    for data_path in data_paths:
        volume_data = load_volume(data_path)

        if volume_data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume_data.ndim}D")

        for axis in [Axis.XY, Axis.YZ, Axis.XZ]:
            volume_shape = axis_handler.get_volume_shape(axis, volume_data.shape)
            num_slices = volume_shape[0]
            slice_height, slice_width = volume_shape[1], volume_shape[2]

            stride = tile_size - overlap
            tiles_y = max(1, (slice_height - overlap) // stride)
            tiles_x = max(1, (slice_width - overlap) // stride)

            if slice_height % stride != 0:
                tiles_y += 1
            if slice_width % stride != 0:
                tiles_x += 1

            total += num_slices * tiles_y * tiles_x

    return total


def _process_slice(args):
    data_path, label_path, output_dir, tile_size, overlap = args
    slice_info = extract_tiles_from_slice(
        data_path, label_path, output_dir, tile_size=tile_size, overlap=overlap
    )
    return (data_path, label_path, slice_info)


def create_datasets(
    data_paths: VolumePaths,
    label_paths: VolumePaths,
    output_dir: Path,
    split_ratios: Optional[SplitRatios] = None,
    random_seed: int = 42,
    tile_size: int = 256,
    overlap: int = DEFAULT_TILE_OVERLAP,
) -> None:
    """Create train/val/test tile datasets from paired 3D input volumes."""
    split_ratios = DEFAULT_SPLIT_RATIOS.copy() if split_ratios is None else split_ratios

    if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_seed)

    total_tiles = calculate_total_tiles(data_paths, tile_size=tile_size, overlap=overlap)
    logger.info(f"Total number of tiles across all slices: {total_tiles}")

    total_tiles = int(total_tiles * 1.05)
    logger.info(f"Allocating space for {total_tiles} tiles (including 5% safety margin)")

    for split_name in split_ratios.keys():
        (output_dir / split_name / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            data_path,
            label_path,
            output_dir,
            tile_size,
            overlap,
        )
        for data_path, label_path in zip(data_paths, label_paths)
    ]
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(_process_slice, task) for task in tasks]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Processing slices in parallel"
        ):
            results.append(f.result())

    for data_path, label_path, slice_info in results:
        n_slices = len(slice_info)
        split_probs = np.array([split_ratios[split] for split in split_ratios.keys()])
        split_probs /= split_probs.sum()
        assignments = np.random.choice(list(split_ratios.keys()), size=n_slices, p=split_probs)

        for split_name, info in zip(assignments, slice_info):
            base_name = info.volume_id
            orient_name = info.orientation
            tile_idx = info.slice_idx

            tile_name = f"{base_name}_{orient_name}_tile{tile_idx:03d}"
            data_filename = output_dir / "data" / f"{tile_name}.tif"
            label_filename = output_dir / "labels" / f"{tile_name}.tif"

            new_data_filename = output_dir / split_name / "data" / f"{tile_name}.tif"
            new_label_filename = output_dir / split_name / "labels" / f"{tile_name}.tif"

            if data_filename.exists() and label_filename.exists():
                data_filename.rename(new_data_filename)
                label_filename.rename(new_label_filename)
            else:
                raise FileNotFoundError(
                    "Expected extracted tile files before dataset split assignment, but missing "
                    f"{data_filename} or {label_filename}."
                )

    logger.info("Dataset creation complete!")


def find_matching_volume_pairs(data_dir: Path, label_dir: Path) -> List[VolumePathPair]:
    """Find matching volume files (H5, TIFF, TIF) in data and label directories.

    Handles both:
    1. Files with identical stems (e.g., 'volume1.tif' and 'volume1.tif')
    2. Files with different naming patterns but matching indices
       (e.g., 'synthetic_slice_000.tif' and 'label_image_000.tif')
    """
    data_files = []
    for ext in ["*.h5", "*.tif", "*.tiff"]:
        data_files.extend(list(data_dir.glob(ext)))

    label_files = []
    for ext in ["*.h5", "*.tif", "*.tiff"]:
        label_files.extend(list(label_dir.glob(ext)))

    data_by_stem = {f.stem: f for f in data_files}
    label_by_stem = {f.stem: f for f in label_files}
    common_stems = set(data_by_stem.keys()) & set(label_by_stem.keys())

    if common_stems:
        return sorted(
            [(str(data_by_stem[stem]), str(label_by_stem[stem])) for stem in common_stems]
        )

    data_by_index = {}
    for f in data_files:
        match = re.search(r"(\d+)", f.stem)
        if match:
            index = match.group(1)
            data_by_index[index] = f

    label_by_index = {}
    for f in label_files:
        match = re.search(r"(\d+)", f.stem)
        if match:
            index = match.group(1)
            label_by_index[index] = f

    common_indices = set(data_by_index.keys()) & set(label_by_index.keys())

    if common_indices:
        return sorted(
            [(str(data_by_index[idx]), str(label_by_index[idx])) for idx in common_indices]
        )

    raise ValueError(
        f"No matching volume files found in {data_dir} and {label_dir}. "
        "Files should either have identical names or contain matching numerical indices."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the preprocessing CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Preprocess paired 3D H5/TIFF volumes into train/val/test tile datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data slice files (TIFF)",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="Directory containing label slice files (TIFF)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save tile datasets",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Size of tiles (square)")
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help=f"Overlap between adjacent tiles (default: {DEFAULT_TILE_OVERLAP})",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of test set")
    parser.add_argument("--seed", type=int, default=69, help="Random seed for reproducibility")
    return parser


def main() -> None:
    """Run the preprocessing CLI."""

    parser = build_argument_parser()
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
        split_ratios={
            "train": 1 - args.val_ratio - args.test_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        random_seed=args.seed,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
