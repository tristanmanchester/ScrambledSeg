import logging
from pathlib import Path
from typing import Optional, Tuple, TypedDict, Union
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import albumentations as A

logger = logging.getLogger(__name__)


class ArraySample(TypedDict):
    """Dataset sample backed by NumPy arrays."""

    image: np.ndarray
    mask: np.ndarray


class TensorSample(TypedDict):
    """Dataset sample backed by PyTorch tensors."""

    image: torch.Tensor
    mask: torch.Tensor

class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class SynchrotronDataset(Dataset):
    """Dataset for loading synchrotron tomography data."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        subset_fraction: float = 1.0,
        random_seed: Optional[int] = None,
        maxworkers: int = 24,
        cache_size: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.maxworkers = maxworkers
        self.cache_size = cache_size
        self._cache: dict[int, ArraySample] = {}
        self._invalid_indices: set[int] = set()
        
        self.data_shape: Optional[tuple[int, ...]] = None
        self.label_shape: Optional[tuple[int, ...]] = None
        self.multi_class: Optional[bool] = None
        self.class_values: Optional[np.ndarray] = None

        try:
            self._setup_data_directories()
            if subset_fraction < 1.0:
                self._apply_subset_selection(subset_fraction, random_seed)
            self._initialize_dataset_metadata()
            if self.cache_size > 0:
                self._preload_cache()
        except Exception as e:
            raise DatasetError(f"Failed to initialize dataset: {str(e)}")

    def _setup_data_directories(self):
        """Set up and validate data directories and files."""
        data_path = self.data_dir / self.split / 'data'
        label_path = self.data_dir / self.split / 'labels'

        if not data_path.exists() or not label_path.exists():
            raise DatasetError(f"Data directory not found: {data_path} or {label_path}")

        data_files = sorted(list(data_path.glob('*.tiff')) + list(data_path.glob('*.tif')))
        label_files = sorted(list(label_path.glob('*.tiff')) + list(label_path.glob('*.tif')))

        if not data_files or not label_files:
            raise DatasetError(f"No .tif files found in {data_path} or {label_path}")
            
        if len(data_files) != len(label_files):
            raise DatasetError(
                f"Mismatched file count in {self.split} split: "
                f"{len(data_files)} images, {len(label_files)} masks"
            )

        self.data_files = data_files
        self.label_files = label_files
        self.n_samples = len(data_files)
        logger.info(f"Initialized {self.split} dataset with {self.n_samples} samples")

    def _initialize_dataset_metadata(self):
        """Load representative metadata for the currently selected files."""
        if self.n_samples == 0:
            raise DatasetError("Dataset is empty after subset selection")

        self.data_shape = None
        self.label_shape = None
        self.multi_class = None

        sample = self._load_from_file(0)
        self.data_shape = sample['image'].shape
        self.label_shape = sample['mask'].shape

        max_scan_files = min(50, self.n_samples)
        scan_indices = np.linspace(0, self.n_samples - 1, max_scan_files, dtype=int)
        
        all_unique_values = set()
        all_unique_values.update(np.unique(sample['mask']))

        for idx in scan_indices[1:]:
            try:
                label = np.ascontiguousarray(tifffile.imread(str(self.label_files[idx])))
                file_unique = np.unique(label)
                all_unique_values.update(file_unique)
            except Exception as e:
                logger.warning(f"Could not scan file {self.label_files[idx]} for class values: {e}")

        all_unique_values = np.array(sorted(all_unique_values))

        if self._has_multiclass_labels(all_unique_values):
            logger.info(f"Detected multi-class segmentation with {len(all_unique_values)} unique values across dataset: {all_unique_values}")
            self.multi_class = True
        else:
            self.multi_class = False
            logger.info("Detected binary segmentation masks")

        self.class_values = all_unique_values

        logger.info(f"Data shape: {self.data_shape}")
        logger.info(f"Label shape: {self.label_shape}")

    def _preload_cache(self):
        """Preload samples into cache."""
        n_to_cache = min(self.cache_size, self.n_samples)
        logger.info(f"Preloading {n_to_cache} samples into cache...")
        
        self._cache = {}
        for idx in range(n_to_cache):
            sample = self._load_from_file(idx)
            self._cache[idx] = sample

    @staticmethod
    def _has_multiclass_labels(unique_values: np.ndarray) -> bool:
        """Return whether a label value set represents multi-class segmentation."""
        return len(unique_values) > 2 or (
            len(unique_values) == 2 and not np.array_equal(unique_values, np.array([0, 1]))
        )
            
    def _load_from_file(self, idx: int) -> ArraySample:
        """Load a sample from the TIFF file."""
        data = np.ascontiguousarray(tifffile.imread(str(self.data_files[idx]))).astype(np.float32)

        label_raw = np.ascontiguousarray(tifffile.imread(str(self.label_files[idx])))

        unique_values = np.unique(label_raw)
        if self._has_multiclass_labels(unique_values):
            label = label_raw.astype(np.int64)

            if self.multi_class is None:
                self.multi_class = True
                logger.info(f"Detected multi-class segmentation with {len(unique_values)} unique values: {unique_values}")
        else:
            label = label_raw.astype(np.float32)

            if self.multi_class is None:
                self.multi_class = False
                logger.info("Detected binary segmentation masks")

        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        if label.ndim == 2:
            label = np.expand_dims(label, 0)

        if self.data_shape is None:
            self.data_shape = data.shape
            self.label_shape = label.shape
        elif data.shape != self.data_shape:
            raise ValueError(f"Data shape mismatch at index {idx}. Expected {self.data_shape}, got {data.shape}")
        elif label.shape != self.label_shape:
            raise ValueError(f"Label shape mismatch at index {idx}. Expected {self.label_shape}, got {label.shape}")
        
        return {
            'image': data,
            'mask': label
        }
    
    def _apply_subset_selection(self, subset_fraction: float, random_seed: Optional[int]):
        """Apply subset selection to the dataset."""
        if random_seed is not None:
            np.random.seed(random_seed)
        num_samples = int(self.n_samples * subset_fraction)
        if num_samples <= 0:
            raise DatasetError("Subset selection resulted in an empty dataset")
        indices = np.random.choice(self.n_samples, num_samples, replace=False)
        self.data_files = [self.data_files[i] for i in indices]
        self.label_files = [self.label_files[i] for i in indices]
        self.n_samples = num_samples
        self._cache.clear()
        logger.info(f"Using {self.n_samples} samples ({subset_fraction*100:.1f}% of data)")

    def _get_real_index(self, idx: int) -> int:
        """Convert a dataset index to the backing file index."""
        if len(self._invalid_indices) == self.n_samples:
            raise IndexError("No valid items left in dataset")
        
        real_idx = idx
        for invalid_idx in sorted(self._invalid_indices):
            if invalid_idx <= real_idx:
                real_idx += 1
            else:
                break
        
        if real_idx >= self.n_samples:
            raise IndexError("Index out of range")
        
        return real_idx
    
    def _process_image(self, image: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """Process an image array into the required format."""
        if not is_mask and self.normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            image = image.astype(np.float32)
        elif is_mask:
            if self.multi_class:
                # Multi-class masks must stay integer-typed for CrossEntropyLoss.
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = np.round(image).astype(np.int64)
                elif image.dtype == np.bool_:
                    image = image.astype(np.int64)
                else:
                    image = image.astype(np.int64)
            else:
                image = image.astype(np.float32)
        else:
            image = image.astype(np.float32)
            
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        
        return torch.from_numpy(image)
    
    def _apply_transforms(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         """Apply transforms to an image and mask pair."""
         if not self.transform:
             return image, mask

         image_np = image.numpy()[0]
         mask_np = mask.numpy()[0]
         original_mask_dtype = mask_np.dtype
         transformed = self.transform(image=image_np, mask=mask_np)
         transformed_mask = transformed['mask'].astype(original_mask_dtype)
         image = torch.from_numpy(np.expand_dims(transformed['image'], axis=0))
         mask = torch.from_numpy(np.expand_dims(transformed_mask, axis=0))
         return image, mask
    
    def __len__(self) -> int:
        return self.n_samples - len(self._invalid_indices)

    def __getitem__(self, idx: int) -> TensorSample:
        """Get a single item from the dataset."""
        real_idx = self._get_real_index(idx)

        try:
            sample = self._cache.get(real_idx)
            
            if sample is None:
                sample = self._load_from_file(real_idx)

                if len(self._cache) < self.cache_size:
                    self._cache[real_idx] = {
                        'image': sample['image'].copy(),
                        'mask': sample['mask'].copy()
                    }
            else:
                sample = {
                    'image': sample['image'].copy(),
                    'mask': sample['mask'].copy()
                }

            image = self._process_image(sample['image'], is_mask=False)
            mask = self._process_image(sample['mask'], is_mask=True)
            transformed_image, transformed_mask = self._apply_transforms(image, mask)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                'image': transformed_image,
                'mask': transformed_mask
            }
        except Exception as exc:
            raise DatasetError(
                "Failed to load sample "
                f"{idx} (resolved index {real_idx}) from "
                f"{self.data_files[real_idx]} and {self.label_files[real_idx]}: {exc}"
            ) from exc
    
    def get_items_by_filenames(self, filenames: list[str]) -> list[int]:
        """Get dataset indices for given filenames."""
        raise NotImplementedError
        
    def get_all_filenames(self) -> list[str]:
        """Get all filenames in the dataset."""
        raise NotImplementedError
    
    def __del__(self):
        """Clean up when the dataset is deleted."""
        torch.cuda.empty_cache()