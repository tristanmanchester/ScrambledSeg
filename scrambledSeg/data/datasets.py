import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from multiprocessing import Lock
import albumentations as A
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetError(Exception):
    """Base exception for dataset-related errors"""
    pass

class SynchrotronDataset(Dataset):
    """Dataset for loading synchrotron tomography data"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        subset_fraction: float = 1.0,
        random_seed: int = None,
        maxworkers: int = 24,
        cache_size: int = 0
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.maxworkers = maxworkers
        self.cache_size = cache_size
        self._invalid_indices = set()
        
        # Add this default initialization
        self.data_shape = None  
        self.label_shape = None

        try:
            self._setup_data_directories()
            if subset_fraction < 1.0:
                self._apply_subset_selection(subset_fraction, random_seed)
        except Exception as e:
            raise DatasetError(f"Failed to initialize dataset: {str(e)}")

    def _setup_data_directories(self):
        """Set up and validate data directories and files"""
        data_path = self.data_dir / self.split / 'data'
        label_path = self.data_dir / self.split / 'labels'
        
        # Validate directories exist
        if not data_path.exists() or not label_path.exists():
            raise DatasetError(f"Data directory not found: {data_path} or {label_path}")
        
        # Get and sort files
        data_files = sorted(list(data_path.glob('*.tiff')) + list(data_path.glob('*.tif')))
        label_files = sorted(list(label_path.glob('*.tiff')) + list(label_path.glob('*.tif')))
        
        # Verify matching
        if not data_files or not label_files:
            raise DatasetError(f"No .tif files found in {data_path} or {label_path}")
            
        if len(data_files) != len(label_files):
            raise DatasetError(
                f"Mismatched file count in {self.split} split: "
                f"{len(data_files)} images, {len(label_files)} masks"
            )
            
        # Store files and number of samples
        self.data_files = data_files
        self.label_files = label_files
        self.n_samples = len(data_files)
        logger.info(f"Initialized {self.split} dataset with {self.n_samples} samples")
        
        # Load sample and set shapes first
        sample = self._load_from_file(0)
        self.data_shape = sample['image'].shape
        self.label_shape = sample['mask'].shape
        
        # Log dataset information
        logger.info(f"Data shape: {self.data_shape}")
        logger.info(f"Label shape: {self.label_shape}")
        
        # Preload into cache if requested
        if self.cache_size > 0:
            self._preload_cache()

    def _preload_cache(self):
        """Preload samples into cache."""
        n_to_cache = min(self.cache_size, self.n_samples)
        logger.info(f"Preloading {n_to_cache} samples into cache...")
        
        self._cache = {}
        for idx in range(n_to_cache):
            sample = self._load_from_file(idx)
            self._cache[idx] = sample
            
    def _load_from_file(self, idx):
        """Load a sample from the TIFF file."""
        # Load data and labels, ensuring contiguous memory layout
        data = np.ascontiguousarray(tifffile.imread(str(self.data_files[idx]))).astype(np.float32)
        label = np.ascontiguousarray(tifffile.imread(str(self.label_files[idx]))).astype(np.float32)

        # Ensure correct shapes
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
        if label.ndim == 2:
            label = np.expand_dims(label, 0)
        
        # For the first sample, set the expected shapes
        if self.data_shape is None:
            self.data_shape = data.shape
            self.label_shape = label.shape
        # For subsequent samples, ensure shapes match
        elif data.shape != self.data_shape:
            raise ValueError(f"Data shape mismatch at index {idx}. Expected {self.data_shape}, got {data.shape}")
        elif label.shape != self.label_shape:
            raise ValueError(f"Label shape mismatch at index {idx}. Expected {self.label_shape}, got {label.shape}")
        
        return {
            'image': data,
            'mask': label
        }
    
    def _apply_subset_selection(self, subset_fraction: float, random_seed: Optional[int]):
        """Apply subset selection to the dataset"""
        if random_seed is not None:
            np.random.seed(random_seed)
        num_samples = int(self.n_samples * subset_fraction)
        indices = np.random.choice(self.n_samples, num_samples, replace=False)
        self.data_files = [self.data_files[i] for i in indices]
        self.label_files = [self.label_files[i] for i in indices]
        self.n_samples = num_samples
        logger.info(f"Using {self.n_samples} samples ({subset_fraction*100:.1f}% of data)")

    def _get_real_index(self, idx: int) -> int:
        """Convert dataset index to real index accounting for invalid indices"""
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
        """Process an image array into the required format"""
        if not is_mask and self.normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            
        image = image.astype(np.float32)
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        
        return torch.from_numpy(image)
    
    def _apply_transforms(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         """Apply transforms to image and mask pair"""
         if not self.transform:
             return image, mask
         
         # Convert tensors to numpy arrays
         image_np = image.numpy()[0] # Remove the extra dimension
         mask_np = mask.numpy()[0]

         transformed = self.transform(image=image_np, mask=mask_np)
         
         image = torch.from_numpy(np.expand_dims(transformed['image'], axis=0))
         mask = torch.from_numpy(np.expand_dims(transformed['mask'], axis=0))
         return image, mask
    
    def __len__(self) -> int:
        return self.n_samples - len(self._invalid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        try:
            real_idx = self._get_real_index(idx)
            
            # Check cache first
            sample = self._cache.get(real_idx)
            
            if sample is None:
                # Load from file with timeout protection
                sample = self._load_from_file(real_idx)
                    
                # Add to cache if space available
                if len(self._cache) < self.cache_size:
                    self._cache[real_idx] = {
                        'image': sample['image'].copy(),
                        'mask': sample['mask'].copy()
                    }
            else:
                # Make a copy to ensure we don't modify cached data
                sample = {
                    'image': sample['image'].copy(),
                    'mask': sample['mask'].copy()
                }
            
            # Process and return
            image = self._process_image(sample['image'], is_mask=False)
            mask = self._process_image(sample['mask'], is_mask=True)
            transformed_image, transformed_mask = self._apply_transforms(image, mask)
            
            # Add memory checks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                'image': transformed_image,
                'mask': transformed_mask
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            self._invalid_indices.add(real_idx)
            # Return a new index if possible
            if len(self._invalid_indices) < self.n_samples - 1:
                return self.__getitem__((idx + 1) % self.n_samples)
            raise RuntimeError("No valid samples remaining in dataset")
    
    def get_items_by_filenames(self, filenames: List[str]) -> List[int]:
        """Get dataset indices for given filenames"""
        # Not applicable for this dataset
        raise NotImplementedError
        
    def get_all_filenames(self) -> List[str]:
        """Get all filenames in the dataset"""
        # Not applicable for this dataset
        raise NotImplementedError
    
    def __del__(self):
        """Cleanup when dataset is deleted"""
        torch.cuda.empty_cache()