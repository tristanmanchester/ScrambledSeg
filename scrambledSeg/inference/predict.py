"""Prediction module for segmentation models."""
import torch
import numpy as np
import h5py
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from ..models.segformer import CustomSegformer
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PredictionAxis(str, Enum):
    """Prediction axes for segmentation."""
    X = 'x'  # Predict along X axis (view YZ planes)
    Y = 'y'  # Predict along Y axis (view XZ planes)
    Z = 'z'  # Predict along Z axis (view XY planes)

class SingleAxisPredictor:
    """Predictor class for making predictions along a single axis."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_variant: str = "b1",  # Changed default to b1
        batch_size: int = 8,
        device: str = None,
        threshold_params: Optional[Dict[str, Any]] = None,
        prediction_axis: str = 'z'
    ):
        """Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_variant: SegFormer model variant ('b0', 'b1', 'b2', 'b3', 'b4', 'b5')
            batch_size: Batch size for predictions
            device: Device to run predictions on
            threshold_params: Parameters for thresholding and cleanup
            prediction_axis: Axis along which to make predictions ('x', 'y', or 'z')
        """
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.prediction_axis = PredictionAxis(prediction_axis.lower())
        self.model_variant = model_variant.lower()
        
        # Default threshold parameters matching trainer
        self.threshold_params = threshold_params or {
            'threshold': 0.5,
            'enable_cleanup': True,
            'cleanup_kernel_size': 3,
            'cleanup_threshold': 5,
            'min_hole_size_factor': 64
        }
        
        # Load and prepare model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        logger.info(f"Initialized predictor:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model variant: {self.model_variant}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Prediction axis: {self.prediction_axis}")
        logger.info(f"  Threshold: {self.threshold_params['threshold']}")

    def _normalize_tensor(self, tensor: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize tensor/array to [0, 1] range."""
        is_tensor = torch.is_tensor(tensor)
        
        if is_tensor:
            tensor = tensor.detach().cpu()
        
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]
        
        if is_tensor:
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
        else:
            min_val = np.min(tensor)
            max_val = np.max(tensor)
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
                
        return tensor

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint

            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # Initialize model with specified variant
            encoder_name = f"nvidia/mit-{self.model_variant}"
            logger.info(f"Initializing model with encoder: {encoder_name}")
            
            model = CustomSegformer(encoder_name=encoder_name)
            model.load_state_dict(state_dict)
            return model.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")

    def _get_binary_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Get binary predictions using thresholding and cleanup.
        
        Args:
            predictions: Model predictions after sigmoid (B, C, H, W)
            
        Returns:
            Binary predictions tensor (B, C, H, W)
        """
        # Apply threshold
        threshold = self.threshold_params.get('threshold', 0.5)
        binary_pred = (predictions > threshold).float()
        
        # Apply morphological cleanup if enabled
        if self.threshold_params.get('enable_cleanup', True):
            kernel_size = self.threshold_params.get('cleanup_kernel_size', 3)
            cleanup_threshold = self.threshold_params.get('cleanup_threshold', 5)
            
            # Ensure kernel size is odd
            kernel_size = max(3, kernel_size + (kernel_size + 1) % 2)
            pad_size = kernel_size // 2
            
            # Create cleanup kernel
            cleanup_kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                                      device=predictions.device)
            
            # Process each slice in the batch separately
            batch_size = binary_pred.size(0)
            cleaned_preds = []
            
            for i in range(batch_size):
                # Get single prediction (1, C, H, W)
                single_pred = binary_pred[i:i+1]
                
                # Apply morphological operation
                padded_binary = F.pad(single_pred, 
                                    (pad_size, pad_size, pad_size, pad_size),
                                    mode='constant', 
                                    value=0)
                connected = F.conv2d(padded_binary, cleanup_kernel, padding=0)
                
                # Calculate threshold based on kernel size
                auto_threshold = (kernel_size * kernel_size) / 2
                final_threshold = min(cleanup_threshold, auto_threshold)
                
                # Apply cleanup
                cleaned = torch.where(connected >= final_threshold, 
                                    single_pred, 
                                    torch.zeros_like(single_pred))
                cleaned_preds.append(cleaned)
            
            # Stack back into batch
            binary_pred = torch.cat(cleaned_preds, dim=0)
        
        return binary_pred

    def _transform_volume(self, volume: np.ndarray) -> np.ndarray:
        """Transform volume based on prediction axis."""
        if self.prediction_axis == PredictionAxis.X:
            return np.transpose(volume, (2, 1, 0))  # XY->ZY view
        elif self.prediction_axis == PredictionAxis.Y:
            return np.transpose(volume, (1, 0, 2))  # XY->XZ view
        else:  # Z axis (default)
            return volume  # Keep XY view

    def _inverse_transform_volume(self, volume: np.ndarray) -> np.ndarray:
        """Inverse transform volume back to original orientation."""
        if self.prediction_axis == PredictionAxis.X:
            return np.transpose(volume, (2, 1, 0))
        elif self.prediction_axis == PredictionAxis.Y:
            return np.transpose(volume, (1, 0, 2))
        else:  # Z axis (default)
            return volume

    def predict(
        self, 
        input_path: str, 
        output_path: str, 
        h5_dataset_path: str = 'data'
    ) -> None:
        """Predict segmentation for a volume."""
        # Load volume
        with h5py.File(input_path, 'r') as f:
            volume = f[h5_dataset_path][:]
        
        logger.info(f"Loaded volume of shape {volume.shape} from {input_path}")
        
        # Transform volume for prediction axis
        volume = self._transform_volume(volume)
        
        # Normalize exactly as in training
        volume = volume.astype(np.float32) / 65535.0
        
        # Get dimensions
        D, H, W = volume.shape
        logger.info(f"Processing {D} slices of size {H}x{W}")
        
        # Initialize output volumes
        logits_volume = np.zeros_like(volume, dtype=np.float32)
        probs_volume = np.zeros_like(volume, dtype=np.float32)
        binary_volume = np.zeros_like(volume, dtype=np.float32)
        
        # Process slices in batches
        n_slices = D
        n_batches = (n_slices + self.batch_size - 1) // self.batch_size
        
        for start_idx in tqdm(range(0, n_slices, self.batch_size), total=n_batches, desc="Predicting"):
            end_idx = min(start_idx + self.batch_size, n_slices)
            batch_slices = []
            
            # Prepare batch
            for idx in range(start_idx, end_idx):
                slice_data = volume[idx]
                slice_tensor = torch.from_numpy(
                    np.expand_dims(slice_data, 0)  # Add channel dimension
                ).float()
                batch_slices.append(slice_tensor)
                
            # Stack batch
            batch = torch.stack(batch_slices).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Get probabilities with same normalization as training
                probs = F.sigmoid(logits)
                
                # Convert to numpy while keeping gradients for normalization
                logits_np = logits.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                
                # Apply normalization as in training visualization
                normalized_logits = np.array([self._normalize_tensor(slice_logits) 
                                           for slice_logits in logits_np])
                normalized_probs = np.array([self._normalize_tensor(slice_probs) 
                                          for slice_probs in probs_np])
                
                # Get binary predictions after normalization
                binary = self._get_binary_predictions(torch.from_numpy(normalized_probs).to(self.device))
                binary = binary.cpu().numpy()
                
                # Store results - keeping all dimensions correct
                logits_volume[start_idx:end_idx] = normalized_logits.reshape(-1, H, W)
                probs_volume[start_idx:end_idx] = normalized_probs.reshape(-1, H, W)
                binary_volume[start_idx:end_idx] = binary.reshape(-1, H, W)
        
        # Transform volumes back to original orientation
        logits_volume = self._inverse_transform_volume(logits_volume)
        probs_volume = self._inverse_transform_volume(probs_volume)
        binary_volume = self._inverse_transform_volume(binary_volume)
        
        # Save results
        logger.info(f"Saving predictions to {output_path}")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('logits', data=logits_volume, compression='gzip')
            f.create_dataset('probabilities', data=probs_volume, compression='gzip')
            f.create_dataset('binary', data=binary_volume, compression='gzip')
        
        logger.info("Prediction completed successfully")

def predict_volume(
    checkpoint_path: str, 
    input_path: str, 
    output_dir: str,
    h5_dataset_path: str = 'data',
    batch_size: int = 8,
    show_progress: bool = True,
    prediction_axis: str = 'z',
    threshold_params: Optional[Dict[str, Any]] = None,
    model_variant: str = 'b1'  # Changed default to b1
) -> None:
    """Predict segmentation for a volume."""
    # Initialize predictor
    predictor = SingleAxisPredictor(
        checkpoint_path=checkpoint_path,
        model_variant=model_variant,
        batch_size=batch_size,
        threshold_params=threshold_params,
        prediction_axis=prediction_axis
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path
    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_prediction.h5")
    
    # Run prediction
    predictor.predict(
        input_path=input_path, 
        output_path=output_path, 
        h5_dataset_path=h5_dataset_path
    )
    
    if show_progress:
        print(f"Prediction saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict segmentation for a volume")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input H5 file")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save predictions")
    parser.add_argument("--dataset-path", type=str, default="data",
                      help="Path to dataset within H5 file")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for predictions")
    parser.add_argument("--axis", type=str, default="z",
                      help="Prediction axis (x, y, or z)")
    parser.add_argument("--model-variant", type=str, default="b1",
                      help="SegFormer model variant (b0, b1, b2, b3, b4, b5)")
    
    args = parser.parse_args()
    
    predict_volume(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_dir=args.output_dir,
        h5_dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        prediction_axis=args.axis,
        model_variant=args.model_variant
    )