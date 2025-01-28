"""Command-line interface for running multi-axis prediction on tomographic volumes."""
import argparse
import logging
import torch
from pathlib import Path
import time
import h5py

from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.prediction.predictor import TomoPredictor, PredictionMode, EnsembleMethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_prediction_mode(mode_str):
    """Convert string input to PredictionMode enum."""
    mode_map = {
        'SINGLE_AXIS': PredictionMode.SINGLE_AXIS,
        'THREE_AXIS': PredictionMode.THREE_AXIS,
        'TWELVE_AXIS': PredictionMode.TWELVE_AXIS
    }
    return mode_map[mode_str.upper()]

def process_file(input_path, output_path, predictor, dataset_path):
    """Process a single H5 file."""
    logger.info(f"Processing {input_path}")
    start_time = time.time()
    
    predictor.predict_volume(
        input_path=str(input_path),
        output_path=str(output_path),
        dataset_path=dataset_path
    )
    
    end_time = time.time()
    logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run predictions on H5 files using trained model.")
    parser.add_argument("input", type=str, help="Path to H5 file or directory containing H5 files")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="/data",
                       help="Path within H5 file where data is stored (default: /data)")
    parser.add_argument("--mode", type=str, choices=['SINGLE_AXIS', 'THREE_AXIS', 'TWELVE_AXIS'],
                       default='SINGLE_AXIS', help="Prediction mode")
    parser.add_argument("--output_dir", type=str, default="predictions",
                       help="Output directory for predictions (default: predictions)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model configuration
    model_config = {
        "encoder_name": "nvidia/segformer-b4-finetuned-ade-512-512",
        "num_classes": 1
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info("Loading model...")
    model = CustomSegformer(**model_config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Try different ways to load state dict
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        logger.info("Failed to load state dict directly, trying without 'model.' prefix")
        new_state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        try:
            model.load_state_dict(new_state_dict)
            logger.info("Successfully loaded state dict after removing prefix")
        except Exception as e:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Successfully loaded state dict with strict=False")
    
    # Create predictor
    predictor = TomoPredictor(
        model=model,
        prediction_mode=get_prediction_mode(args.mode),
        ensemble_method=EnsembleMethod.MEAN,
        batch_size=args.batch_size,
        device=args.device
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        if not input_path.suffix == '.h5':
            raise ValueError(f"Input file must be an H5 file, got {input_path}")
        output_path = output_dir / f"{input_path.stem}_prediction.h5"
        process_file(input_path, output_path, predictor, args.data_path)
    else:
        # Process directory
        h5_files = list(input_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No H5 files found in directory {input_path}")
        
        for h5_file in h5_files:
            output_path = output_dir / f"{h5_file.stem}_prediction.h5"
            process_file(h5_file, output_path, predictor, args.data_path)

if __name__ == "__main__":
    main()
