"""Command-line interface for running predictions on H5 volumes and TIFF images."""
import argparse
import logging
import torch
from pathlib import Path
import time

from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.prediction.predictor import Predictor, PredictionMode, EnsembleMethod

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

def process_file(input_path, output_path, predictor, dataset_path=None):
    """Process a single file."""
    logger.info(f"Processing {input_path}")
    start_time = time.time()
    
    predictor.predict(
        input_path=str(input_path),
        output_path=str(output_path),
        dataset_path=dataset_path
    )
    
    end_time = time.time()
    logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run predictions on H5 volumes or TIFF images using trained model."
    )
    parser.add_argument("input", type=str, 
                       help="Path to input file or directory (supports .h5, .tif, .tiff)")
    parser.add_argument("checkpoint", type=str, 
                       help="Path to model checkpoint")
    
    # H5-specific arguments
    parser.add_argument("--data_path", type=str, default="/data",
                       help="Path within H5 file where data is stored (default: /data)")
    parser.add_argument("--mode", type=str, choices=['SINGLE_AXIS', 'THREE_AXIS', 'TWELVE_AXIS'],
                       default='SINGLE_AXIS', help="Prediction mode for H5 volumes")
    parser.add_argument("--ensemble_method", type=str, choices=['mean', 'voting'],
                       default='mean', help="Method for combining predictions")
    
    # TIFF-specific arguments
    parser.add_argument("--tile_size", type=int, default=512,
                       help="Size of tiles for processing large images (default: 512)")
    parser.add_argument("--tile_overlap", type=int, default=64,
                       help="Overlap between tiles (default: 64)")
    
    # General arguments
    parser.add_argument("--output_dir", type=str, default="predictions",
                       help="Output directory for predictions (default: predictions)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: cuda if available, else cpu)")
    parser.add_argument("--precision", type=str, choices=['32', '16', 'bf16'], default='bf16',
                       help="Precision to use for inference (default: bf16)")
    parser.add_argument("--encoder_name", type=str,
                       default="nvidia/segformer-b4-finetuned-ade-512-512",
                       help="Hugging Face model identifier for the SegFormer backbone")
    parser.add_argument("--num_classes", type=int, default=4,
                       help="Number of segmentation classes, including background")
    parser.add_argument("--in_channels", type=int, default=1,
                       help="Number of input channels expected by the model")
    parser.add_argument("--cache_dir", type=str, default=".model_cache",
                       help="Directory for caching downloaded model weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                       help="Initialize the model from config without downloading pretrained weights")
    parser.set_defaults(pretrained=True)
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model configuration
    model_config = {
        "encoder_name": args.encoder_name,
        "num_classes": args.num_classes,
        "pretrained": args.pretrained,
        "in_channels": args.in_channels,
        "cache_dir": args.cache_dir,
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    logger.info(
        "Loading model '%s' with %d class(es) and %d input channel(s)",
        args.encoder_name,
        args.num_classes,
        args.in_channels,
    )
    model = CustomSegformer(**model_config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Try different ways to load state dict
    try:
        model.load_state_dict(state_dict)
    except Exception:
        logger.info("Failed to load state dict directly, trying without 'model.' prefix")
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
            logger.info("Successfully loaded state dict after removing prefix")
        except Exception as error:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info("Loaded state dict with strict=False due to: %s", error)
    
    # Create predictor
    predictor = Predictor(
        model=model,
        prediction_mode=get_prediction_mode(args.mode),
        ensemble_method=args.ensemble_method,
        batch_size=args.batch_size,
        device=args.device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        precision=args.precision
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        if input_path.suffix.lower() not in ['.h5', '.tif', '.tiff']:
            raise ValueError(f"Input file must be H5 or TIFF, got {input_path}")
        output_path = output_dir / f"{input_path.stem}_prediction{input_path.suffix}"
        process_file(input_path, output_path, predictor, args.data_path)
    else:
        # Process directory
        h5_files = list(input_path.glob("*.h5"))
        tiff_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
        all_files = h5_files + tiff_files
        
        if not all_files:
            raise ValueError(f"No H5 or TIFF files found in directory {input_path}")
        
        for file_path in all_files:
            output_path = output_dir / f"{file_path.stem}_prediction{file_path.suffix}"
            process_file(file_path, output_path, predictor, args.data_path)

if __name__ == "__main__":
    main()
