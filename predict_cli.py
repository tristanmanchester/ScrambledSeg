"""Command-line interface for ScrambledSeg inference workflows."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.prediction.predictor import Predictor, PredictionMode


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def get_prediction_mode(mode_str: str) -> PredictionMode:
    """Convert a string value to the corresponding :class:`PredictionMode`."""

    mode_map = {
        "SINGLE_AXIS": PredictionMode.SINGLE_AXIS,
        "THREE_AXIS": PredictionMode.THREE_AXIS,
        "TWELVE_AXIS": PredictionMode.TWELVE_AXIS,
    }

    try:
        return mode_map[mode_str.upper()]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(
            f"Unsupported prediction mode '{mode_str}'. Choose from {list(mode_map)}."
        ) from exc


def process_file(
    input_path: Path,
    output_path: Path,
    predictor: Predictor,
    dataset_path: Optional[str] = None,
) -> None:
    """Run inference on a single file and persist the output."""

    logger.info("Processing %s", input_path)
    start_time = time.time()

    predictor.predict(
        input_path=str(input_path),
        output_path=str(output_path),
        dataset_path=dataset_path,
    )

    elapsed = time.time() - start_time
    logger.info("Prediction completed in %.2f seconds", elapsed)
    logger.info("Results saved to %s", output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run predictions on H5 volumes or TIFF images using a trained model."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input file or directory (supports .h5, .tif, .tiff)",
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")

    # H5-specific arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data",
        help="Path within H5 file where data is stored (default: /data)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["SINGLE_AXIS", "THREE_AXIS", "TWELVE_AXIS"],
        default="SINGLE_AXIS",
        help="Prediction mode for H5 volumes",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        choices=["mean", "voting"],
        default="mean",
        help="Method for combining predictions",
    )

    # TIFF-specific arguments
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Size of tiles for processing large images (default: 512)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="Overlap between tiles (default: 32)",
    )

    # General arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Output directory for predictions (default: predictions)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for prediction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "16", "bf16"],
        default="bf16",
        help="Precision to use for inference (default: bf16)",
    )

    return parser


def load_model(checkpoint_path: Path, device: str) -> CustomSegformer:
    """Load a ``CustomSegformer`` model from a checkpoint."""

    model_config = {
        "encoder_name": "nvidia/segformer-b4-finetuned-ade-512-512",
        "num_classes": 4,  # Multi-phase segmentation with 4 classes (0, 1, 2, 3)
    }

    logger.info("Loading model...")
    model = CustomSegformer(**model_config)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'state_dict' entry.")

    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        logger.info("Falling back to loading state dict without 'model.' prefix")
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(cleaned_state_dict)
        except RuntimeError:
            logger.warning("Loading with strict=False due to mismatched keys.")
            model.load_state_dict(cleaned_state_dict, strict=False)

    return model


def main() -> None:
    """Entry point for the prediction command-line interface."""

    parser = build_argument_parser()
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device)

    predictor = Predictor(
        model=model,
        prediction_mode=get_prediction_mode(args.mode),
        ensemble_method=args.ensemble_method,
        batch_size=args.batch_size,
        device=device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        precision=args.precision,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    input_path = Path(args.input)
    if input_path.is_file():
        if input_path.suffix.lower() not in [".h5", ".tif", ".tiff"]:
            raise ValueError(f"Input file must be H5 or TIFF, got {input_path}")
        output_path = output_dir / f"{input_path.stem}_prediction{input_path.suffix}"
        process_file(input_path, output_path, predictor, args.data_path)
    else:
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
