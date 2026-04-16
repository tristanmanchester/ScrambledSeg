"""Command-line interface for ScrambledSeg inference workflows."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from ..models.segformer import CustomSegformer
from .data import DEFAULT_DATASET_PATH
from .errors import ModelLoadError, PredictionInputError
from .predictor import PredictionMode, Predictor
from .tiff_utils import (
    DEFAULT_TILE_OVERLAP,
    TiffInputKind,
    TiffOutputKind,
)

logger = logging.getLogger(__name__)
PREDICTION_MODE_CHOICES = [mode.value for mode in PredictionMode]
SUPPORTED_INPUT_SUFFIXES = {".h5", ".tif", ".tiff"}
TIFF_INPUT_KIND_CHOICES = [kind.value for kind in TiffInputKind]
TIFF_OUTPUT_KIND_CHOICES = [kind.value for kind in TiffOutputKind]
REQUIRED_CHECKPOINT_MODEL_FIELDS = ("encoder_name", "num_classes", "pretrained")


def configure_logging(level: str) -> None:
    """Update the global logging level from CLI input."""

    try:
        numeric_level = getattr(logging, level.upper())
    except AttributeError as exc:  # pragma: no cover - guarding against invalid input
        raise ValueError(f"Unsupported log level '{level}'.") from exc

    logging.basicConfig(
        level=numeric_level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


def process_h5_file(
    input_path: Path,
    output_path: Path,
    predictor: Predictor,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> None:
    """Run inference on a single H5 volume and persist the output."""

    logger.info("Processing %s", input_path)
    start_time = time.time()
    predictor.predict_volume(
        input_path=input_path,
        output_path=output_path,
        dataset_path=dataset_path,
    )

    elapsed = time.time() - start_time
    logger.info("Prediction completed in %.2f seconds", elapsed)
    logger.info("Results saved to %s", output_path)


def process_tiff_file(
    input_path: Path,
    output_path: Path,
    predictor: Predictor,
    tiff_input_kind: str = TiffInputKind.AUTO.value,
    tiff_output_kind: str = TiffOutputKind.LABELS.value,
) -> None:
    """Run inference on a single TIFF input and persist the output."""

    logger.info("Processing %s", input_path)
    start_time = time.time()

    predictor.predict_tiff(
        input_path=input_path,
        output_path=output_path,
        input_kind=TiffInputKind.parse(tiff_input_kind),
        output_kind=TiffOutputKind.parse(tiff_output_kind),
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
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path within H5 file where data is stored (default: /data)",
    )
    parser.add_argument(
        "--data_path",
        dest="dataset_path",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=PREDICTION_MODE_CHOICES,
        default=PredictionMode.SINGLE_AXIS.value,
        help="Prediction mode for H5 volumes",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Size of tiles for processing large images (default: 512)",
    )
    parser.add_argument(
        "--tile_size",
        dest="tile_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help=f"Overlap between tiles (default: {DEFAULT_TILE_OVERLAP})",
    )
    parser.add_argument(
        "--tile_overlap",
        dest="tile_overlap",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory for predictions (default: predictions)",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for prediction")
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--tiff-input-kind",
        type=str,
        choices=TIFF_INPUT_KIND_CHOICES,
        default=TiffInputKind.AUTO.value,
        help=(
            "How to interpret TIFF inputs: 'image' for a single grayscale or channel-last "
            "image, 'stack' for a multi-slice volume, or 'auto' to infer when unambiguous."
        ),
    )
    parser.add_argument(
        "--tiff-output-kind",
        type=str,
        choices=TIFF_OUTPUT_KIND_CHOICES,
        default=TiffOutputKind.LABELS.value,
        help="Whether TIFF predictions should be saved as labels or probabilities.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level to use (e.g. INFO, DEBUG)",
    )
    return parser


def _extract_model_config(checkpoint: dict[str, object]) -> dict[str, object]:
    """Validate and return the serialized model config from a checkpoint."""

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    model_config = (
        hyper_parameters.get("model_config", {})
        if isinstance(hyper_parameters, dict)
        else {}
    )
    if not isinstance(model_config, dict):
        model_config = {}

    missing_fields = [field for field in REQUIRED_CHECKPOINT_MODEL_FIELDS if field not in model_config]
    if missing_fields:
        raise ModelLoadError(
            "Checkpoint is missing hyper_parameters.model_config fields "
            f"{missing_fields}. Required fields: {list(REQUIRED_CHECKPOINT_MODEL_FIELDS)} "
            "with optional encoder_revision."
        )

    return {
        "encoder_name": model_config["encoder_name"],
        "encoder_revision": model_config.get("encoder_revision"),
        "num_classes": model_config["num_classes"],
        "pretrained": model_config["pretrained"],
    }


def load_model(checkpoint_path: Path, device: str) -> CustomSegformer:
    """Load a ``CustomSegformer`` model from a checkpoint."""

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "state_dict" not in checkpoint:
        raise ModelLoadError("Checkpoint does not contain a 'state_dict' entry.")

    model_config = _extract_model_config(checkpoint)

    logger.info("Loading model...")
    model = CustomSegformer(**model_config)

    state_dict = checkpoint["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        logger.info("Falling back to loading state dict without 'model.' prefix")
        cleaned_state_dict = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
        try:
            model.load_state_dict(cleaned_state_dict)
        except RuntimeError as exc:
            expected_keys = set(model.state_dict().keys())
            provided_keys = set(cleaned_state_dict.keys())
            missing_keys = sorted(expected_keys - provided_keys)
            unexpected_keys = sorted(provided_keys - expected_keys)

            def _preview(keys: list[str]) -> str:
                preview = keys[:5]
                suffix = "..." if len(keys) > 5 else ""
                return f"{preview}{suffix}"

            raise ModelLoadError(
                "Checkpoint state_dict does not match the model after removing an optional "
                f"'model.' prefix. Missing keys: {_preview(missing_keys)}; "
                f"unexpected keys: {_preview(unexpected_keys)}."
            ) from exc

    return model


def main() -> None:
    """Entry point for the prediction command-line interface."""

    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device)

    predictor = Predictor(
        model=model,
        prediction_mode=args.mode,
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
        if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise PredictionInputError(f"Input file must be H5 or TIFF, got {input_path}")
        output_path = output_dir / f"{input_path.stem}_prediction{input_path.suffix}"
        if input_path.suffix.lower() == ".h5":
            process_h5_file(
                input_path,
                output_path,
                predictor,
                args.dataset_path,
            )
        else:
            process_tiff_file(
                input_path,
                output_path,
                predictor,
                args.tiff_input_kind,
                args.tiff_output_kind,
            )
        return

    h5_files = list(input_path.glob("*.h5"))
    tiff_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    all_files = h5_files + tiff_files

    if not all_files:
        raise PredictionInputError(f"No H5 or TIFF files found in directory {input_path}")

    for file_path in all_files:
        output_path = output_dir / f"{file_path.stem}_prediction{file_path.suffix}"
        if file_path.suffix.lower() == ".h5":
            process_h5_file(
                file_path,
                output_path,
                predictor,
                args.dataset_path,
            )
        else:
            process_tiff_file(
                file_path,
                output_path,
                predictor,
                args.tiff_input_kind,
                args.tiff_output_kind,
            )


if __name__ == "__main__":
    main()
