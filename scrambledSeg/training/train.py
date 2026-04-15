"""Training script for SegFormer model."""

import argparse
import logging
import os
import warnings
from typing import TypedDict

import pytorch_lightning as pl
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch.utils.data import DataLoader

from scrambledSeg.data.datasets import SynchrotronDataset
from scrambledSeg.losses import create_loss
from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.training.rich_progress import RichProgressCallback
from scrambledSeg.training.trainer import PredictionParams, SegformerTrainer, VisualizationConfig
from scrambledSeg.transforms import create_train_transform, create_val_transform

warnings.filterwarnings("ignore", message="A new version of Albumentations is available")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tensorboardX.*")
warnings.filterwarnings("ignore", message=".*LOCAL_RANK.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*")
warnings.filterwarnings("ignore", message=".*plain ModelCheckpoint.*")

console = Console()

logger = logging.getLogger(__name__)


class LoggingConfig(TypedDict, total=False):
    """Logging settings accepted by the training entrypoint."""

    level: str


class DataloaderConfig(TypedDict, total=False):
    """Subset of dataloader settings consumed by the training entrypoint."""

    num_workers: int
    persistent_workers: bool
    pin_memory: bool
    cache_size: int


class LossParams(TypedDict, total=False):
    """Loss kwargs supported by the training config."""

    ce_weight: float
    dice_weight: float
    smooth: float
    bce_weight: float


class LossConfig(TypedDict, total=False):
    """Loss configuration block used by the training entrypoint."""

    type: str
    params: LossParams


class OptimizerConfig(TypedDict, total=False):
    """Optimizer configuration block used by the training entrypoint."""

    type: str
    params: dict[str, float]


class SchedulerParams(TypedDict, total=False):
    """Scheduler kwargs supported by the default training config."""

    max_lr: float
    epochs: int
    steps_per_epoch: int
    pct_start: float
    div_factor: float
    final_div_factor: float


class SchedulerConfig(TypedDict, total=False):
    """Scheduler configuration block used by the training entrypoint."""

    type: str
    params: SchedulerParams


class RegularizationConfig(TypedDict, total=False):
    """Regularization settings read by the training entrypoint."""

    gradient_clip_val: float


class TestModeSettings(TypedDict, total=False):
    """Overrides applied when config enables test mode."""

    num_epochs: int
    batch_size: int
    data_fraction: float


class TrainingConfig(TypedDict, total=False):
    """Top-level training configuration consumed by this module."""

    processed_data_dir: str
    encoder_name: str
    pretrained: bool
    num_classes: int
    num_epochs: int
    batch_size: int
    data_fraction: float
    gradient_accumulation_steps: int
    log_dir: str
    test_mode: bool
    dataloader: DataloaderConfig
    loss: LossConfig
    prediction: PredictionParams
    thresholding: PredictionParams
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    regularization: RegularizationConfig
    test_mode_settings: TestModeSettings


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def setup_logging(config: LoggingConfig) -> None:
    """Set up logging configuration with Rich handler."""
    log_level = config.get("level", "INFO")

    logging.getLogger().handlers = []

    rich_handler = RichHandler(
        console=console, show_time=True, show_path=False, rich_tracebacks=True
    )

    logging.basicConfig(
        level=getattr(logging, log_level), format="%(name)s - %(message)s", handlers=[rich_handler]
    )


def create_datasets(config: TrainingConfig) -> dict[str, DataLoader]:
    """Create training and validation datasets."""
    with console.status("[bold green]Creating data transforms..."):
        train_transform = create_train_transform(config)
        val_transform = create_val_transform(config)

    dataloader_config = config.get("dataloader", {})
    batch_size = config.get("batch_size", 16)
    num_workers = dataloader_config.get("num_workers", 4)
    cache_size = dataloader_config.get("cache_size", 0)

    with console.status("[bold blue]Loading training dataset..."):
        train_dataset = SynchrotronDataset(
            data_dir=config["processed_data_dir"],
            split="train",
            transform=train_transform,
            normalize=True,
            cache_size=cache_size,
            subset_fraction=config.get("data_fraction", 1.0),
            random_seed=42,
        )

    with console.status("[bold blue]Loading validation dataset..."):
        val_dataset = SynchrotronDataset(
            data_dir=config["processed_data_dir"],
            split="val",
            transform=val_transform,
            normalize=True,
            cache_size=cache_size,
            subset_fraction=config.get("data_fraction", 1.0),
            random_seed=43,
        )

    data_loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=dataloader_config.get("pin_memory", True),
            persistent_workers=dataloader_config.get("persistent_workers", True),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=dataloader_config.get("pin_memory", True),
            persistent_workers=dataloader_config.get("persistent_workers", True),
        ),
    }

    return data_loaders


def create_trainer_module(
    config: TrainingConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> SegformerTrainer:
    """Create trainer module."""
    with console.status("[bold magenta]Initializing SegFormer model..."):
        model = CustomSegformer(
            encoder_name=config.get("encoder_name", "nvidia/mit-b0"),
            num_classes=config.get("num_classes", 1),
            pretrained=config.get("pretrained", True),
            ignore_mismatched_sizes=True,
        )

    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "crossentropy_dice")

    params = loss_config.get("params", {}).copy()

    if loss_type == "crossentropy_dice" and "ce_weight" not in params and "bce_weight" in params:
        params["ce_weight"] = params.pop("bce_weight")

    criterion = create_loss(loss_type, **params)

    # Preserve legacy thresholding keys if the newer prediction block is absent.
    source_config = config.get("prediction", config.get("thresholding", {}))
    prediction_config = {
        "enable_cleanup": source_config.get("enable_cleanup", True),
        "cleanup_kernel_size": source_config.get("cleanup_kernel_size", 3),
        "cleanup_threshold": source_config.get("cleanup_threshold", 5),
        "min_hole_size_factor": source_config.get("min_hole_size_factor", 64),
    }

    optimizer_config = config.get("optimizer", {})
    optimizer_cls = getattr(torch.optim, optimizer_config.get("type", "AdamW"))
    optimizer = optimizer_cls(model.parameters(), **optimizer_config.get("params", {"lr": 0.001}))

    scheduler_config = config.get("scheduler", {})
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config.get("type", "OneCycleLR"))
    scheduler_params = scheduler_config.get("params", {}).copy()

    if scheduler_params.get("steps_per_epoch", -1) == -1:
        scheduler_params["steps_per_epoch"] = len(train_dataloader)

    scheduler = scheduler_cls(optimizer, **scheduler_params)

    trainer_module = SegformerTrainer(
        model=model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["num_epochs"],
        visualization=config.get("visualization", {}),
        log_dir=config.get("log_dir", "logs"),
        test_mode=config.get("test_mode", False),
        threshold_params=prediction_config,
        num_classes=config["num_classes"],
    )

    return trainer_module


def display_config_info(config: TrainingConfig, config_path: str) -> None:
    """Display training configuration in a nice table."""
    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Config File", str(config_path))
    table.add_row("Encoder", config.get("encoder_name", "nvidia/mit-b0"))
    table.add_row("Num Classes", str(config.get("num_classes", 1)))
    table.add_row("Batch Size", str(config.get("batch_size", 16)))
    table.add_row("Num Epochs", str(config.get("num_epochs", 100)))
    table.add_row(
        "Learning Rate", str(config.get("optimizer", {}).get("params", {}).get("lr", 0.001))
    )
    table.add_row("Data Directory", config.get("processed_data_dir", "N/A"))
    table.add_row("Test Mode", str(config.get("test_mode", False)))

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device += f" ({torch.cuda.get_device_name(0)})"
    table.add_row("Device", device)

    console.print("\n")
    console.print(table)
    console.print("\n")


def main(config_path: str) -> None:
    """Main training function."""
    console.print(
        Panel.fit(
            Text("ScrambledSeg Training", style="bold magenta"),
            subtitle="SegFormer for Synchrotron X-ray Tomography",
        )
    )

    torch.set_float32_matmul_precision("high")

    with console.status("[bold yellow]Loading configuration..."):
        with open(config_path) as f:
            config: TrainingConfig = yaml.safe_load(f)

    if config.get("test_mode", False):
        console.print("[yellow]Test mode enabled - using reduced settings[/yellow]")
        test_settings = config["test_mode_settings"]
        config.update(
            {
                "num_epochs": test_settings["num_epochs"],
                "batch_size": test_settings["batch_size"],
                "data_fraction": test_settings["data_fraction"],
            }
        )

    display_config_info(config, config_path)

    setup_logging(config.get("logging", {}))

    console.print("[bold green]Setting up datasets...[/bold green]")
    data_loaders = create_datasets(config)

    train_size = len(data_loaders["train"].dataset)
    val_size = len(data_loaders["val"].dataset)
    console.print(f"[OK] Training samples: [bold green]{train_size:,}[/bold green]")
    console.print(f"[OK] Validation samples: [bold green]{val_size:,}[/bold green]")

    console.print("\n[bold green]Setting up training pipeline...[/bold green]")
    trainer_module = create_trainer_module(
        config, train_dataloader=data_loaders["train"], val_dataloader=data_loaders["val"]
    )

    trainer_config = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "gradient_clip_val": config.get("regularization", {}).get("gradient_clip_val", 1.0),
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": config.get("gradient_accumulation_steps", 1),
        "check_val_every_n_epoch": 1,
        "callbacks": [trainer_module.vis_callback, RichProgressCallback()],
        "max_epochs": config["num_epochs"],
        "enable_progress_bar": False,
        "logger": False,
    }

    with console.status("[bold cyan]Initializing PyTorch Lightning trainer..."):
        trainer = pl.Trainer(**trainer_config)

    console.print("\n")
    console.print(Panel.fit("[STARTING] Training", style="bold green"))

    try:
        trainer.fit(trainer_module)
        console.print("\n")
        console.print(Panel.fit("[SUCCESS] Training Completed Successfully!", style="bold green"))

        if hasattr(trainer_module, "vis_callback") and hasattr(
            trainer_module.vis_callback, "metrics_file"
        ):
            metrics_file = trainer_module.vis_callback.metrics_file
            if metrics_file.exists():
                console.print(f"[INFO] Metrics saved to: [cyan]{metrics_file}[/cyan]")

    except KeyboardInterrupt:
        console.print("\n")
        console.print(Panel.fit("[INTERRUPTED] Training interrupted by user", style="bold yellow"))
    except Exception as e:
        console.print("\n")
        console.print(Panel.fit(f"[ERROR] Training failed: {str(e)}", style="bold red"))
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the training CLI argument parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    return parser


def cli() -> None:
    """Run the training CLI."""

    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
