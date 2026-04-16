"""Training entrypoint for the ScrambledSeg SegFormer workflow."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scrambledSeg.training.config import (
    DatasetConfig,
    LoggingConfig,
    TrainerModuleConfig,
    TrainerRuntimeConfig,
    TrainingAppConfig,
    normalize_training_config,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from scrambledSeg.training.trainer import SegformerTrainer


_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

console = Console()
logger = logging.getLogger(__name__)


def configure_training_environment() -> None:
    """Apply process-wide defaults before importing the heavy training stack."""

    for name, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)

    warnings.filterwarnings("ignore", message="A new version of Albumentations is available")
    warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
    warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*tensorboardX.*")
    warnings.filterwarnings("ignore", message=".*LOCAL_RANK.*")
    warnings.filterwarnings("ignore", message=".*Checkpoint directory.*")
    warnings.filterwarnings("ignore", message=".*plain ModelCheckpoint.*")


def setup_logging(config: LoggingConfig) -> None:
    """Set up logging with a Rich handler."""

    logging.getLogger().handlers = []
    numeric_level = getattr(logging, config.level.upper())

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(name)s - %(message)s",
        handlers=[rich_handler],
    )


def create_dataloaders(config: DatasetConfig) -> dict[str, DataLoader]:
    """Create the training and validation dataloaders."""

    from torch.utils.data import DataLoader

    from scrambledSeg.data.datasets import SynchrotronDataset
    from scrambledSeg.training.transforms import create_train_transform, create_val_transform

    with console.status("[bold green]Creating data transforms..."):
        train_transform = create_train_transform(config.augmentation)
        val_transform = create_val_transform()

    dataloader_config = config.dataloader

    with console.status("[bold blue]Loading training dataset..."):
        train_dataset = SynchrotronDataset(
            data_dir=config.processed_data_dir,
            split="train",
            transform=train_transform,
            normalize=True,
            cache_size=dataloader_config.cache_size,
            subset_fraction=config.data_fraction,
            random_seed=42,
        )

    with console.status("[bold blue]Loading validation dataset..."):
        val_dataset = SynchrotronDataset(
            data_dir=config.processed_data_dir,
            split="val",
            transform=val_transform,
            normalize=True,
            cache_size=dataloader_config.cache_size,
            subset_fraction=config.data_fraction,
            random_seed=43,
        )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=dataloader_config.persistent_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            persistent_workers=dataloader_config.persistent_workers,
        ),
    }


def create_trainer_module(
    config: TrainerModuleConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> SegformerTrainer:
    """Create the Lightning module for training."""

    import torch

    from scrambledSeg.losses import create_loss
    from scrambledSeg.models.segformer import CustomSegformer
    from scrambledSeg.training.trainer import SegformerTrainer

    with console.status("[bold magenta]Initializing SegFormer model..."):
        model = CustomSegformer(
            encoder_name=config.model.encoder_name,
            encoder_revision=config.model.encoder_revision,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            ignore_mismatched_sizes=True,
        )

    criterion = create_loss(config.loss.type, **dict(config.loss.params))

    optimizer_cls = getattr(torch.optim, config.optimizer.type)
    optimizer = optimizer_cls(model.parameters(), **dict(config.optimizer.params))

    scheduler_cls = getattr(torch.optim.lr_scheduler, config.scheduler.type)
    scheduler_params = dict(config.scheduler.params)
    if scheduler_params.get("steps_per_epoch", -1) == -1:
        scheduler_params["steps_per_epoch"] = len(train_dataloader)
    scheduler = scheduler_cls(optimizer, **scheduler_params)

    return SegformerTrainer(
        model=model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        visualization=config.visualization,
        log_dir=config.log_dir,
        test_mode=config.test_mode,
        threshold_params=config.prediction,
        num_classes=config.num_classes,
        model_config=config.model,
    )


def display_config_info(config: TrainingAppConfig, config_path: str) -> None:
    """Display the effective training configuration."""

    import torch

    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Config File", str(config_path))
    table.add_row("Encoder", config.trainer_module.model.encoder_name)
    table.add_row("Num Classes", str(config.trainer_module.num_classes))
    table.add_row("Batch Size", str(config.dataset.batch_size))
    table.add_row("Num Epochs", str(config.trainer_runtime.num_epochs))
    table.add_row("Learning Rate", str(config.trainer_module.optimizer.params.get("lr", 0.001)))
    table.add_row("Data Directory", config.dataset.processed_data_dir)
    table.add_row("Test Mode", str(config.trainer_module.test_mode))

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device += f" ({torch.cuda.get_device_name(0)})"
    table.add_row("Device", device)

    console.print("\n")
    console.print(table)
    console.print("\n")


def build_trainer_config(
    config: TrainerRuntimeConfig,
    vis_callback: Any,
) -> dict[str, object]:
    """Build the PyTorch Lightning trainer kwargs from runtime-only settings."""

    import torch

    from scrambledSeg.training.rich_progress import RichProgressCallback

    return {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "gradient_clip_val": config.gradient_clip_val,
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": config.gradient_accumulation_steps,
        "check_val_every_n_epoch": 1,
        "callbacks": [vis_callback, RichProgressCallback()],
        "max_epochs": config.num_epochs,
        "enable_progress_bar": False,
        "logger": False,
    }


def create_lightning_trainer(trainer_config: dict[str, object]) -> Any:
    """Create the PyTorch Lightning trainer instance."""

    import pytorch_lightning as pl

    return pl.Trainer(**trainer_config)


def main(config_path: str) -> None:
    """Run training from a config file."""

    configure_training_environment()

    import torch

    console.print(
        Panel.fit(
            Text("ScrambledSeg Training", style="bold magenta"),
            subtitle="SegFormer for Synchrotron X-ray Tomography",
        )
    )

    torch.set_float32_matmul_precision("high")

    with console.status("[bold yellow]Loading configuration..."):
        with open(config_path) as handle:
            raw_config = yaml.safe_load(handle) or {}
        config = normalize_training_config(raw_config)

    if config.trainer_module.test_mode:
        console.print("[yellow]Test mode enabled; applying reduced settings.[/yellow]")

    display_config_info(config, config_path)
    setup_logging(config.logging)

    console.print("[bold green]Setting up datasets...[/bold green]")
    data_loaders = create_dataloaders(config.dataset)

    train_size = len(data_loaders["train"].dataset)
    val_size = len(data_loaders["val"].dataset)
    console.print(f"Training samples: [bold green]{train_size:,}[/bold green]")
    console.print(f"Validation samples: [bold green]{val_size:,}[/bold green]")

    console.print("\n[bold green]Setting up training pipeline...[/bold green]")
    trainer_module = create_trainer_module(
        config.trainer_module,
        train_dataloader=data_loaders["train"],
        val_dataloader=data_loaders["val"],
    )
    trainer_config = build_trainer_config(config.trainer_runtime, trainer_module.vis_callback)

    with console.status("[bold cyan]Initializing PyTorch Lightning trainer..."):
        trainer = create_lightning_trainer(trainer_config)

    console.print("\n")
    console.print(Panel.fit("Training started", style="bold green"))

    try:
        trainer.fit(trainer_module)
        console.print("\n")
        console.print(Panel.fit("Training finished successfully", style="bold green"))

        if hasattr(trainer_module, "vis_callback") and hasattr(
            trainer_module.vis_callback, "metrics_file"
        ):
            metrics_file = trainer_module.vis_callback.metrics_file
            if metrics_file.exists():
                console.print(f"Metrics written to [cyan]{metrics_file}[/cyan]")

    except KeyboardInterrupt:
        console.print("\n")
        console.print(Panel.fit("Training interrupted by user", style="bold yellow"))
    except Exception as exc:
        console.print("\n")
        console.print(Panel.fit(f"Training failed: {exc}", style="bold red"))
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
