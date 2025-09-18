"""Training script for SegFormer model."""
import os
import logging
import argparse
import yaml
import warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.data.datasets import SynchrotronDataset
from scrambledSeg.training.trainer import SegformerTrainer
from scrambledSeg.losses import create_loss  # Import loss factory function instead
from scrambledSeg.visualization.callbacks import VisualizationCallback
from scrambledSeg.transforms import create_train_transform, create_val_transform
from scrambledSeg.training.rich_progress import RichProgressCallback
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich import print as rprint

# Suppress repetitive warnings
warnings.filterwarnings('ignore', message='A new version of Albumentations is available')
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')
warnings.filterwarnings('ignore', category=FutureWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=UserWarning, module='pytorch_lightning')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*tensorboardX.*')
warnings.filterwarnings('ignore', message='.*LOCAL_RANK.*')
warnings.filterwarnings('ignore', message='.*Checkpoint directory.*')
warnings.filterwarnings('ignore', message='.*plain ModelCheckpoint.*')

# Initialize Rich console
console = Console()

logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable Albumentations update checks
# Add these to prevent OpenMP conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def setup_logging(config: dict) -> None:
    """Set up logging configuration with Rich handler."""
    log_level = config.get('level', 'INFO')
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Set up Rich logging handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True
    )
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(name)s - %(message)s',
        handlers=[rich_handler]
    )

def create_datasets(config: dict):
    """Create training and validation datasets."""
    with console.status("[bold green]Creating data transforms..."):
        train_transform = create_train_transform(config)
        val_transform = create_val_transform(config)
    
    # Get dataloader parameters
    dataloader_config = config.get('dataloader', {})
    batch_size = config.get('batch_size', 16)
    num_workers = dataloader_config.get('num_workers', 4)
    cache_size = dataloader_config.get('cache_size', 0)
    
    # Create datasets with progress indicators
    with console.status("[bold blue]Loading training dataset..."):
        train_dataset = SynchrotronDataset(
            data_dir=config['processed_data_dir'],
            split='train',
            transform=train_transform,
            normalize=True,
            cache_size=cache_size,
            subset_fraction=config.get('data_fraction', 1.0),
            random_seed=42
        )

    with console.status("[bold blue]Loading validation dataset..."):
        val_dataset = SynchrotronDataset(
            data_dir=config['processed_data_dir'],
            split='val',
            transform=val_transform,
            normalize=True,
            cache_size=cache_size,
            subset_fraction=config.get('data_fraction', 1.0),
            random_seed=43
        )

    data_loaders = {
        'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=dataloader_config.get('pin_memory', True),
                persistent_workers=dataloader_config.get('persistent_workers', True)
            ),
        'val': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=dataloader_config.get('pin_memory', True),
                persistent_workers=dataloader_config.get('persistent_workers', True)
            )
    }
    
    return data_loaders

def create_trainer_module(config: dict, train_dataloader, val_dataloader):
    """Create trainer module."""
    # Create model with progress indicator
    with console.status("[bold magenta]Initializing SegFormer model..."):
        model = CustomSegformer(
            encoder_name=config.get('encoder_name', 'nvidia/mit-b0'),
            num_classes=config.get('num_classes', 1),
            pretrained=config.get('pretrained', True),
            ignore_mismatched_sizes=True,
            in_channels=config.get('in_channels', 1)
        )
    
    # Create loss function using factory
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'crossentropy_dice')
    
    # Handle different parameter names between loss types
    params = loss_config.get('params', {}).copy()
    
    # Rename parameters for CrossEntropyDiceLoss if needed
    if loss_type == 'crossentropy_dice' and 'ce_weight' not in params and 'bce_weight' in params:
        params['ce_weight'] = params.pop('bce_weight')
        
    criterion = create_loss(loss_type, **params)

    # Create prediction config with all cleanup parameters 
    # Use 'prediction' section if available, fall back to 'thresholding' for compatibility
    source_config = config.get('prediction', config.get('thresholding', {}))
    prediction_config = {
        'enable_cleanup': source_config.get('enable_cleanup', True),
        'cleanup_kernel_size': source_config.get('cleanup_kernel_size', 3),
        'cleanup_threshold': source_config.get('cleanup_threshold', 5),
        'min_hole_size_factor': source_config.get('min_hole_size_factor', 64)
    }

    # Create optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_cls = getattr(torch.optim, optimizer_config.get('type', 'AdamW'))
    optimizer = optimizer_cls(
        model.parameters(),
        **optimizer_config.get('params', {'lr': 0.001})
    )
    
    # Create scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config.get('type', 'OneCycleLR'))
    scheduler_params = scheduler_config.get('params', {}).copy()
    
    # Calculate steps_per_epoch if not provided
    if scheduler_params.get('steps_per_epoch', -1) == -1:
        scheduler_params['steps_per_epoch'] = len(train_dataloader)
    
    scheduler = scheduler_cls(
        optimizer,
        **scheduler_params
    )
    
    # Create trainer module
    trainer_module = SegformerTrainer(
        model=model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        visualization=config.get('visualization', {}),
        log_dir=config.get('log_dir', 'logs'),
        test_mode=config.get('test_mode', False),
        threshold_params=prediction_config,
        num_classes=config['num_classes']
    )
    
    return trainer_module

def display_config_info(config: dict, config_path: str):
    """Display training configuration in a nice table."""
    table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Key configuration parameters
    table.add_row("Config File", str(config_path))
    table.add_row("Encoder", config.get('encoder_name', 'nvidia/mit-b0'))
    table.add_row("Num Classes", str(config.get('num_classes', 1)))
    table.add_row("Batch Size", str(config.get('batch_size', 16)))
    table.add_row("Num Epochs", str(config.get('num_epochs', 100)))
    table.add_row("Learning Rate", str(config.get('optimizer', {}).get('params', {}).get('lr', 0.001)))
    table.add_row("Data Directory", config.get('processed_data_dir', 'N/A'))
    table.add_row("Test Mode", str(config.get('test_mode', False)))
    
    # Device info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device += f" ({torch.cuda.get_device_name(0)})"
    table.add_row("Device", device)
    
    console.print("\n")
    console.print(table)
    console.print("\n")

def main(config_path):
    """Main training function."""
    # Print banner
    console.print(Panel.fit(
        Text("ScrambledSeg Training", style="bold magenta"),
        subtitle="SegFormer for Synchrotron X-ray Tomography"
    ))
    
    # Enable tensor core optimizations
    torch.set_float32_matmul_precision('high')
    
    # Load config
    with console.status("[bold yellow]Loading configuration..."):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Apply test mode settings if enabled
    if config.get('test_mode', False):
        console.print("[yellow]Test mode enabled - using reduced settings[/yellow]")
        test_settings = config['test_mode_settings']
        config.update({
            'num_epochs': test_settings['num_epochs'],
            'batch_size': test_settings['batch_size'],
            'data_fraction': test_settings['data_fraction']
        })
    
    # Display configuration
    display_config_info(config, config_path)
    
    # Set up logging
    setup_logging(config.get('logging', {}))
    
    # Create datasets
    console.print("[bold green]Setting up datasets...[/bold green]")
    data_loaders = create_datasets(config)
    
    # Display dataset info
    train_size = len(data_loaders['train'].dataset)
    val_size = len(data_loaders['val'].dataset)
    console.print(f"[OK] Training samples: [bold green]{train_size:,}[/bold green]")
    console.print(f"[OK] Validation samples: [bold green]{val_size:,}[/bold green]")
    
    # Create trainer module
    console.print("\n[bold green]Setting up training pipeline...[/bold green]")
    trainer_module = create_trainer_module(
        config,
        train_dataloader=data_loaders['train'],
        val_dataloader=data_loaders['val']
    )
    
    # Configure trainer
    trainer_config = {
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'precision': 'bf16-mixed',
        'gradient_clip_val': config.get('regularization', {}).get('gradient_clip_val', 1.0),
        'gradient_clip_algorithm': 'norm',
        'accumulate_grad_batches': config.get('gradient_accumulation_steps', 1),
        'check_val_every_n_epoch': 1,
        'callbacks': [trainer_module.vis_callback, RichProgressCallback()],
        'max_epochs': config['num_epochs'],
        'enable_progress_bar': False,  # Disable Lightning's progress bar in favor of Rich
        'logger': False  # Disable default logger to reduce noise
    }
    
    # Create trainer
    with console.status("[bold cyan]Initializing PyTorch Lightning trainer..."):
        trainer = pl.Trainer(**trainer_config)
    
    # Train model
    console.print("\n")
    console.print(Panel.fit("[STARTING] Training", style="bold green"))
    
    try:
        trainer.fit(trainer_module)
        console.print("\n")
        console.print(Panel.fit("[SUCCESS] Training Completed Successfully!", style="bold green"))
        
        # Display final metrics if available
        if hasattr(trainer_module, 'vis_callback') and hasattr(trainer_module.vis_callback, 'metrics_file'):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
