"""Training script for SegFormer model."""
import os
import logging
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from scrambledSeg.models.segformer import CustomSegformer
from scrambledSeg.data.datasets import SynchrotronDataset
from scrambledSeg.training.trainer import SegformerTrainer
from scrambledSeg.losses import BCEDiceLoss  # Updated import
from scrambledSeg.visualization.callbacks import VisualizationCallback
from scrambledSeg.transforms import create_train_transform, create_val_transform
from tqdm import tqdm

logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Add these to prevent OpenMP conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def setup_logging(config: dict) -> None:
    """Set up logging configuration."""
    log_level = config.get('level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_datasets(config: dict):
    """Create training and validation datasets."""
    # Create transforms
    train_transform = create_train_transform(config)
    val_transform = create_val_transform(config)
    
    # Get dataloader parameters
    dataloader_config = config.get('dataloader', {})
    batch_size = config.get('batch_size', 16)
    num_workers = dataloader_config.get('num_workers', 4)
    cache_size = dataloader_config.get('cache_size', 0)
    
    # Create dataloaders
    train_dataset = SynchrotronDataset(
        data_dir=config['processed_data_dir'],
        split='train',
        transform=train_transform,
        normalize=True,
        cache_size=cache_size,
        subset_fraction=config.get('data_fraction', 1.0),
        random_seed=42
    )

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
    # Create model
    model = CustomSegformer(
        encoder_name=config.get('encoder_name', 'nvidia/mit-b0'),
        num_classes=config.get('num_classes', 1),
        pretrained=config.get('pretrained', True),
        ignore_mismatched_sizes=True,
        dropout_rate=config.get('regularization', {}).get('dropout_rate', 0.1),
        enable_intermediate_supervision=config.get('enable_intermediate_supervision', True)
    )
    
    # Create BCE+Dice loss function
    loss_config = config.get('loss', {})
    criterion = BCEDiceLoss(
        bce_weight=loss_config.get('params', {}).get('bce_weight', 0.5),
        dice_weight=loss_config.get('params', {}).get('dice_weight', 0.5),
        smooth=loss_config.get('params', {}).get('smooth', 1.0)
    )

    # Create threshold config with all cleanup parameters
    threshold_config = {
        'threshold': config.get('thresholding', {}).get('threshold', 0.5),
        'enable_cleanup': config.get('thresholding', {}).get('enable_cleanup', True),
        'cleanup_kernel_size': config.get('thresholding', {}).get('cleanup_kernel_size', 3),
        'cleanup_threshold': config.get('thresholding', {}).get('cleanup_threshold', 5),
        'min_hole_size_factor': config.get('thresholding', {}).get('min_hole_size_factor', 64)
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
        threshold_params=threshold_config
    )
    
    return trainer_module

def main(config_path):
    """Main training function."""
    # Enable tensor core optimizations
    torch.set_float32_matmul_precision('high')
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Apply test mode settings if enabled
    if config.get('test_mode', False):
        test_settings = config['test_mode_settings']
        config.update({
            'num_epochs': test_settings['num_epochs'],
            'batch_size': test_settings['batch_size'],
            'data_fraction': test_settings['data_fraction']
        })
    
    # Set up logging
    setup_logging(config.get('logging', {}))
    
    # Create datasets
    data_loaders = create_datasets(config)
    
    # Create trainer module
    trainer_module = create_trainer_module(
        config,
        train_dataloader=data_loaders['train'],
        val_dataloader=data_loaders['val']
    )
    
    # Configure trainer
    trainer_config = {
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'precision': '16-mixed',
        'gradient_clip_val': config.get('regularization', {}).get('gradient_clip_val', 1.0),
        'gradient_clip_algorithm': 'norm',
        'accumulate_grad_batches': config.get('gradient_accumulation_steps', 1),
        'check_val_every_n_epoch': 1,
        'callbacks': [trainer_module.vis_callback],
        'max_epochs': config['num_epochs']
    }
    
    # Create trainer
    trainer = pl.Trainer(**trainer_config)
    
    # Train model
    trainer.fit(trainer_module)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    args = parser.parse_args()
    main(args.config)