#!/usr/bin/env python3
"""
Main training script for composition analysis models.

This script orchestrates the entire training pipeline including dataset preparation,
model training, hyperparameter optimization, and evaluation.

Usage:
    python train.py --config configs/training_config.json
    python train.py --config configs/training_config.json --optimize
    python train.py --resume ./training_outputs/checkpoints/latest.pth
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.trainer import CompositionTrainer, train_model
from training.dataset_loader import create_data_loaders
from training.hyperparameter_optimization import run_hyperparameter_optimization
from utils.validation_api import validate_analysis_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_reproducibility(seed: int = 42):
    """Set up reproducibility for training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Reproducibility set up with seed: {seed}")


def setup_device() -> torch.device:
    """Set up the computing device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_sections = ['data', 'model', 'training', 'optimizer', 'loss']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate data paths
    data_config = config['data']
    required_paths = ['train_data_dir', 'val_data_dir', 'train_annotations', 'val_annotations']
    
    for path_key in required_paths:
        if path_key in data_config:
            path = Path(data_config[path_key])
            if not path.exists():
                logger.warning(f"Path does not exist: {path} (will be created if needed)")
    
    # Validate model parameters
    model_config = config['model']
    if model_config.get('hidden_size', 768) % model_config.get('num_attention_heads', 12) != 0:
        logger.error("hidden_size must be divisible by num_attention_heads")
        return False
    
    # Validate batch size vs GPU memory
    batch_size = config['training']['batch_size']
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if batch_size > 64 and gpu_memory_gb < 8:
            logger.warning(f"Large batch size ({batch_size}) with limited GPU memory ({gpu_memory_gb:.1f} GB)")
    
    return True


def create_output_directories(output_dir: str):
    """Create necessary output directories."""
    output_path = Path(output_dir)
    directories = [
        output_path,
        output_path / 'checkpoints',
        output_path / 'logs',
        output_path / 'visualizations',
        output_path / 'predictions'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directories in: {output_path}")


def prepare_datasets(config: Dict[str, Any]) -> bool:
    """
    Prepare and validate datasets for training.
    
    Args:
        config: Training configuration
        
    Returns:
        True if datasets are ready
    """
    data_config = config['data']
    
    # Check if dataset directories exist
    train_dir = Path(data_config['train_data_dir'])
    val_dir = Path(data_config['val_data_dir'])
    
    if not train_dir.exists() or not val_dir.exists():
        logger.error("Dataset directories not found. Please prepare your datasets first.")
        logger.info("Expected structure:")
        logger.info("  datasets/")
        logger.info("    cadb/")
        logger.info("      train/")
        logger.info("      val/")
        logger.info("      test/")
        logger.info("      annotations/")
        logger.info("        train.csv")
        logger.info("        val.csv")
        logger.info("        test.csv")
        return False
    
    # Try to create data loaders to validate dataset
    try:
        logger.info("Validating datasets...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Test loading a batch
        train_batch = next(iter(train_loader))
        logger.info(f"Batch shape: {train_batch['image'].shape}")
        logger.info("Dataset validation successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False


def resume_training(checkpoint_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional new configuration (overrides checkpoint config)
        
    Returns:
        True if resuming was successful
    """
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Use checkpoint config if no new config provided
        if config is None:
            config = checkpoint['config']
        
        # Create trainer
        trainer = CompositionTrainer(config)
        
        # Load checkpoint
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_score = checkpoint.get('best_val_score', 0.0)
        
        logger.info(f"Resumed training from epoch {trainer.current_epoch}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Continue training
        remaining_epochs = config['training']['epochs'] - trainer.current_epoch
        if remaining_epochs > 0:
            config['training']['epochs'] = remaining_epochs
            trainer.train(train_loader, val_loader)
        else:
            logger.info("Training already completed according to checkpoint")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to resume training: {e}")
        return False


def run_training(config: Dict[str, Any]) -> bool:
    """
    Run the main training process.
    
    Args:
        config: Training configuration
        
    Returns:
        True if training completed successfully
    """
    try:
        # Validate configuration
        if not validate_config(config):
            return False
        
        # Create output directories
        create_output_directories(config['output_dir'])
        
        # Prepare datasets
        if not prepare_datasets(config):
            return False
        
        # Set up reproducibility
        setup_reproducibility(config.get('seed', 42))
        
        # Set up device
        device = setup_device()
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = CompositionTrainer(config)
        
        # Start training
        logger.info("Starting training process...")
        start_time = time.time()
        
        trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Final evaluation on test set
        logger.info("Running final evaluation on test set...")
        test_metrics = trainer.validate_epoch(test_loader)
        
        logger.info("Test Set Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save final results
        results_path = Path(config['output_dir']) / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'test_metrics': test_metrics,
                'training_time': training_time,
                'best_val_score': trainer.best_val_score,
                'config': config
            }, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_hyperparameter_optimization(config_path: str, optimization_config_path: str, 
                                   n_trials: int = 100) -> bool:
    """
    Run hyperparameter optimization.
    
    Args:
        config_path: Path to base training configuration
        optimization_config_path: Path to optimization configuration
        n_trials: Number of optimization trials
        
    Returns:
        True if optimization completed successfully
    """
    try:
        logger.info("Starting hyperparameter optimization...")
        
        best_config = run_hyperparameter_optimization(
            base_config_path=config_path,
            optimization_config_path=optimization_config_path,
            n_trials=n_trials
        )
        
        # Train final model with best configuration
        logger.info("Training final model with best hyperparameters...")
        success = run_training(best_config)
        
        return success
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train composition analysis models')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--optimization-config', type=str,
                       default='configs/hyperparameter_optimization.json',
                       help='Path to hyperparameter optimization configuration')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of hyperparameter optimization trials')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       default='auto', help='Device to use for training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set up debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {args.config}: {e}")
        sys.exit(1)
    
    # Override device if specified
    if args.device != 'auto':
        config['hardware']['device'] = args.device
    
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Resume training if specified
    if args.resume:
        logger.info(f"Resuming training from: {args.resume}")
        success = resume_training(args.resume, config)
        sys.exit(0 if success else 1)
    
    # Run hyperparameter optimization if requested
    if args.optimize:
        logger.info("Running hyperparameter optimization...")
        success = run_hyperparameter_optimization(
            args.config, 
            args.optimization_config, 
            args.n_trials
        )
    else:
        # Run normal training
        logger.info("Running normal training...")
        success = run_training(config)
    
    # Exit with appropriate code
    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()