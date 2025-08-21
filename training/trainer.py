"""
Training pipeline for composition analysis models.

This module implements the main training loop with support for multi-task learning,
transfer learning from CLIP, and comprehensive evaluation metrics.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import json
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import wandb

from .dataset_loader import create_data_loaders, CLIPDatasetAdapter
from .losses import get_loss_function
from .metrics import CompositionMetrics
from models.hybrid_net import HybridCompositionNet

logger = logging.getLogger(__name__)

class CompositionTrainer:
    """
    Main trainer class for composition analysis models.
    
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.

        Args:
            config: Training configuration dictionary
        
        """

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize loss function
        self.criterion = get_loss_function(config['loss'])

        # Initialize metrics
        self.metrics = CompositionMetrics()

        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # CLIP adapter for transfer learning
        self.clip_adapter = None
        if config.get('use_clip_features', False):
            self.clip_adapter = CLIPDatasetAdapter(
                config.get('clip_model_name', 'openai/clip-vit-base-patch32')
            )
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping state
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize experiment tracking
        self._init_experiment_tracking()

    def flatten_config(self, config_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested configuration dictionary for MLflow logging.
        
        Args:
            config_dict: Configuration dictionary to flatten
            parent_key: Parent key for nested dictionaries
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""

        model_config = self.config['model']

        model = HybridCompositionNet(
            img_size = model_config.get('img_size', 224),
            patch_size = model_config.get('patch_size', 16),
            num_channels = model_config.get('num_channels', 3),
            hidden_size = model_config.get('hidden_size', 768),
            num_attention_heads = model_config.get('num_attention_heads', 12),
            num_hidden_layers = model_config.get('num_hidden_layers', 12),
            backbone = model_config.get('backbone', 'resnet50')
        )

        # Load pretrained weights if specified

        if 'pretrained_path' in model_config:
            self._load_pretrained_weights(model, model_config['pretrained_path'])

        return model
    
    def _load_pretrained_weights(self, model: nn.Module, pretrained_path: str):
        """Load pretrained weights with flexible key matching."""

        try:
            checkpoint = torch.load(pretrained_path, map_location = self.device)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']

            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            
            else:
                state_dict = checkpoint

            # Handle key mismatches (common in transfer learning)

            model_dict = model.state_dict()
            filtered_dict = {}

            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                
                else:
                    logger.warning(f"Skipping layer {k} due to size mismatch")

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)

            logger.info(f"Loaded pretrained weights from {pretrained_path}")
            logger.info(f"Loaded {len(filtered_dict)}/{len(state_dict)} layers")
        
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""

        optimizer_config = self.config['optimizer']
        optimizer_type = optimizer_config.get('type', 'adamw')

        # Separate learning rates for different components
        backbone_lr = optimizer_config.get('backbone_lr', 1e-5)
        head_lr = optimizer_config.get('head_lr', 1e-3)

        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'backbones' in n],

                'lr': backbone_lr,
                'name': 'backbone'
            },

            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'backbone' not in n],
                
                'lr': head_lr,
                'name': 'heads'
            }
        ]

        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                weight_decay = optimizer_config.get('weight_decay', 1e-4),
                betas = optimizer_config.get('betas', (0.9, 0.999))
            )

        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum = optimizer_config.get('momentum', 0.9),
                weight_decay = optimizer_config.get('weight_decay', 1e-4)
            )
        
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""

        if 'scheduler' not in self.config:
            return None
        
        scheduler_config = self.config['scheduler']
        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = self.config['training']['epochs'],
                eta_min = scheduler_config.get('min_lr', 1e-7)
            )

        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size = scheduler_config.get('step_size', 30),
                gamma = scheduler_config.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode = 'max',
                factor = scheduler_config.get('factor', 0.5),
                patience = scheduler_config.get('patience', 5),
                min_lr = scheduler_config.get('min_lr', 1e-7)
            )

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return scheduler
    
    def _init_experiment_tracking(self):
        """Initialize MLflow and Weights & Biases tracking."""

        # MLflow setup
        mlflow.set_tracking_uri(self.config.get('mlflow_uri', 'mlruns'))
        experiment_name = self.config.get('experiment_name', 'composition_training')

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)

            else:
                experiment_id = experiment.experiment_id

            mlflow.start_run(experiment_id = experiment_id)

            # Log hyperparameters

            mlflow.log_params(self.flatten_config(self.config))

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")

        # Weights & Biases setup
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project = self.config.get('wandb_project', 'composition-analysis'),
                    name = self.config.get('run_name'),
                    config = self.config
                )

            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")

    def _flatten_config(self, config: Dict, prefix: str = '') -> Dict:
        """Flatten nested configuration for MLflow logging."""

        flat_config = {}

        for key, value in config.items():
            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, f"{prefix}{key}."))

            else:
                flat_config[f"{prefix}{key}"] = value
            
        return flat_config
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        
        """

        self.model.train()

        total_loss = 0.0
        task_losses = {}
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc = f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items()
                       if k != 'image' and k != 'image_path'}
            
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(device_type = device_type, dtype = torch.float16):
                    predictions = self.model(images)

                    # Add CLIP features if using transfer learning
                    if self.clip_adapter is not None:
                        clip_features = self.clip_adapter.extract_features(images)
                        # Here you could modify predictions based on CLIP features
                        # For now, we'll use them as additional supervision

                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total']
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                        continue

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping before step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training']['max_grad_norm'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total']
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                    continue

                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training']['max_grad_norm'])
                
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for task, task_loss in loss_dict.items():
                if task != 'total':
                    if task not in task_losses:
                        task_losses[task] = 0.0
                    task_losses[task] += task_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
            })

        # Calculate average losses
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        return {
            'total_loss': avg_total_loss,
            **avg_task_losses
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        
        """

        self.model.eval()

        total_loss = 0.0
        task_losses = {}
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc = "Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items()
                           if k != 'image' and k != 'image_path'}
                
                # Forward pass
                if self.use_amp:
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type = device_type, dtype = torch.float16):
                        predictions = self.model(images)
                        loss_dict = self.criterion(predictions, targets)
                        loss = loss_dict['total']

                else:
                    predictions = self.model(images)
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total']

                # Update metrics
                total_loss += loss.item()
                for task, task_loss in loss_dict.items():
                    if task != 'total':
                        if task not in task_losses:
                            task_losses[task] = 0.0
                        task_losses[task] += task_loss.item()

                # Store predictions and targets for metrics calculation
                all_predictions.append({k: v.cpu() for k, v in predictions.items()})
                all_targets.append({k: v.cpu() for k, v in targets.items()})

        # Calculate average losses
        num_batches = len(val_loader)
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        # Calculate composition-specific metrics
        composition_metrics = self.metrics.compute_metrics(all_predictions, all_targets)

        return {
            'total_loss': avg_total_loss,
            **avg_task_losses,
            **composition_metrics
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        """

        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('overall_score', val_metrics['total_loss']))

                else:
                    self.scheduler.step()

            # Store metrics
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])

            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Early stopping check (this will update best_val_score if improved)
            should_stop = self._should_early_stop(val_metrics)

            # Save checkpoint
            current_score = val_metrics.get('overall_score', -val_metrics['total_loss'])
            is_best = current_score > self.best_val_score
            
            if is_best:
                self.best_val_score = current_score

            self._save_checkpoint(epoch, is_best, train_metrics, val_metrics)

            # Check if early stopping was triggered
            if should_stop:
                logger.info("Early stopping triggered")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time: .2f} seconds.")

        # Final model save
        self._save_final_model()

    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to various tracking systems."""

        # Console logging
        logger.info(f"Epoch {epoch}:")
        logger.info(f" Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f" Val Loss: {val_metrics['total_loss']:.4f}")

        if 'overall_score' in val_metrics:
            logger.info(f" Val Overall Score: {val_metrics['overall_score']:.4f}")

        # MLflow logging
        try:
            mlflow.log_metric("train_loss", train_metrics['total_loss'], step = epoch)
            mlflow.log_metric('val_loss', val_metrics['total_loss'], step = epoch)

            for task, loss in train_metrics.items():
                if task != 'total_loss':
                    mlflow.log_metric(f"train_{task}_loss", loss, step = epoch)
            
            for metric, value in val_metrics.items():
                if metric != 'total_loss':
                    mlflow.log_metric(f"val_{metric}", value, step = epoch)

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

        if self.config.get('use_wandb', False):
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    **{f"train_{k}": v for k, v in train_metrics.items() if
                       k != 'total_loss'},
                    **{f"val_{k}": v for k, v in val_metrics.items() if 
                       k != 'total_loss'}
                })
            
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {e}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool,
                         train_metrics: Dict, val_metrics: Dict):
        
        """Save model checkpoint."""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with score: {self.best_val_score:.4f}")

        # Save periodic checkpoints
        if epoch % self.config.get('save_frequency', 10) == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def _should_early_stop(self, val_metrics: Dict) -> bool:
        """Check if early stopping should be triggered."""

        if not self.config.get('early_stopping', False):
            return False
        
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('early_stopping_min_delta', 1e-4)

        # Get current validation score (use overall_score if available, otherwise use negative loss)
        current_score = val_metrics.get('overall_score', -val_metrics['total_loss'])
        
        # Check if current score is better than best score by at least min_delta
        if current_score > self.best_val_score + min_delta:
            self.epochs_without_improvement = 0
            self.best_epoch = self.current_epoch
            logger.info(f"Validation score improved from {self.best_val_score:.4f} to {current_score:.4f}")
            return False
        else:
            self.epochs_without_improvement += 1
            logger.info(f"No improvement for {self.epochs_without_improvement} epochs (best: {self.best_val_score:.4f} at epoch {self.best_epoch})")
            
            if self.epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
            
        return False 
    
    def _save_final_model(self):
        """Save the final trained model for inference."""

        # Save model for MLflow
        try:
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name = self.config.get('model_name', 'composition_model')
            )
        
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")

        # Save model weights separately
        final_weights_path = self.output_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_score': self.best_val_score
        }, final_weights_path)

        logger.info(f"Final model saved to {final_weights_path}")

def train_model(config_path: str):
        """
        Main function to train a composition model from configuration file.
    
    Args:
        config_path: Path to training configuration file
        
        """

        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Set up logging
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)

        # Create trainer
        trainer = CompositionTrainer(config)

        # Start training
        trainer.train(train_loader, val_loader)

        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = trainer.validate_epoch(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

        # Clean up
        try:
            mlflow.end_run()
        
        except:
            pass

        if config.get('use_wandb', False):
            try:
                wandb.finish()
            
            except:
                pass