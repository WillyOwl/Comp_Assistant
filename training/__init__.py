"""
Training package for composition analysis models.

This package provides comprehensive training infrastructure including:
- Dataset loading and preprocessing
- Multi-task loss functions
- Training loops with mixed precision
- Hyperparameter optimization
- CLIP-based transfer learning
- MLflow integration for experiment tracking
- Comprehensive evaluation metrics
"""

from .trainer import CompositionTrainer, train_model
from .dataset_loader import CompositionDataset, create_data_loaders, CLIPDatasetAdapter
from .losses import (
    EarthMoversDistance, 
    CompositionMultiTaskLoss, 
    FocalLoss, 
    ContrastiveLoss,
    get_loss_function
)
from .metrics import CompositionMetrics, PerformanceBenchmark
from .hyperparameter_optimization import (
    CompositionHyperparameterOptimizer,
    run_hyperparameter_optimization,
    GridSearchOptimizer
)
from .clip_transfer import (
    CLIPAestheticAdapter,
    CLIPEnhancedCompositionNet,
    create_clip_enhanced_model
)

__version__ = "1.0.0"

__all__ = [
    # Main training
    'CompositionTrainer',
    'train_model',
    
    # Data loading
    'CompositionDataset',
    'create_data_loaders',
    'CLIPDatasetAdapter',
    
    # Loss functions
    'EarthMoversDistance',
    'CompositionMultiTaskLoss',
    'FocalLoss',
    'ContrastiveLoss',
    'get_loss_function',
    
    # Metrics
    'CompositionMetrics',
    'PerformanceBenchmark',
    
    # Hyperparameter optimization
    'CompositionHyperparameterOptimizer',
    'run_hyperparameter_optimization',
    'GridSearchOptimizer',
    
    # CLIP transfer learning
    'CLIPAestheticAdapter',
    'CLIPEnhancedCompositionNet',
    'create_clip_enhanced_model',
]