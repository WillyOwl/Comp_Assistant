"""
Utilities Module for AI Composition Assistant

Provides helper functions, validation tools, and common utilities
for the composition analysis system.
"""

# Stage 1: Training/Preprocessing validation utilities
from .validation import ImageValidator, validate_batch_images, get_image_statistics, check_image_quality

# Stage 4: API/Web validation utilities  
from .validation_api import (
    validate_image_format, 
    validate_file_size, 
    validate_analysis_config,
    validate_batch_size,
    sanitize_filename,
    validate_image_dimensions,
    validate_api_token,
    validate_metadata,
    ValidationError,
    SUPPORTED_IMAGE_FORMATS,
    MAX_FILE_SIZE
)

# Dataset utilities
from .dataset_setup import DatasetManager
from .synthetic_dataset_generator import SyntheticCompositionDataset

__all__ = [
    # Stage 1 validation (training/preprocessing)
    'ImageValidator', 
    'validate_batch_images', 
    'get_image_statistics', 
    'check_image_quality',
    
    # Stage 4 validation (API/web)
    'validate_image_format',
    'validate_file_size', 
    'validate_analysis_config',
    'validate_batch_size',
    'sanitize_filename',
    'validate_image_dimensions',
    'validate_api_token',
    'validate_metadata',
    'ValidationError',
    'SUPPORTED_IMAGE_FORMATS',
    'MAX_FILE_SIZE',
    
    # Dataset utilities
    'DatasetManager',
    'SyntheticCompositionDataset'
]