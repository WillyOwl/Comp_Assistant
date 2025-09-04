"""
Validation utilities for the AI Composition Assistant

This module provides validation functions for API inputs, image formats,
and configuration parameters.
"""

import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'
}

# Maximum file size (in bytes) - 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

def validate_image_format(filename: str) -> bool:
    """
    Validate if the image format is supported.
    
    Args:
        filename: Name of the image file
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in SUPPORTED_IMAGE_FORMATS

def validate_file_size(file_size: int) -> bool:
    """
    Validate if the file size is within acceptable limits.
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        bool: True if size is acceptable, False otherwise
    """
    return 0 < file_size <= MAX_FILE_SIZE

def validate_analysis_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate analysis configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate analysis depth
    if 'analysis_depth' in config:
        valid_depths = {'basic', 'standard', 'comprehensive'}
        if config['analysis_depth'] not in valid_depths:
            errors.append(f"Invalid analysis_depth. Must be one of: {valid_depths}")
    
    # Validate rule weights
    if 'rule_weights' in config:
        rule_weights = config['rule_weights']
        if not isinstance(rule_weights, dict):
            errors.append("rule_weights must be a dictionary")
        else:
            valid_rules = {
                'rule_of_thirds', 'leading_lines', 'symmetry', 
                'depth_layering', 'color_harmony'
            }
            
            for rule, weight in rule_weights.items():
                if rule not in valid_rules:
                    errors.append(f"Invalid rule '{rule}'. Valid rules: {valid_rules}")
                
                if not isinstance(weight, (int, float)) or not (0 <= weight <= 1):
                    errors.append(f"Rule weight for '{rule}' must be a number between 0 and 1")
    
    # Validate max_suggestions
    if 'max_suggestions' in config:
        max_suggestions = config['max_suggestions']
        if not isinstance(max_suggestions, int) or not (1 <= max_suggestions <= 20):
            errors.append("max_suggestions must be an integer between 1 and 20")
    
    # Validate boolean fields
    boolean_fields = ['return_visualizations', 'include_technical_metrics']
    for field in boolean_fields:
        if field in config and not isinstance(config[field], bool):
            errors.append(f"{field} must be a boolean value")
    
    return len(errors) == 0, errors

def validate_batch_size(batch_size: int) -> bool:
    """
    Validate batch size for batch processing.
    
    Args:
        batch_size: Number of images in batch
        
    Returns:
        bool: True if batch size is acceptable, False otherwise
    """
    return 1 <= batch_size <= 100  # Maximum 100 images per batch

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove any path components
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

def validate_image_dimensions(width: int, height: int) -> tuple[bool, List[str]]:
    """
    Validate image dimensions are within acceptable ranges.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Minimum dimensions (too small images may not provide meaningful analysis)
    if width < 100 or height < 100:
        errors.append("Image dimensions too small (minimum 100x100 pixels)")
    
    # Maximum dimensions (prevent memory issues)
    if width > 10000 or height > 10000:
        errors.append("Image dimensions too large (maximum 10000x10000 pixels)")
    
    # Aspect ratio check (prevent extremely distorted images)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 10:
        errors.append("Image aspect ratio too extreme (maximum 10:1)")
    
    return len(errors) == 0, errors

def validate_api_token(token: str) -> bool:
    """
    Validate API token format and structure.
    
    Args:
        token: API token string
        
    Returns:
        bool: True if token is valid format, False otherwise
    """
    if not token:
        return False
    
    # For demo purposes - implement proper JWT validation in production
    if token == "demo-token":
        return True
    
    # Basic format validation
    if len(token) < 10:
        return False
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[A-Za-z0-9_-]+$', token):
        return False
    
    return True

def validate_metadata(metadata: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate optional metadata provided with requests.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(metadata, dict):
        return False, ["Metadata must be a dictionary"]
    
    # Limit metadata size
    if len(str(metadata)) > 10000:  # 10KB limit
        errors.append("Metadata too large (maximum 10KB)")
    
    # Validate metadata keys and values
    for key, value in metadata.items():
        if not isinstance(key, str):
            errors.append("Metadata keys must be strings")
        
        if len(key) > 100:
            errors.append(f"Metadata key '{key}' too long (maximum 100 characters)")
        
        # Validate value types (allow basic JSON-serializable types)
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            errors.append(f"Metadata value for '{key}' must be JSON-serializable")
    
    return len(errors) == 0, errors

class ValidationError(Exception):
    """Custom exception for validation errors"""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []