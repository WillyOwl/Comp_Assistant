"""
Validation utilities for the AI Composition Assistant.

Provides image validation, quality checks, and preprocessing validation
to ensure robust input handling and error prevention.
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validator for image files and processing results."""
    
    def __init__(self, 
                 min_size: Tuple[int, int] = (32, 32),
                 max_size: Tuple[int, int] = (4096, 4096),
                 max_file_size_mb: float = 50.0,
                 supported_formats: List[str] = None):
        """
        Initialize the image validator.
        
        Args:
            min_size: Minimum allowed image dimensions (width, height)
            max_size: Maximum allowed image dimensions (width, height)
            max_file_size_mb: Maximum file size in megabytes
            supported_formats: List of supported file extensions
        """
        self.min_size = min_size
        self.max_size = max_size
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        if supported_formats is None:
            self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        else:
            self.supported_formats = [fmt.lower() for fmt in supported_formats]
    
    def validate_file_path(self, file_path: str) -> Dict[str, bool]:
        """
        Validate file path and basic file properties.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'exists': False,
            'readable': False,
            'size_ok': False,
            'format_supported': False,
            'valid': False
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return results
            results['exists'] = True
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                logger.warning(f"File is not readable: {file_path}")
                return results
            results['readable'] = True
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size_bytes:
                logger.warning(f"File too large: {file_size} bytes > {self.max_file_size_bytes}")
                return results
            if file_size == 0:
                logger.warning(f"File is empty: {file_path}")
                return results
            results['size_ok'] = True
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_ext not in self.supported_formats:
                logger.warning(f"Unsupported format: {file_ext}")
                return results
            results['format_supported'] = True
            
            results['valid'] = all([
                results['exists'],
                results['readable'], 
                results['size_ok'],
                results['format_supported']
            ])
            
        except Exception as e:
            logger.error(f"Error validating file path {file_path}: {str(e)}")
        
        return results
    
    def validate_image_content(self, image: np.ndarray) -> Dict[str, bool]:
        """
        Validate loaded image content and properties.
        
        Args:
            image: Loaded image as numpy array
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'not_empty': False,
            'valid_shape': False,
            'valid_dtype': False,
            'size_in_range': False,
            'no_corruption': False,
            'valid': False
        }
        
        try:
            # Check if image is not empty
            if image is None or image.size == 0:
                logger.warning("Image is empty or None")
                return results
            results['not_empty'] = True
            
            # Check image shape
            if len(image.shape) not in [2, 3]:
                logger.warning(f"Invalid image shape: {image.shape}")
                return results
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                logger.warning(f"Invalid number of channels: {image.shape[2]}")
                return results
            results['valid_shape'] = True
            
            # Check data type
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                logger.warning(f"Invalid data type: {image.dtype}")
                return results
            results['valid_dtype'] = True
            
            # Check image dimensions
            h, w = image.shape[:2]
            if w < self.min_size[0] or h < self.min_size[1]:
                logger.warning(f"Image too small: {w}x{h} < {self.min_size}")
                return results
            if w > self.max_size[0] or h > self.max_size[1]:
                logger.warning(f"Image too large: {w}x{h} > {self.max_size}")
                return results
            results['size_in_range'] = True
            
            # Check for corruption (NaN, infinite values)
            if np.isnan(image).any() or np.isinf(image).any():
                logger.warning("Image contains NaN or infinite values")
                return results
            results['no_corruption'] = True
            
            results['valid'] = all([
                results['not_empty'],
                results['valid_shape'],
                results['valid_dtype'],
                results['size_in_range'],
                results['no_corruption']
            ])
            
        except Exception as e:
            logger.error(f"Error validating image content: {str(e)}")
        
        return results
    
    def validate_preprocessing_output(self, 
                                    processed_image: np.ndarray,
                                    features: Optional[Dict] = None,
                                    expected_shape: Optional[Tuple] = None) -> Dict[str, bool]:
        """
        Validate preprocessing pipeline output.
        
        Args:
            processed_image: Preprocessed image
            features: Extracted features dictionary
            expected_shape: Expected output image shape
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'image_valid': False,
            'shape_correct': False,
            'normalized': False,
            'features_valid': False,
            'valid': False
        }
        
        try:
            # Validate basic image properties
            image_validation = self.validate_image_content(processed_image)
            results['image_valid'] = image_validation['valid']
            
            # Check expected shape
            if expected_shape is not None:
                if processed_image.shape[:len(expected_shape)] == expected_shape:
                    results['shape_correct'] = True
                else:
                    logger.warning(f"Shape mismatch: {processed_image.shape} != {expected_shape}")
            else:
                results['shape_correct'] = True
            
            # Check if image appears normalized
            if processed_image.dtype in [np.float32, np.float64]:
                min_val, max_val = processed_image.min(), processed_image.max()
                if 0.0 <= min_val <= max_val <= 1.0:
                    results['normalized'] = True
                elif -3.0 <= min_val <= max_val <= 3.0:  # Might be standardized
                    results['normalized'] = True
                else:
                    logger.warning(f"Image values out of expected range: [{min_val}, {max_val}]")
            else:
                results['normalized'] = True  # Assume integer images are valid
            
            # Validate features if provided
            if features is not None:
                expected_features = ['edges', 'lines', 'corners', 'contours', 'gray']
                features_present = all(feat in features for feat in expected_features)
                if features_present:
                    results['features_valid'] = True
                else:
                    missing = [f for f in expected_features if f not in features]
                    logger.warning(f"Missing features: {missing}")
            else:
                results['features_valid'] = True
            
            results['valid'] = all([
                results['image_valid'],
                results['shape_correct'],
                results['normalized'],
                results['features_valid']
            ])
            
        except Exception as e:
            logger.error(f"Error validating preprocessing output: {str(e)}")
        
        return results


def validate_batch_images(image_paths: List[str], 
                         validator: Optional[ImageValidator] = None) -> Dict[str, Dict]:
    """
    Validate a batch of image files.
    
    Args:
        image_paths: List of image file paths to validate
        validator: ImageValidator instance (creates default if None)
        
    Returns:
        Dictionary mapping file paths to validation results
    """
    if validator is None:
        validator = ImageValidator()
    
    results = {}
    for path in image_paths:
        results[path] = validator.validate_file_path(path)
    
    return results


def get_image_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with image statistics
    """
    stats = {}
    
    try:
        stats['mean'] = float(np.mean(image))
        stats['std'] = float(np.std(image))
        stats['min'] = float(np.min(image))
        stats['max'] = float(np.max(image))
        stats['median'] = float(np.median(image))
        
        # Calculate per-channel statistics for color images
        if len(image.shape) == 3:
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                stats[f'channel_{i}_mean'] = float(np.mean(channel))
                stats[f'channel_{i}_std'] = float(np.std(channel))
        
    except Exception as e:
        logger.error(f"Error calculating image statistics: {str(e)}")
    
    return stats


def check_image_quality(image: np.ndarray, 
                       min_contrast: float = 0.1,
                       max_noise_level: float = 0.3) -> Dict[str, bool]:
    """
    Perform basic image quality checks.
    
    Args:
        image: Input image
        min_contrast: Minimum acceptable contrast level
        max_noise_level: Maximum acceptable noise level
        
    Returns:
        Dictionary with quality check results
    """
    quality = {
        'sufficient_contrast': False,
        'low_noise': False,
        'not_overexposed': False,
        'not_underexposed': False,
        'good_quality': False
    }
    
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        if gray.dtype == np.uint8:
            gray_norm = gray.astype(np.float32) / 255.0
        else:
            gray_norm = gray
        
        # Check contrast
        contrast = np.std(gray_norm)
        quality['sufficient_contrast'] = contrast >= min_contrast
        
        # Estimate noise level using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_level = 1.0 / (1.0 + laplacian_var / 1000.0)  # Normalize
        quality['low_noise'] = noise_level <= max_noise_level
        
        # Check exposure
        mean_brightness = np.mean(gray_norm)
        quality['not_overexposed'] = mean_brightness <= 0.95
        quality['not_underexposed'] = mean_brightness >= 0.05
        
        # Overall quality assessment
        quality['good_quality'] = all([
            quality['sufficient_contrast'],
            quality['low_noise'],
            quality['not_overexposed'],
            quality['not_underexposed']
        ])
        
    except Exception as e:
        logger.error(f"Error checking image quality: {str(e)}")
    
    return quality


if __name__ == "__main__":
    # Example usage
    validator = ImageValidator()
    print("ImageValidator initialized successfully!")
    print(f"Supported formats: {validator.supported_formats}")
    print(f"Size limits: {validator.min_size} to {validator.max_size}")