"""
Image Preprocessing Module for Composition Assistant

Handles image acquisition, preprocessing, noise reduction, normalization,
and color space optimization as outlined in Stage one of the development pipeline.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Dict, Optional, Union
import logging

# COnfigure Logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Comprehensive image preprocessing class for composition analysis.
    
    Features:
    - Multi-format image loading
    - Adaptive resizing with aspect ratio preservation
    - Color space conversion and optimization
    - Noise reduction and normalization
    - Composition-specific feature extraction
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 preserve_aspect_ratio: bool = True,
                 noise_reduction: bool = True):
        """
        Initialize the ImagePreprocessor.
        Args:
            target_size: Target dimensions for processed images (width, height)
            preserve_aspect_ratio: Whether to maintain original aspect ratio
            noise_reduction: Whether to apply noise reduction algorithms
        """
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.noise_reduction = noise_reduction

    # Define normalization transforms for PyTorch compatibility

        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
    ])

        logger.info(f"ImagePreprocessor initialized with target_size = {target_size}")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path with format validation.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array in BGR format

        Raises:
            ValueError: If image cannot be loaded or is invalid
        """

        try:
            # Try OpenCV first for better performance

            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL for additional format support

                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if image is None or image.size == 0:
                raise ValueError(f"Could not load image from {image_path}")
            
            logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")

            return image
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Failed to load image: {str(e)}")
        
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions with optional aspect ratio preservation

        Args:
            image: Input image as numpy array

        Returns:
            Resized image
        """

        h, w = image.shape[: 2]
        target_w, target_h = self.target_size

        if self.preserve_aspect_ratio:
            # Calculate scaling factor to fit within target size

            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize maintaining aspect ratio

            resized = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LANCZOS4)

            # Create canvas and center the resized image

            canvas = np.zeros((target_h, target_w, 3), dtype = image.dtype)

            y_offset = (target_h - new_h) // 2 # Interger division (round down)
            x_offset = (target_w - new_w) // 2

            canvas[y_offset: y_offset + new_h, x_offset: x_offset + new_w] = resized

            return canvas
        
        else:
            # Direct resize without preserving aspect ratio

            return cv2.resize(image, self.target_size, interpolation = cv2.INTER_LANCZOS4)
        
    def apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction algorithms for better feature extraction

        Args:
            image: Input imnge

        Returns:
            Denoised image
        """

        if not self.noise_reduction:
            return image
        
        #Apply Non-Local Means Denoising

        if len(image.shape) == 3:
            #Color image

            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        else:
            #Grayscale image

            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        return denoised
    
    def convert_color_space(self, image: np.ndarray,
                            target_space: str) -> np.ndarray:
        """
        Convert image to specified color space.

        Args:
            image: Input image in BGR format
            target_space: Target color space ('RGB', 'HSV', 'LAB', 'GRAY')

        Returns:
            Converted image
        """

        if target_space == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        elif target_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        elif target_space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        elif target_space == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:
            raise ValueError(f"Unsupported color space: {target_space}")
        
    def normalize_image(self, image: np.ndarray,
                        method: str) -> np.ndarray:
        """
        Normalize image pixel values.

        Args:
            image: Input image
            method: Normalization method ('standand', 'minmax', 'pytorch')

        Returns:
            Normalized image
        """

        if method ==  'standard':
            # Standard Normalization [0, 1]

            return image.astype(np.float32) / 255.0
        
        elif method == 'minmax':
            # Minmax Normalization

            return (image - image.min()) / (image.max() - image.min())
        
        elif method == 'pytorch':
            # PyTorch ImageNet Normalization

            image_rgb = self.convert_color_space(image, 'RGB')
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
            return np.array(self.normalize_transform(pil_image))
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
    def extract_composition_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract basic composition-related features from the image.

        Args:
            image: Input image

        Returns:
            Dictionary containing extracted features
        """

        # Convert to grascale for feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 100, threshold = 50,
                                minLineLength = 30, maxLineGap = 10)
        
        # Corner detection using Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(gray, maxCorners = 100,
                                          qualityLevel = 0.01, minDistance = 10)
        
        # Contour detection for shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'edges': edges,
            'lines': lines if lines is not None else np.array([]),
            'corners': corners if corners is not None else np.array([]),
            'contours': contours,
            'gray': gray
        }

        logger.debug(f"Extracted features: {len(features)} types")
        return features
    
    def preprocess(self, image_path: str,
                   return_features: bool,
                   color_space: str = 'RGB',
                   normalization: str = 'standard') -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Complete preprocessing pipeline for a single image

        Args:
            image_path: Path to input image
            return_features: Whether to extract and return compsition features
            color_space: Target color space for final image
            normalization: Normalization method to apply

        Returns:
            Tuple of (processed_image, features_dict)
        """

        try:
            # Load image
            image = self.load_image(image_path)

            # Apply noise reduction
            image = self.apply_noise_reduction(image)

            # Resize image
            image = self.resize_image(image)

            # Extract feature before color conversion (if requested)
            features = None

            if return_features:
                features = self.extract_composition_features(image)

            # Convert color space
            image = self.convert_color_space(image, color_space)

            # Normalize
            image = self.normalize_image(image, normalization)

            logger.info(f"Successfully preprocessed imageL {image_path}")
            return image, features
        
        except Exception as e:
            logger.error(f"Preprocessing failed for {image_path}: {str(e)}")
            raise

    def batch_preprocess(self, image_paths: list,
                         **kwargs) -> Tuple[np.ndarray, list]:
        
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image file paths
            **kwargs: Arguments to pass to preprocess method

        Returns:
            Tuple of (stacked_images, list_of_features)
        """

        processed_images = []
        all_features = []

        for path in image_paths:
            try:
                image, features = self.preprocess(path, **kwargs)
                processed_images.append(image)
                all_features.append(features)

            except Exception as e:
                logger.warning(f"Skipping {path} due to error: {str(e)}")
                continue

        if processed_images:
            # Stack images for batch processing
            images_array = np.stack(processed_images, axis = 0)
            logger.info(f"Batch processed {len(processed_images)} images")
            return images_array, all_features
        
        else:
            raise ValueError("No images could be processed successfully")
        
def create_preprocessing_pipeline(config: Optional[Dict] = None) -> ImagePreprocessor:
    """
    Factory function to create a configured preprocessing pipeline.

    Args:
        config: Configuration dictionary with preprocessing parameters

    Returns:
        Configured ImagePreprocessor instance
    """

    if config is None:
        config = {}

    return ImagePreprocessor(
        target_size = config.get('target_size', (224, 224)),
        preserve_aspect_ratio = config.get('preserve_aspect_ratio', True),
        noise_reduction = config.get('noise_reduction', True)
    )

if __name__ == "__main__":
    # Example usage and testing

    preprocessor = ImagePreprocessor()

    print("ImagePreprocessor initialized successfully!")
    print(f"Target size: {preprocessor.target_size}")
    print(f"Preserve aspect ratio: {preprocessor.preserve_aspect_ratio}")
    print(f"Noise reduction: {preprocessor.noise_reduction}")