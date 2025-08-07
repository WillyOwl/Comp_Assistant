#!/usr/bin/env python3
"""
Stage One Demonstration: Image Acquisition and Preprocessing

This script demonstrates the complete Stage One pipeline for the AI Composition Assistant,
showcasing image loading, preprocessing, feature extraction, and validation.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tempfile
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import ImagePreprocessor, create_preprocessing_pipeline
from utils.validation import ImageValidator, get_image_statistics, check_image_quality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_image(width=400, height=300, save_path=None):
    """
    Create a demo image with composition elements for testing.
    
    Args:
        width: Image width
        height: Image height
        save_path: Path to save the image (optional)
        
    Returns:
        Path to the created image
    """
    # Create a canvas
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add background gradient
    for i in range(height):
        intensity = int(255 * (i / height) * 0.3 + 50)
        image[i, :] = [intensity, intensity + 20, intensity + 40]
    
    # Add some geometric shapes for composition analysis
    # Rule of thirds lines
    third_w, third_h = width // 3, height // 3
    cv2.line(image, (third_w, 0), (third_w, height), (255, 255, 255), 2)
    cv2.line(image, (2 * third_w, 0), (2 * third_w, height), (255, 255, 255), 2)
    cv2.line(image, (0, third_h), (width, third_h), (255, 255, 255), 2)
    cv2.line(image, (0, 2 * third_h), (width, 2 * third_h), (255, 255, 255), 2)
    
    # Add leading lines
    cv2.line(image, (0, height), (width, third_h), (200, 200, 255), 3)
    cv2.line(image, (width, height), (0, third_h), (255, 200, 200), 3)
    
    # Add some interesting points at rule of thirds intersections
    intersections = [(third_w, third_h), (2 * third_w, third_h), 
                    (third_w, 2 * third_h), (2 * third_w, 2 * third_h)]
    
    for i, (x, y) in enumerate(intersections):
        color = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)][i]
        cv2.circle(image, (x, y), 15, color, -1)
        cv2.circle(image, (x, y), 15, (255, 255, 255), 2)
    
    # Add some texture and details
    # Random noise for texture
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add rectangular shapes
    cv2.rectangle(image, (50, 50), (150, 120), (180, 180, 180), 2)
    cv2.rectangle(image, (width - 150, height - 120), (width - 50, height - 50), (160, 160, 160), -1)
    
    # Save the image
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.jpg')
    
    cv2.imwrite(save_path, image)
    logger.info(f"Demo image created: {save_path}")
    return save_path


def demonstrate_stage_one():
    """
    Demonstrate the complete Stage One preprocessing pipeline.
    """
    print("\n" + "="*80)
    print("AI COMPOSITION ASSISTANT - STAGE ONE DEMONSTRATION")
    print("="*80)
    
    # Step 1: Create demo image
    print("\n1. Creating demo image with composition elements...")
    demo_image_path = create_demo_image()
    print(f"   ✓ Demo image created: {demo_image_path}")
    
    # Step 2: Initialize validator
    print("\n2. Initializing image validator...")
    validator = ImageValidator()
    print(f"   ✓ Validator initialized")
    print(f"   - Supported formats: {validator.supported_formats}")
    print(f"   - Size limits: {validator.min_size} to {validator.max_size}")
    
    # Step 3: Validate the demo image
    print("\n3. Validating demo image...")
    validation_results = validator.validate_file_path(demo_image_path)
    print("   Validation results:")
    for key, value in validation_results.items():
        status = "✓" if value else "✗"
        print(f"   {status} {key}: {value}")
    
    # Step 4: Initialize preprocessor
    print("\n4. Initializing image preprocessor...")
    config = {
        'target_size': (224, 224),
        'preserve_aspect_ratio': True,
        'noise_reduction': True
    }
    preprocessor = create_preprocessing_pipeline(config)
    print(f"   ✓ Preprocessor initialized with config: {config}")
    
    # Step 5: Load and analyze original image
    print("\n5. Loading and analyzing original image...")
    original_image = preprocessor.load_image(demo_image_path)
    print(f"   ✓ Original image loaded: {original_image.shape}")
    
    # Get image statistics
    stats = get_image_statistics(original_image)
    print("   Original image statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.2f}")
    
    # Check image quality
    quality = check_image_quality(original_image)
    print("   Image quality assessment:")
    for key, value in quality.items():
        status = "✓" if value else "✗"
        print(f"   {status} {key}: {value}")
    
    # Step 6: Run complete preprocessing pipeline
    print("\n6. Running complete preprocessing pipeline...")
    processed_image, features = preprocessor.preprocess(
        demo_image_path,
        return_features=True,
        color_space='RGB',
        normalization='standard'
    )
    print(f"   ✓ Image preprocessed: {processed_image.shape}")
    print(f"   ✓ Features extracted: {len(features)} feature types")
    
    # Step 7: Analyze extracted features
    print("\n7. Analyzing extracted features...")
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, np.ndarray):
            if feature_data.ndim == 2:  # 2D arrays (edges, gray)
                print(f"   - {feature_name}: {feature_data.shape} (2D array)")
            elif feature_data.ndim == 3:  # 3D arrays (lines, corners)
                print(f"   - {feature_name}: {feature_data.shape if len(feature_data) > 0 else 'empty'}")
            elif feature_data.ndim == 1:  # 1D arrays
                print(f"   - {feature_name}: {len(feature_data)} items")
        elif isinstance(feature_data, list):
            print(f"   - {feature_name}: {len(feature_data)} contours")
        else:
            print(f"   - {feature_name}: {type(feature_data)}")
    
    # Specific feature analysis
    if features['lines'] is not None and len(features['lines']) > 0:
        print(f"   ✓ Detected {len(features['lines'])} lines (for leading line analysis)")
    else:
        print("   - No lines detected")
    
    if features['corners'] is not None and len(features['corners']) > 0:
        print(f"   ✓ Detected {len(features['corners'])} corner points (for rule of thirds)")
    else:
        print("   - No corners detected")
    
    print(f"   ✓ Detected {len(features['contours'])} contours (for shape analysis)")
    
    # Step 8: Validate preprocessing output
    print("\n8. Validating preprocessing output...")
    output_validation = validator.validate_preprocessing_output(
        processed_image, features, expected_shape=(224, 224)
    )
    print("   Output validation results:")
    for key, value in output_validation.items():
        status = "✓" if value else "✗"
        print(f"   {status} {key}: {value}")
    
    # Step 9: Demonstrate batch processing
    print("\n9. Demonstrating batch processing...")
    # Create multiple demo images
    batch_paths = []
    for i in range(3):
        path = create_demo_image(save_path=tempfile.mktemp(suffix=f'_batch_{i}.jpg'))
        batch_paths.append(path)
    
    try:
        batch_images, batch_features = preprocessor.batch_preprocess(
            batch_paths, return_features=True
        )
        print(f"   ✓ Batch processed {batch_images.shape[0]} images")
        print(f"   ✓ Batch output shape: {batch_images.shape}")
        print(f"   ✓ Features extracted for {len(batch_features)} images")
    
    finally:
        # Cleanup batch files
        for path in batch_paths:
            if os.path.exists(path):
                os.unlink(path)
    
    # Step 10: Performance summary
    print("\n10. Stage One completion summary...")
    print("   ✓ Image acquisition and loading")
    print("   ✓ Noise reduction and filtering") 
    print("   ✓ Color space conversion and optimization")
    print("   ✓ Normalization and standardization")
    print("   ✓ Feature extraction (edges, lines, corners, contours)")
    print("   ✓ Validation and quality control")
    print("   ✓ Batch processing capabilities")
    
    print(f"\n   📊 Final processed image shape: {processed_image.shape}")
    print(f"   📊 Pixel value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    print(f"   📊 Feature extraction completed: {len(features)} types")
    
    # Cleanup
    if os.path.exists(demo_image_path):
        os.unlink(demo_image_path)
    
    print("\n" + "="*80)
    print("STAGE ONE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("Ready to proceed to Stage Two: Feature Detection and Extraction")
    print("="*80)


if __name__ == "__main__":
    try:
        demonstrate_stage_one()
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)