#!/usr/bin/env python3
"""
Stage Two Demonstration: Feature Detection and Extraction

This script demonstrates the complete Stage Two pipeline for the AI Composition Assistant,
showcasing advanced feature detection using hybrid CNN-ViT architecture and specialized
detectors for compositional elements.
"""

import os
import sys
import numpy as np
import cv2
import torch
import logging
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_net import HybridCompositionNet
from models.feature_detectors import (
    RuleOfThirdsDetector,
    LeadingLinesDetector,
    SymmetryDetector,
    DepthAnalyzer
)
from preprocessing import ImagePreprocessor, create_preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_image(width=400, height=300, save_path=None):
    """
    Create a demo image with various composition elements for testing.
    
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
    
    # Add symmetric elements
    cv2.circle(image, (width//2, height//2), min(width, height)//4, (200, 200, 200), -1)
    cv2.rectangle(image, (width//4, height//4), (3*width//4, 3*height//4), (180, 180, 180), 2)
    
    # Add leading lines
    cv2.line(image, (0, height), (width//2, 0), (255, 200, 200), 3)
    cv2.line(image, (width, height), (width//2, 0), (200, 255, 200), 3)
    
    # Add rule of thirds elements
    third_w, third_h = width // 3, height // 3
    for i in range(2):
        x = (i + 1) * third_w
        for j in range(2):
            y = (j + 1) * third_h
            cv2.circle(image, (x, y), 10, (255, 255, 255), -1)
    
    # Add depth elements
    # Foreground
    cv2.rectangle(image, (50, height-100), (150, height-50), (255, 255, 255), -1)
    # Middle ground
    polygon_pts = np.array([[width//2-50, height//2-30], 
                           [width//2+50, height//2-30],
                           [width//2+30, height//2+30],
                           [width//2-30, height//2+30]], dtype=np.int32)
    cv2.fillPoly(image, [polygon_pts], (200, 200, 200))  # Changed from polygon to fillPoly
    # Background
    cv2.circle(image, (width-100, 100), 30, (150, 150, 150), -1)
    
    # Save the image
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.jpg')
    
    cv2.imwrite(save_path, image)
    logger.info(f"Demo image created: {save_path}")
    return save_path


def visualize_results(image, results, save_path=None):
    """
    Visualize detection results.
    
    Args:
        image: Input image
        results: Dictionary containing detection results
        save_path: Path to save visualization (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    
    # Rule of thirds
    rot_vis = image.copy()
    for point, score in zip(results['rule_of_thirds']['points'], 
                          results['rule_of_thirds']['scores']):
        cv2.circle(rot_vis, point, int(10 * score), (0, 255, 0), -1)
    axes[1].imshow(cv2.cvtColor(rot_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Rule of Thirds Detection')
    
    # Leading lines
    lines_vis = image.copy()
    if results['leading_lines']['lines']:
        for line, score in zip(results['leading_lines']['lines'],
                             results['leading_lines']['scores']):
            x1, y1, x2, y2 = line
            color = (0, int(255 * score), 0)
            cv2.line(lines_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    axes[2].imshow(cv2.cvtColor(lines_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Leading Lines Detection')
    
    # Depth map
    depth_map = results['depth']['depth_map']
    depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    axes[3].imshow(depth_vis, cmap='viridis')
    axes[3].set_title('Depth Analysis')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Visualization saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def demonstrate_stage_two():
    """
    Demonstrate the complete Stage Two feature detection pipeline.
    """
    print("\n" + "="*80)
    print("AI COMPOSITION ASSISTANT - STAGE TWO DEMONSTRATION")
    print("="*80)
    
    # Step 1: Create demo image
    print("\n1. Creating demo image with composition elements...")
    demo_image_path = create_demo_image()
    print(f"   ✓ Demo image created: {demo_image_path}")
    
    # Step 2: Initialize models and detectors
    print("\n2. Initializing models and detectors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize hybrid model
    hybrid_net = HybridCompositionNet().to(device)
    print(f"   ✓ Hybrid CNN-ViT model initialized on {device}")
    
    # Initialize specialized detectors
    rot_detector = RuleOfThirdsDetector()
    lines_detector = LeadingLinesDetector()
    symmetry_detector = SymmetryDetector()
    depth_analyzer = DepthAnalyzer()
    
    print("   ✓ Specialized detectors initialized:")
    print("     - Rule of Thirds Detector")
    print("     - Leading Lines Detector")
    print("     - Symmetry Detector")
    print("     - Depth Analyzer")
    
    # Step 3: Load and preprocess image
    print("\n3. Loading and preprocessing image...")
    preprocessor = create_preprocessing_pipeline({'target_size': (224, 224)})
    image = cv2.imread(demo_image_path)
    processed_image, features = preprocessor.preprocess(demo_image_path, return_features=True)
    print(f"   ✓ Image preprocessed to shape: {processed_image.shape}")
    print(f"   ✓ Extracted {len(features)} feature types")
    
    # Step 4: Run hybrid model inference
    print("\n4. Running hybrid model inference...")
    with torch.no_grad():
        model_input = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
        model_input = model_input.float().to(device)
        hybrid_results = hybrid_net(model_input)
    print("   ✓ Hybrid model inference completed")
    
    # Step 5: Run specialized detectors
    print("\n5. Running specialized detectors...")
    
    # Rule of thirds detection
    try:
        rot_results = rot_detector.detect(image)
        print("   ✓ Rule of thirds detection completed:")
        print(f"     - Found {len(rot_results['points'])} intersection points")
    except Exception as e:
        print(f"   ✗ Rule of thirds detection failed: {str(e)}")
        rot_results = {'points': [], 'scores': []}
    
    # Leading lines detection
    try:
        lines_results = lines_detector.detect(image)
        print("   ✓ Leading lines detection completed:")
        print(f"     - Detected {len(lines_results['lines'])} lines")
        if lines_results['vanishing_points']:
            print(f"     - Found {len(lines_results['vanishing_points'])} vanishing points")
    except Exception as e:
        print(f"   ✗ Leading lines detection failed: {str(e)}")
        lines_results = {'lines': [], 'scores': [], 'vanishing_points': []}
    
    # Symmetry detection
    try:
        symmetry_results = symmetry_detector.detect(image)
        print("   ✓ Symmetry detection completed:")
        print(f"     - Dominant type: {symmetry_results['dominant_type']}")
        print(f"     - Horizontal score: {symmetry_results['horizontal_score']:.3f}")
        print(f"     - Vertical score: {symmetry_results['vertical_score']:.3f}")
        print(f"     - Radial score: {symmetry_results['radial_score']:.3f}")
    except Exception as e:
        print(f"   ✗ Symmetry detection failed: {str(e)}")
        symmetry_results = {'horizontal_score': 0.0, 'vertical_score': 0.0, 'radial_score': 0.0, 'dominant_type': None}
    
    # Depth analysis
    try:
        depth_results = depth_analyzer.analyze(image)
        print("   ✓ Depth analysis completed:")
        print(f"     - Detected {len(depth_results['layers'])} depth layers")
        print(f"     - Found {len(depth_results['focal_points'])} focal points")
    except Exception as e:
        print(f"   ✗ Depth analysis failed: {str(e)}")
        depth_results = {'layers': [], 'focal_points': [], 'depth_map': np.zeros((100, 100)), 'depth_statistics': {'std_depth': 1.0}}
    
    # Step 6: Combine and analyze results
    print("\n6. Analyzing combined results...")
    combined_results = {
        'rule_of_thirds': rot_results,
        'leading_lines': lines_results,
        'symmetry': symmetry_results,
        'depth': depth_results
    }
    
    # Calculate overall composition score
    try:
        rot_score = np.mean(rot_results['scores']) if rot_results['scores'] else 0.0
        lines_score = np.mean(lines_results['scores']) if lines_results['scores'] else 0.0
        symmetry_score = max(symmetry_results['horizontal_score'],
                           symmetry_results['vertical_score'],
                           symmetry_results['radial_score'])
        depth_score = (1.0 / (1.0 + depth_results['depth_statistics']['std_depth']))
        
        composition_score = (
            rot_score * 0.3 +  # Rule of thirds weight
            lines_score * 0.3 +  # Leading lines weight
            symmetry_score * 0.2 +  # Symmetry weight
            depth_score * 0.2  # Depth weight
        )
    except Exception as e:
        print(f"   Warning: Score calculation failed: {str(e)}")
        composition_score = 0.0
    
    print(f"   ✓ Overall composition score: {composition_score:.3f}")
    
    # Step 7: Visualize results
    print("\n7. Visualizing results...")
    vis_path = tempfile.mktemp(suffix='_results.png')
    visualize_results(image, combined_results, vis_path)
    print(f"   ✓ Results visualization saved: {vis_path}")
    
    # Step 8: Generate composition suggestions
    print("\n8. Generating composition suggestions...")
    suggestions = []
    
    # Rule of thirds suggestions
    if np.mean(rot_results['scores']) < 0.3:
        suggestions.append("Consider placing key elements at rule of thirds intersections")
    
    # Leading lines suggestions
    if len(lines_results['lines']) < 2:
        suggestions.append("Look for or create leading lines to guide viewer attention")
    elif not lines_results['vanishing_points']:
        suggestions.append("Try to align leading lines toward a common vanishing point")
    
    # Symmetry suggestions
    if max(symmetry_results['horizontal_score'],
           symmetry_results['vertical_score'],
           symmetry_results['radial_score']) < 0.4:
        suggestions.append("Consider incorporating more symmetrical elements")
    
    # Depth suggestions
    if depth_results['depth_statistics']['depth_range'] < 0.3:
        suggestions.append("Add more depth layers to create better spatial separation")
    if len(depth_results['focal_points']) < 2:
        suggestions.append("Create more distinct focal points at different depths")
    
    print("   Composition suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    # Cleanup
    if os.path.exists(demo_image_path):
        os.unlink(demo_image_path)
    if os.path.exists(vis_path):
        os.unlink(vis_path)
    
    print("\n" + "="*80)
    print("STAGE TWO DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    try:
        demonstrate_stage_two()
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)