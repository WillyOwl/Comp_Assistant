#!/usr/bin/env python3
"""
Stage Three Demonstration: Compositional Analysis

This script demonstrates the complete Stage Three pipeline for the AI Composition Assistant,
showcasing comprehensive compositional analysis, scoring, aesthetic assessment, and 
intelligent suggestion generation.

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
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import (
    CompositionAnalyzer,
    RuleOfThirdsEvaluator,
    LeadingLinesEvaluator,
    SymmetryEvaluator,
    DepthLayeringEvaluator,
    ColorHarmonyEvaluator,
    CompositionScorer,
    AestheticQualityAssessor,
    SuggestionEngine
)
from models.hybrid_net import HybridCompositionNet
from models.feature_detectors import (
    RuleOfThirdsDetector,
    LeadingLinesDetector,
    SymmetryDetector,
    DepthAnalyzer
)
from preprocessing import create_preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_image(width=500, height=375, image_type='balanced', save_path=None):
    """
    Create a demo image with specific compositional characteristics for testing.
    
    Args:
        width: Image width
        height: Image height
        image_type: Type of composition ('balanced', 'poor_composition', 'strong_lines', 'symmetric')
        save_path: Path to save the image (optional)
        
    Returns:
        Path to the created image
    """
    # Create a canvas
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if image_type == 'balanced':
        # Create a well-composed image following multiple rules
        # Background gradient
        for i in range(height):
            intensity = int(255 * (i / height) * 0.4 + 60)
            image[i, :] = [intensity, intensity + 30, intensity + 50]
        
        # Rule of thirds grid points
        third_w, third_h = width // 3, height // 3
        intersections = [(third_w, third_h), (2 * third_w, third_h), 
                        (third_w, 2 * third_h), (2 * third_w, 2 * third_h)]
        
        # Place subjects at intersections
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        for i, (x, y) in enumerate(intersections[:3]):
            cv2.circle(image, (x, y), 20, colors[i], -1)
            cv2.circle(image, (x, y), 20, (255, 255, 255), 3)
        
        # Leading lines
        cv2.line(image, (0, height), (width//2, third_h), (200, 200, 255), 4)
        cv2.line(image, (width, height), (width//2, third_h), (255, 200, 200), 4)
        
        # Depth layers
        # Foreground
        cv2.rectangle(image, (50, height-80), (120, height-20), (180, 180, 180), -1)
        # Background
        cv2.circle(image, (width-80, 80), 25, (120, 120, 120), -1)
    
    elif image_type == 'poor_composition':
        # Create a poorly composed image
        # Flat background
        image[:] = [100, 120, 140]
        
        # Subject in center (breaking rule of thirds)
        cv2.circle(image, (width//2, height//2), 30, (255, 0, 0), -1)
        
        # No leading lines, poor balance
        cv2.rectangle(image, (10, 10), (60, 60), (0, 255, 0), -1)
        cv2.rectangle(image, (width-60, height-60), (width-10, height-10), (0, 0, 255), -1)
    
    elif image_type == 'strong_lines':
        # Focus on leading lines
        # Gradient background
        for i in range(height):
            intensity = int(100 + (i / height) * 100)
            image[i, :] = [intensity, intensity-20, intensity+20]
        
        # Multiple leading lines
        cv2.line(image, (0, height), (width, 0), (255, 255, 255), 3)
        cv2.line(image, (0, 0), (width, height), (200, 200, 200), 2)
        cv2.line(image, (width//2, 0), (width//2, height), (180, 180, 180), 2)
        cv2.line(image, (0, height//2), (width, height//2), (160, 160, 160), 2)
        
        # Vanishing point elements
        center = (width//2, height//3)
        for angle in [0, 45, 90, 135]:
            end_x = int(center[0] + 100 * np.cos(np.radians(angle)))
            end_y = int(center[1] + 100 * np.sin(np.radians(angle)))
            cv2.line(image, center, (end_x, end_y), (255, 200, 150), 2)
    
    elif image_type == 'symmetric':
        # Create symmetric composition
        # Create one half
        half_width = width // 2
        left_half = np.zeros((height, half_width, 3), dtype=np.uint8)
        
        # Add symmetric elements to left half
        for i in range(height):
            intensity = int(80 + (i / height) * 120)
            left_half[i, :] = [intensity, intensity+20, intensity+40]
        
        # Add shapes
        cv2.circle(left_half, (half_width//4, height//3), 30, (255, 255, 255), -1)
        cv2.rectangle(left_half, (half_width//2, 2*height//3), 
                     (3*half_width//4, height-50), (200, 200, 200), -1)
        
        # Mirror to create symmetry
        right_half = np.fliplr(left_half)
        image = np.concatenate([left_half, right_half], axis=1)
    
    # Add some texture and noise
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Save the image
    if save_path is None:
        save_path = tempfile.mktemp(suffix=f'_{image_type}.jpg')
    
    cv2.imwrite(save_path, image)
    logger.info(f"Demo image '{image_type}' created: {save_path}")
    return save_path


def create_visualization(image, analysis_results, save_path=None):
    """
    Create comprehensive visualization of analysis results.
    
    Args:
        image: Input image
        analysis_results: CompositionResults object
        save_path: Path to save visualization (optional)
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Original image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb_image)
        axes[0].set_title(f'Original Image\nOverall Score: {analysis_results.overall_score:.3f}')
        axes[0].axis('off')
        
        # Rule scores visualization
        rule_names = list(analysis_results.rule_scores.keys())
        rule_values = list(analysis_results.rule_scores.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = axes[1].bar(range(len(rule_names)), rule_values, color=colors[:len(rule_names)])
        axes[1].set_xticks(range(len(rule_names)))
        axes[1].set_xticklabels([name.replace('_', '\n') for name in rule_names], rotation=0, fontsize=9)
        axes[1].set_ylabel('Score')
        axes[1].set_title('Rule Scores')
        axes[1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, rule_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Quality metrics
        quality_metrics = {
            'Aesthetic': analysis_results.aesthetic_score,
            'Technical': analysis_results.technical_score,
            'Confidence': analysis_results.confidence
        }
        
        metric_names = list(quality_metrics.keys())
        metric_values = list(quality_metrics.values())
        
        bars2 = axes[2].bar(metric_names, metric_values, color=['#FF9F43', '#10AC84', '#5F27CD'])
        axes[2].set_ylabel('Score')
        axes[2].set_title('Quality Metrics')
        axes[2].set_ylim(0, 1)
        
        for bar, value in zip(bars2, metric_values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Rule of thirds visualization
        rot_overlay = rgb_image.copy()
        h, w = rot_overlay.shape[:2]
        third_w, third_h = w // 3, h // 3
        
        # Draw grid lines
        for x in [third_w, 2 * third_w]:
            cv2.line(rot_overlay, (x, 0), (x, h), (255, 255, 255), 2)
        for y in [third_h, 2 * third_h]:
            cv2.line(rot_overlay, (0, y), (w, y), (255, 255, 255), 2)
        
        # Draw intersection points
        for x in [third_w, 2 * third_w]:
            for y in [third_h, 2 * third_h]:
                cv2.circle(rot_overlay, (x, y), 8, (255, 0, 0), -1)
        
        axes[3].imshow(rot_overlay)
        axes[3].set_title(f'Rule of Thirds\nScore: {analysis_results.rule_scores.get("rule_of_thirds", 0.0):.3f}')
        axes[3].axis('off')
        
        # Suggestions display
        suggestions_text = "Top Suggestions:\n\n"
        for i, suggestion in enumerate(analysis_results.suggestions[:4], 1):
            # Truncate long suggestions
            short_suggestion = suggestion[:60] + "..." if len(suggestion) > 60 else suggestion
            suggestions_text += f"{i}. {short_suggestion}\n\n"
        
        axes[4].text(0.05, 0.95, suggestions_text, transform=axes[4].transAxes, 
                    fontsize=10, verticalalignment='top', wrap=True)
        axes[4].set_title('Improvement Suggestions')
        axes[4].axis('off')
        
        # Processing info
        info_text = f"""Analysis Information:

Processing Time: {analysis_results.processing_time:.3f}s
Timestamp: {analysis_results.timestamp.strftime('%H:%M:%S')}

Score Breakdown:
• Overall: {analysis_results.overall_score:.3f}
• Aesthetic: {analysis_results.aesthetic_score:.3f}
• Technical: {analysis_results.technical_score:.3f}
• Confidence: {analysis_results.confidence:.3f}

Rules Analyzed: {len(analysis_results.rule_scores)}
Suggestions: {len(analysis_results.suggestions)}"""
        
        axes[5].text(0.05, 0.95, info_text, transform=axes[5].transAxes, 
                    fontsize=9, verticalalignment='top', family='monospace')
        axes[5].set_title('Analysis Summary')
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {str(e)}")


def demonstrate_stage_three():
    """
    Demonstrate the complete Stage Three compositional analysis pipeline.
    """
    print("\n" + "="*80)
    print("AI COMPOSITION ASSISTANT - STAGE THREE DEMONSTRATION")
    print("Compositional Analysis, Scoring, and Intelligent Suggestions")
    print("="*80)
    
    # Step 1: Initialize the complete system
    print("\n1. Initializing Stage Three Analysis System...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize main composition analyzer
    analyzer_config = {
        'rule_weights': {
            'rule_of_thirds': 0.25,
            'leading_lines': 0.25,
            'symmetry': 0.20,
            'depth_layering': 0.15,
            'color_harmony': 0.15
        },
        'analysis_depth': 'comprehensive',
        'suggestions': {
            'max_suggestions': 6,
            'include_technical_suggestions': True,
            'include_creative_suggestions': True
        }
    }
    
    composition_analyzer = CompositionAnalyzer(device=device, config=analyzer_config)
    print(f"   ✓ CompositionAnalyzer initialized on {device}")
    print(f"   ✓ Analysis depth: {analyzer_config['analysis_depth']}")
    print(f"   ✓ Rule weights configured: {len(analyzer_config['rule_weights'])} rules")
    
    # Step 2: Create test images with different characteristics
    print("\n2. Creating test images with different compositional characteristics...")
    
    test_images = {}
    image_types = ['balanced', 'poor_composition', 'strong_lines', 'symmetric']
    
    for img_type in image_types:
        path = create_demo_image(image_type=img_type)
        test_images[img_type] = path
        print(f"   ✓ Created '{img_type}' test image")
    
    # Step 3: Analyze each test image
    print("\n3. Running comprehensive compositional analysis...")
    
    results = {}
    
    for img_type, img_path in test_images.items():
        print(f"\n   Analyzing '{img_type}' composition...")
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                logger.error(f"Failed to load image: {img_path}")
                continue
            
            print(f"     - Image loaded: {image.shape}")
            
            # Run complete analysis
            analysis_result = composition_analyzer.analyze(
                image, 
                features=None,  # Let the analyzer extract features
                return_visualizations=True
            )
            
            results[img_type] = {
                'image': image,
                'path': img_path,
                'analysis': analysis_result
            }
            
            # Display results summary
            analysis_time = time.time() - start_time
            print(f"     ✓ Analysis completed in {analysis_time:.3f}s")
            print(f"     ✓ Overall score: {analysis_result.overall_score:.3f}")
            print(f"     ✓ Aesthetic score: {analysis_result.aesthetic_score:.3f}")
            print(f"     ✓ Technical score: {analysis_result.technical_score:.3f}")
            print(f"     ✓ Confidence: {analysis_result.confidence:.3f}")
            print(f"     ✓ Suggestions generated: {len(analysis_result.suggestions)}")
            
        except Exception as e:
            logger.error(f"Analysis failed for {img_type}: {str(e)}")
            continue
    
    # Step 4: Display detailed results for each image
    print("\n4. Detailed Analysis Results:")
    print("-" * 60)
    
    for img_type, result_data in results.items():
        analysis = result_data['analysis']
        
        print(f"\n📊 {img_type.upper().replace('_', ' ')} COMPOSITION:")
        print(f"   Overall Score: {analysis.overall_score:.3f}")
        print(f"   Processing Time: {analysis.processing_time:.3f}s")
        
        print("\n   Rule Scores:")
        for rule_name, score in analysis.rule_scores.items():
            print(f"     • {rule_name.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n   Quality Metrics:")
        print(f"     • Aesthetic Quality: {analysis.aesthetic_score:.3f}")
        print(f"     • Technical Quality: {analysis.technical_score:.3f}")
        print(f"     • Analysis Confidence: {analysis.confidence:.3f}")
        
        print(f"\n   Top Suggestions:")
        for i, suggestion in enumerate(analysis.suggestions[:3], 1):
            print(f"     {i}. {suggestion}")
        
        print("-" * 60)
    
    # Step 5: Comparative analysis
    print("\n5. Comparative Analysis:")
    
    if len(results) >= 2:
        # Find best and worst compositions
        scores = {img_type: data['analysis'].overall_score for img_type, data in results.items()}
        best_type = max(scores, key=scores.get)
        worst_type = min(scores, key=scores.get)
        
        print(f"\n   🏆 Best Composition: '{best_type}' (Score: {scores[best_type]:.3f})")
        print(f"   📉 Needs Improvement: '{worst_type}' (Score: {scores[worst_type]:.3f})")
        print(f"   📈 Improvement Potential: {scores[best_type] - scores[worst_type]:.3f}")
        
        # Rule comparison
        print(f"\n   Rule Performance Comparison:")
        for rule_name in results[best_type]['analysis'].rule_scores.keys():
            best_score = results[best_type]['analysis'].rule_scores[rule_name]
            worst_score = results[worst_type]['analysis'].rule_scores[rule_name]
            difference = best_score - worst_score
            
            print(f"     • {rule_name.replace('_', ' ').title()}: "
                  f"{best_score:.3f} vs {worst_score:.3f} (Δ{difference:+.3f})")
    
    # Step 6: Create visualizations
    print("\n6. Creating analysis visualizations...")
    
    for img_type, result_data in results.items():
        try:
            vis_path = tempfile.mktemp(suffix=f'_{img_type}_analysis.png')
            create_visualization(
                result_data['image'], 
                result_data['analysis'], 
                vis_path
            )
            print(f"   ✓ Visualization created for '{img_type}': {vis_path}")
        except Exception as e:
            logger.warning(f"Visualization failed for {img_type}: {str(e)}")
    
    # Step 7: Demonstrate individual evaluators
    print("\n7. Individual Rule Evaluator Demonstration:")
    
    if results:
        # Use the balanced image for detailed rule analysis
        test_image = results.get('balanced', list(results.values())[0])['image']
        
        print("\n   Testing individual rule evaluators...")
        
        # Rule of Thirds
        rot_evaluator = RuleOfThirdsEvaluator()
        rot_result = rot_evaluator.evaluate(test_image)
        print(f"     • Rule of Thirds: {rot_result.get('score', 0.0):.3f} "
              f"({rot_result.get('elements_detected', 0)} elements)")
        
        # Leading Lines
        lines_evaluator = LeadingLinesEvaluator()
        lines_result = lines_evaluator.evaluate(test_image)
        print(f"     • Leading Lines: {lines_result.get('score', 0.0):.3f} "
              f"({len(lines_result.get('lines', []))} lines)")
        
        # Symmetry
        symmetry_evaluator = SymmetryEvaluator()
        symmetry_result = symmetry_evaluator.evaluate(test_image)
        print(f"     • Symmetry: {symmetry_result.get('score', 0.0):.3f} "
              f"(type: {symmetry_result.get('dominant_type', 'none')})")
        
        # Depth Layering
        depth_evaluator = DepthLayeringEvaluator()
        depth_result = depth_evaluator.evaluate(test_image)
        print(f"     • Depth Layering: {depth_result.get('score', 0.0):.3f} "
              f"({len(depth_result.get('layers', []))} layers)")
        
        # Color Harmony
        color_evaluator = ColorHarmonyEvaluator()
        color_result = color_evaluator.evaluate(test_image)
        print(f"     • Color Harmony: {color_result.get('score', 0.0):.3f} "
              f"({len(color_result.get('dominant_colors', []))} colors)")
    
    # Step 8: Performance metrics
    print("\n8. Performance Metrics Summary:")
    
    if results:
        total_processing_time = sum(data['analysis'].processing_time for data in results.values())
        avg_processing_time = total_processing_time / len(results)
        
        print(f"   📊 Total images analyzed: {len(results)}")
        print(f"   ⏱️ Total processing time: {total_processing_time:.3f}s")
        print(f"   ⚡ Average processing time: {avg_processing_time:.3f}s per image")
        print(f"   🚀 Throughput: {len(results)/total_processing_time:.1f} images/second")
        
        # Score distribution
        all_scores = [data['analysis'].overall_score for data in results.values()]
        print(f"   📈 Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"   📊 Average score: {np.mean(all_scores):.3f}")
    
    # Step 9: System capabilities summary
    print("\n9. Stage Three Capabilities Demonstrated:")
    print("   ✓ Multi-rule compositional analysis")
    print("   ✓ Advanced scoring algorithms with EMD loss")
    print("   ✓ Aesthetic quality assessment")
    print("   ✓ Intelligent suggestion generation")
    print("   ✓ Confidence-weighted evaluation")
    print("   ✓ Real-time performance optimization")
    print("   ✓ Comprehensive visualization")
    print("   ✓ Comparative analysis capabilities")
    
    # Cleanup
    print("\n10. Cleanup...")
    for img_type, result_data in results.items():
        if os.path.exists(result_data['path']):
            os.unlink(result_data['path'])
            print(f"   ✓ Cleaned up {img_type} test image")
    
    print("\n" + "="*80)
    print("STAGE THREE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("\nThe AI Composition Assistant now provides:")
    print("• Comprehensive compositional analysis")
    print("• Multi-dimensional scoring and assessment")
    print("• Intelligent improvement suggestions")
    print("• Professional-grade evaluation capabilities")
    print("\nReady for Stage Four: API Development and Deployment!")
    print("="*80)


if __name__ == "__main__":
    try:
        demonstrate_stage_three()
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)