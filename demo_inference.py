#!/usr/bin/env python3
"""
Composition Analysis Demo Script

This script demonstrates the composition analysis model by:
1. Loading a trained model or using a randomly initialized one
2. Processing input images to analyze composition elements
3. Visualizing results with overlays and annotations

Usage:
    python demo_inference.py --image path/to/image.jpg
    python demo_inference.py --image path/to/image.jpg --model path/to/model.pth
    python demo_inference.py --batch path/to/images/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_net import HybridCompositionNet
from training.dataset_loader import get_composition_transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompositionAnalyzer:
    """
    Main class for composition analysis inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the composition analyzer.
        
        Args:
            model_path: Path to trained model weights, None for random initialization
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Image preprocessing
        self.transform = self._get_transforms()
        
    def _load_model(self, model_path: Optional[str]) -> HybridCompositionNet:
        """Load model with optional pretrained weights."""
        
        # Model configuration (matching training config)
        model_config = {
            'img_size': 224,
            'patch_size': 16,
            'num_channels': 3,
            'hidden_size': 384,
            'num_attention_heads': 6,
            'num_hidden_layers': 6,
            'backbone': 'resnet50'
        }
        
        model = HybridCompositionNet(**model_config)
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading trained weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
        else:
            if model_path:
                logger.warning(f"Model path {model_path} not found, using random initialization")
            else:
                logger.info("Using randomly initialized model for demonstration")
                
        return model.to(self.device)
    
    def _get_transforms(self) -> A.Compose:
        """Get image preprocessing transforms."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze composition of a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
            
        # Process predictions
        results = self._process_predictions(predictions, original_shape)
        results['original_image'] = image_rgb
        results['image_path'] = image_path
        
        return results
    
    def _process_predictions(self, predictions: Dict[str, torch.Tensor], 
                           original_shape: Tuple[int, int]) -> Dict:
        """Process model predictions into interpretable results."""
        
        results = {}
        
        # Rule of thirds analysis
        if 'rule_of_thirds' in predictions:
            rot_logits = predictions['rule_of_thirds'].cpu().squeeze()
            rot_probs = torch.sigmoid(rot_logits).numpy()
            
            # Convert to 3x3 grid
            rot_grid = rot_probs.reshape(3, 3)
            results['rule_of_thirds'] = {
                'grid_probabilities': rot_grid,
                'strong_points': np.where(rot_grid > 0.5),
                'max_strength': float(rot_grid.max()),
                'total_strength': float(rot_grid.sum())
            }
        
        # Leading lines analysis
        if 'leading_lines' in predictions:
            lines_pred = predictions['leading_lines'].cpu().squeeze().numpy()
            
            # Extract line parameters (x1, y1, x2, y2, strength)
            if len(lines_pred) >= 5:
                x1, y1, x2, y2, strength = lines_pred[:5]
                
                # Convert normalized coordinates to image coordinates
                h, w = original_shape
                results['leading_lines'] = {
                    'line_coords': (
                        int(x1 * w), int(y1 * h),
                        int(x2 * w), int(y2 * h)
                    ),
                    'strength': float(strength),
                    'angle': float(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                }
        
        # Symmetry analysis
        if 'symmetry' in predictions:
            symmetry_probs = predictions['symmetry'].cpu().squeeze().numpy()
            symmetry_types = ['none', 'horizontal', 'vertical', 'radial']
            
            results['symmetry'] = {
                'probabilities': {
                    symmetry_types[i]: float(prob) 
                    for i, prob in enumerate(symmetry_probs)
                },
                'dominant_type': symmetry_types[np.argmax(symmetry_probs)],
                'confidence': float(np.max(symmetry_probs))
            }
        
        # Depth analysis
        if 'depth' in predictions:
            depth_score = predictions['depth'].cpu().squeeze().item()
            results['depth'] = {
                'score': float(depth_score),
                'category': self._categorize_depth(depth_score)
            }
            
        return results
    
    def _categorize_depth(self, score: float) -> str:
        """Categorize depth score into human-readable categories."""
        if score < 0.3:
            return "Flat/Low depth"
        elif score < 0.6:
            return "Moderate depth"
        else:
            return "High depth/Strong perspective"
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None, 
                         show: bool = True) -> None:
        """
        Create visualization of composition analysis results.
        
        Args:
            results: Analysis results from analyze_image()
            save_path: Optional path to save visualization
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Composition Analysis: {Path(results['image_path']).name}", 
                    fontsize=16, fontweight='bold')
        
        # Original image
        ax1 = axes[0, 0]
        ax1.imshow(results['original_image'])
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Rule of thirds overlay
        ax2 = axes[0, 1]
        ax2.imshow(results['original_image'])
        self._draw_rule_of_thirds(ax2, results.get('rule_of_thirds', {}))
        ax2.set_title("Rule of Thirds Analysis")
        ax2.axis('off')
        
        # Leading lines and symmetry
        ax3 = axes[1, 0]
        ax3.imshow(results['original_image'])
        self._draw_leading_lines(ax3, results.get('leading_lines', {}))
        self._draw_symmetry_guides(ax3, results.get('symmetry', {}))
        ax3.set_title("Leading Lines & Symmetry")
        ax3.axis('off')
        
        # Analysis summary
        ax4 = axes[1, 1]
        self._draw_analysis_summary(ax4, results)
        ax4.set_title("Analysis Summary")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def _draw_rule_of_thirds(self, ax, rot_data: Dict) -> None:
        """Draw rule of thirds grid and highlight strong points."""
        if not rot_data:
            return
            
        h, w = ax.get_ylim()[0], ax.get_xlim()[1]
        
        # Draw grid lines
        for i in range(1, 3):
            ax.axvline(x=w * i / 3, color='white', linestyle='--', alpha=0.8, linewidth=2)
            ax.axhline(y=h * i / 3, color='white', linestyle='--', alpha=0.8, linewidth=2)
        
        # Highlight strong points
        grid = rot_data.get('grid_probabilities', np.zeros((3, 3)))
        for i in range(3):
            for j in range(3):
                if grid[i, j] > 0.5:
                    # Calculate circle position
                    x = w * (j + 0.5) / 3
                    y = h * (i + 0.5) / 3
                    
                    # Draw circle with intensity based on strength
                    radius = 20 * grid[i, j]
                    circle = patches.Circle((x, y), radius, 
                                          color='red', alpha=0.7)
                    ax.add_patch(circle)
    
    def _draw_leading_lines(self, ax, lines_data: Dict) -> None:
        """Draw detected leading lines."""
        if not lines_data or lines_data.get('strength', 0) < 0.3:
            return
            
        coords = lines_data.get('line_coords')
        if coords:
            x1, y1, x2, y2 = coords
            ax.plot([x1, x2], [y1, y2], 'lime', linewidth=3, alpha=0.8)
            ax.plot([x1, x2], [y1, y2], 'darkgreen', linewidth=1)
            
            # Add arrow to show direction
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='lime', lw=2))
    
    def _draw_symmetry_guides(self, ax, symmetry_data: Dict) -> None:
        """Draw symmetry axis if detected."""
        if not symmetry_data or symmetry_data.get('confidence', 0) < 0.6:
            return
            
        dominant_type = symmetry_data.get('dominant_type', 'none')
        h, w = ax.get_ylim()[0], ax.get_xlim()[1]
        
        if dominant_type == 'horizontal':
            ax.axhline(y=h/2, color='cyan', linestyle='-', alpha=0.8, linewidth=3)
        elif dominant_type == 'vertical':
            ax.axvline(x=w/2, color='cyan', linestyle='-', alpha=0.8, linewidth=3)
        elif dominant_type == 'radial':
            center = (w/2, h/2)
            circle = patches.Circle(center, min(w, h)/4, 
                                  fill=False, color='cyan', linewidth=3, alpha=0.8)
            ax.add_patch(circle)
    
    def _draw_analysis_summary(self, ax, results: Dict) -> None:
        """Draw text summary of analysis results."""
        ax.axis('off')
        
        summary_text = []
        
        # Rule of thirds
        rot_data = results.get('rule_of_thirds', {})
        if rot_data:
            max_strength = rot_data.get('max_strength', 0)
            summary_text.append(f"Rule of Thirds Strength: {max_strength:.2f}")
            if max_strength > 0.5:
                summary_text.append("✓ Strong rule of thirds composition")
            else:
                summary_text.append("• Weak rule of thirds alignment")
        
        summary_text.append("")
        
        # Leading lines
        lines_data = results.get('leading_lines', {})
        if lines_data:
            strength = lines_data.get('strength', 0)
            angle = lines_data.get('angle', 0)
            summary_text.append(f"Leading Lines Strength: {strength:.2f}")
            summary_text.append(f"Line Angle: {angle:.1f}°")
            if strength > 0.5:
                summary_text.append("✓ Strong leading lines detected")
            else:
                summary_text.append("• Weak or no leading lines")
        
        summary_text.append("")
        
        # Symmetry
        symmetry_data = results.get('symmetry', {})
        if symmetry_data:
            dominant = symmetry_data.get('dominant_type', 'none')
            confidence = symmetry_data.get('confidence', 0)
            summary_text.append(f"Symmetry Type: {dominant.capitalize()}")
            summary_text.append(f"Confidence: {confidence:.2f}")
            if confidence > 0.6 and dominant != 'none':
                summary_text.append("✓ Strong symmetrical composition")
            else:
                summary_text.append("• Asymmetrical composition")
        
        summary_text.append("")
        
        # Depth
        depth_data = results.get('depth', {})
        if depth_data:
            score = depth_data.get('score', 0)
            category = depth_data.get('category', 'Unknown')
            summary_text.append(f"Depth Score: {score:.2f}")
            summary_text.append(f"Category: {category}")
        
        # Display text
        text = '\n'.join(summary_text)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description='Composition Analysis Demo')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory of images')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in directory (if --image is a directory)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CompositionAnalyzer(model_path=args.model, device=args.device)
    
    # Determine input files
    input_path = Path(args.image)
    
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir() and args.batch:
        # Find all image files in directory
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = [f for f in input_path.glob('*') 
                      if f.suffix.lower() in extensions]
        logger.info(f"Found {len(image_files)} images in {input_path}")
    else:
        logger.error(f"Invalid input path: {input_path}")
        return
    
    # Process images
    for image_file in image_files:
        logger.info(f"Processing: {image_file}")
        
        try:
            # Analyze image
            results = analyzer.analyze_image(str(image_file))
            
            # Prepare output path
            save_path = None
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"{image_file.stem}_analysis.png"
            
            # Visualize results
            show_plot = len(image_files) == 1  # Only show for single image
            analyzer.visualize_results(results, save_path=save_path, show=show_plot)
            
            # Print summary
            print(f"\n{'='*50}")
            print(f"Analysis Results for: {image_file.name}")
            print(f"{'='*50}")
            
            # Rule of thirds
            rot_data = results.get('rule_of_thirds', {})
            if rot_data:
                print(f"Rule of Thirds Strength: {rot_data.get('max_strength', 0):.3f}")
            
            # Leading lines
            lines_data = results.get('leading_lines', {})
            if lines_data:
                print(f"Leading Lines Strength: {lines_data.get('strength', 0):.3f}")
                print(f"Line Angle: {lines_data.get('angle', 0):.1f}°")
            
            # Symmetry
            symmetry_data = results.get('symmetry', {})
            if symmetry_data:
                dominant = symmetry_data.get('dominant_type', 'none')
                confidence = symmetry_data.get('confidence', 0)
                print(f"Symmetry: {dominant.capitalize()} (confidence: {confidence:.3f})")
            
            # Depth
            depth_data = results.get('depth', {})
            if depth_data:
                print(f"Depth: {depth_data.get('category', 'Unknown')} "
                      f"(score: {depth_data.get('score', 0):.3f})")
                
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            continue
    
    logger.info("Demo completed!")

if __name__ == '__main__':
    main()
