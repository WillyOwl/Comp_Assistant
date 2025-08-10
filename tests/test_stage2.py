"""
Unit tests for Stage 2 components of the AI Composition Assistant.
"""

import os
import sys
import pytest
import numpy as np
import torch
import cv2
from PIL import Image
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_net import HybridCompositionNet, FeaturePyramidNetwork, CompositionHead
from models.feature_detectors import (
    RuleOfThirdsDetector,
    LeadingLinesDetector,
    SymmetryDetector,
    DepthAnalyzer
)


def create_test_image(size=(224, 224)):
    """Create a test image with known composition elements."""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add some basic shapes
    cv2.line(image, (0, 0), (size[1], size[0]), (255, 255, 255), 2)
    cv2.circle(image, (size[1]//2, size[0]//2), 50, (200, 200, 200), -1)
    cv2.rectangle(image, (50, 50), (150, 150), (150, 150, 150), 2)
    
    return image


class TestHybridNet:
    """Tests for the hybrid CNN-ViT network."""
    
    @pytest.fixture
    def model(self):
        return HybridCompositionNet(img_size=224, patch_size=16)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, HybridCompositionNet)
        assert isinstance(model.backbone, torch.nn.Module)
        assert isinstance(model.fpn, FeaturePyramidNetwork)
        assert isinstance(model.vit, torch.nn.Module)
    
    def test_forward_pass(self, model):
        """Test forward pass through the model."""
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert isinstance(output, dict)
        assert 'rule_of_thirds' in output
        assert 'leading_lines' in output
        assert 'symmetry' in output
        assert 'depth' in output
        
        # Check output shapes
        assert output['rule_of_thirds'].shape[1] == 9  # 3x3 grid points
        assert output['leading_lines'].shape[1] == 5  # line params + confidence
        assert output['symmetry'].shape[1] == 4  # symmetry types + confidence
        assert output['depth'].shape[1] == 1  # depth map
    
    def test_composition_head(self):
        """Test composition head components."""
        head = CompositionHead(768, 'rule_of_thirds')
        x = torch.randn(1, 768)
        output = head(x)
        
        assert output.shape == (1, 9)
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestRuleOfThirdsDetector:
    """Tests for Rule of Thirds detector."""
    
    @pytest.fixture
    def detector(self):
        return RuleOfThirdsDetector()
    
    def test_detection(self, detector):
        """Test rule of thirds detection."""
        image = create_test_image()
        results = detector.detect(image)
        
        assert 'points' in results
        assert 'scores' in results
        assert 'saliency_map' in results
        
        assert len(results['points']) == 9  # 3x3 grid
        assert len(results['scores']) == 9
        assert all(0 <= score <= 1 for score in results['scores'])
        assert results['saliency_map'].shape == image.shape[:2]


class TestLeadingLinesDetector:
    """Tests for Leading Lines detector."""
    
    @pytest.fixture
    def detector(self):
        return LeadingLinesDetector()
    
    def test_detection(self, detector):
        """Test leading lines detection."""
        image = create_test_image()
        results = detector.detect(image)
        
        assert 'lines' in results
        assert 'scores' in results
        assert 'vanishing_points' in results
        
        if results['lines']:
            assert len(results['lines']) == len(results['scores'])
            assert all(0 <= score <= 1 for score in results['scores'])
            
            # Check line format
            for line in results['lines']:
                assert len(line) == 4  # x1, y1, x2, y2
                assert all(isinstance(coord, (int, float)) for coord in line)


class TestSymmetryDetector:
    """Tests for Symmetry detector."""
    
    @pytest.fixture
    def detector(self):
        return SymmetryDetector()
    
    def test_detection(self, detector):
        """Test symmetry detection."""
        image = create_test_image()
        results = detector.detect(image)
        
        assert 'horizontal_score' in results
        assert 'vertical_score' in results
        assert 'radial_score' in results
        assert 'dominant_type' in results
        
        assert 0 <= results['horizontal_score'] <= 1
        assert 0 <= results['vertical_score'] <= 1
        assert 0 <= results['radial_score'] <= 1
        assert results['dominant_type'] in ['horizontal', 'vertical', 'radial']
    
    def test_symmetry_computation(self, detector):
        """Test symmetry score computation."""
        # Create perfectly symmetric image
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (50, 0), (50, 100), 255, 2)  # Vertical line
        
        results = detector.detect(image)
        assert results['vertical_score'] > results['horizontal_score']


class TestDepthAnalyzer:
    """Tests for Depth Analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return DepthAnalyzer()
    
    def test_analysis(self, analyzer):
        """Test depth analysis."""
        image = create_test_image()
        results = analyzer.analyze(image)
        
        assert 'depth_map' in results
        assert 'layers' in results
        assert 'focal_points' in results
        assert 'depth_statistics' in results
        
        # Check depth map
        assert results['depth_map'].shape == image.shape[:2]
        assert np.all(results['depth_map'] >= 0)
        
        # Check depth statistics
        stats = results['depth_statistics']
        assert 'mean_depth' in stats
        assert 'std_depth' in stats
        assert 'min_depth' in stats
        assert 'max_depth' in stats
        assert 'depth_range' in stats
        
        # Check focal points
        assert isinstance(results['focal_points'], list)
        for point in results['focal_points']:
            assert 'position' in point
            assert 'depth' in point
            assert 'strength' in point


def test_end_to_end():
    """End-to-end test of the Stage 2 pipeline."""
    # Create test image
    image = create_test_image()
    
    # Initialize all components
    hybrid_net = HybridCompositionNet()
    rot_detector = RuleOfThirdsDetector()
    lines_detector = LeadingLinesDetector()
    symmetry_detector = SymmetryDetector()
    depth_analyzer = DepthAnalyzer()
    
    # Process image through all detectors
    rot_results = rot_detector.detect(image)
    lines_results = lines_detector.detect(image)
    symmetry_results = symmetry_detector.detect(image)
    depth_results = depth_analyzer.analyze(image)
    
    # Run through hybrid network
    with torch.no_grad():
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        hybrid_results = hybrid_net(x)
    
    # Verify all results are present and valid
    assert all(key in hybrid_results for key in 
              ['rule_of_thirds', 'leading_lines', 'symmetry', 'depth'])
    
    assert len(rot_results['points']) == 9
    assert len(lines_results['scores']) == len(lines_results['lines'])
    assert symmetry_results['dominant_type'] in ['horizontal', 'vertical', 'radial']
    assert len(depth_results['focal_points']) <= 5  # Maximum 5 focal points


if __name__ == '__main__':
    pytest.main([__file__])
