"""
Comprehensive tests for the data pipeline robustness.

This module tests the data loading, preprocessing, and augmentation pipeline
to ensure robustness and error handling.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
import importlib.util

# Add project root to path for any relative imports
sys.path.append(str(Path(__file__).parent.parent))

# Import dataset_loader using importlib to avoid VS Code import resolution issues
dataset_loader_path = Path(__file__).parent.parent / "training" / "dataset_loader.py"
spec = importlib.util.spec_from_file_location("dataset_loader", dataset_loader_path)
dataset_loader = importlib.util.module_from_spec(spec)
sys.modules["dataset_loader"] = dataset_loader  # Add to sys.modules to avoid reimporting
spec.loader.exec_module(dataset_loader)

CompositionDataset = dataset_loader.CompositionDataset
get_composition_transforms = dataset_loader.get_composition_transforms
create_data_loaders = dataset_loader.create_data_loaders
CLIPDatasetAdapter = dataset_loader.CLIPDatasetAdapter


class TestDataPipelineRobustness:
    """Test suite for data pipeline robustness."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample images
            for i in range(5):
                # Create RGB image
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(temp_path / f"image_{i}.jpg")
                
            # Create corrupted image
            with open(temp_path / "corrupted.jpg", 'w') as f:
                f.write("not an image")
                
            yield temp_path
    
    @pytest.fixture
    def sample_annotations(self, sample_data_dir):
        """Create sample annotations file."""
        annotations = {}
        
        for i in range(5):
            annotations[f"image_{i}.jpg"] = {
                "rule_of_thirds": {
                    "score": 0.7,
                    "points": [(0.33, 0.33), (0.66, 0.66)],
                    "strength": 0.8
                },
                "leading_lines": {
                    "score": 0.6,
                    "lines": [{
                        "start_x": 0.1, "start_y": 0.1,
                        "end_x": 0.9, "end_y": 0.9,
                        "strength": 0.7
                    }],
                    "strength": 0.7
                },
                "symmetry": {
                    "score": 0.5,
                    "type": "horizontal",
                    "axis": 0.5
                },
                "depth": {
                    "score": 0.4,
                    "layers": 2,
                    "foreground_ratio": 0.3
                },
                "overall_quality": 0.65,
                "aesthetic_score": 0.7
            }
        
        # Add corrupted annotation
        annotations["corrupted.jpg"] = {
            "rule_of_thirds": {"score": 0.0, "points": [], "strength": 0.0},
            "leading_lines": {"score": 0.0, "lines": [], "strength": 0.0},
            "symmetry": {"score": 0.0, "type": "none", "axis": 0.0},
            "depth": {"score": 0.0, "layers": 1, "foreground_ratio": 0.0},
            "overall_quality": 0.0,
            "aesthetic_score": 0.0
        }
        
        # Save as JSON
        annotations_file = sample_data_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
            
        return str(annotations_file)
    
    def test_dataset_initialization(self, sample_data_dir, sample_annotations):
        """Test dataset initialization with valid data."""
        dataset = CompositionDataset(
            data_dir=str(sample_data_dir),
            annotations_file=sample_annotations,
            dataset_type='cadb',
            split='train',
            target_size=(224, 224)
        )
        
        assert len(dataset) == 6  # 5 good + 1 corrupted
        assert dataset.dataset_type == 'cadb'
        assert dataset.split == 'train'
    
    def test_dataset_getitem_valid(self, sample_data_dir, sample_annotations):
        """Test dataset __getitem__ with valid data."""
        dataset = CompositionDataset(
            data_dir=str(sample_data_dir),
            annotations_file=sample_annotations,
            target_size=(224, 224)
        )
        
        sample = dataset[0]  # Get first valid image
        
        # Check sample structure
        assert 'image' in sample
        assert 'image_path' in sample
        assert 'rule_of_thirds' in sample
        assert 'leading_lines' in sample
        assert 'symmetry' in sample
        assert 'depth' in sample
        assert 'overall_quality' in sample
        
        # Check tensor shapes
        assert sample['image'].shape == (3, 224, 224)
        assert sample['rule_of_thirds'].shape == (9,)
        assert sample['leading_lines'].shape == (5,)
        assert sample['symmetry'].shape == (4,)
        assert sample['depth'].shape == (1,)
        assert sample['overall_quality'].shape == (1,)
    
    def test_dataset_corrupted_image_handling(self, sample_data_dir, sample_annotations):
        """Test handling of corrupted images."""
        dataset = CompositionDataset(
            data_dir=str(sample_data_dir),
            annotations_file=sample_annotations,
            target_size=(224, 224)
        )
        
        # Find corrupted image index
        corrupted_idx = None
        for i, path in enumerate(dataset.image_paths):
            if "corrupted.jpg" in path:
                corrupted_idx = i
                break
        
        assert corrupted_idx is not None
        
        # Should return blank image instead of crashing
        sample = dataset[corrupted_idx]
        assert sample['image'].shape == (3, 224, 224)
        # Blank image should be mostly zeros after normalization
        assert torch.all(sample['image'] <= 0.5)  # Normalized values
    
    def test_transforms_train(self):
        """Test training transforms."""
        transforms = get_composition_transforms('train', (224, 224))
        
        # Create sample image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply transforms
        transformed = transforms(image=image)
        result = transformed['image']
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32
    
    def test_transforms_val_test(self):
        """Test validation/test transforms (no augmentation)."""
        for split in ['val', 'test']:
            transforms = get_composition_transforms(split, (224, 224))
            
            # Create sample image
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            # Apply transforms multiple times - should be deterministic
            results = []
            for _ in range(3):
                transformed = transforms(image=image)
                results.append(transformed['image'])
            
            # All results should be identical (no random augmentation)
            for i in range(1, len(results)):
                assert torch.allclose(results[0], results[i])
    
    def test_target_preparation_bounds(self, sample_data_dir, sample_annotations):
        """Test target preparation with out-of-bounds values."""
        dataset = CompositionDataset(
            data_dir=str(sample_data_dir),
            annotations_file=sample_annotations,
            target_size=(224, 224)
        )
        
        # Create annotation with out-of-bounds values
        bad_annotation = {
            "rule_of_thirds": {
                "points": [(1.5, -0.5), (0.5, 2.0)],  # Out of bounds
                "strength": 1.5  # Out of bounds
            },
            "leading_lines": {
                "lines": [{
                    "start_x": -0.5, "start_y": 1.5,
                    "end_x": 2.0, "end_y": -1.0,
                    "strength": 2.0
                }]
            },
            "symmetry": {"type": "invalid_type"},
            "depth": {"score": 2.0},  # Out of bounds
            "overall_quality": -0.5  # Out of bounds
        }
        
        targets = dataset._prepare_targets(bad_annotation)
        
        # Check that values are properly clipped
        assert torch.all(targets['leading_lines'] >= 0.0)
        assert torch.all(targets['leading_lines'] <= 1.0)
        assert torch.all(targets['depth'] >= 0.0)
        assert torch.all(targets['depth'] <= 1.0)
        assert torch.all(targets['overall_quality'] >= 0.0)
        assert torch.all(targets['overall_quality'] <= 1.0)
    
    def test_annotation_parsing_robustness(self, sample_data_dir):
        """Test robustness of annotation parsing."""
        # Create CSV with missing/invalid data
        df = pd.DataFrame({
            'image_path': ['image_0.jpg', 'image_1.jpg'],
            'rule_of_thirds_score': [0.7, None],  # Missing value
            'rot_grid_points': ['0.33,0.33;0.66,0.66', 'invalid_format'],
            'leading_lines_score': [0.6, 0.8],
            'leading_lines': ['0.1,0.1,0.9,0.9,0.7', ''],  # Empty string
            'symmetry_type': ['horizontal', 'invalid_type'],
            'overall_score': [0.65, 'not_a_number']  # Invalid type
        })
        
        csv_file = sample_data_dir / "test_annotations.csv"
        df.to_csv(csv_file, index=False)
        
        dataset = CompositionDataset(
            data_dir=str(sample_data_dir),
            annotations_file=str(csv_file),
            target_size=(224, 224)
        )
        
        # Should not crash and handle missing/invalid data gracefully
        assert len(dataset) == 2
        
        # Test samples
        for i in range(len(dataset)):
            sample = dataset[i]
            # Should have all required keys with valid tensor shapes
            assert all(key in sample for key in [
                'image', 'rule_of_thirds', 'leading_lines', 
                'symmetry', 'depth', 'overall_quality'
            ])
    
    def test_clip_adapter_functionality(self):
        """Test CLIP adapter functionality."""
        try:
            adapter = CLIPDatasetAdapter()
            
            # Create sample batch
            batch_size = 2
            images = torch.randn(batch_size, 3, 224, 224)
            
            # Extract features
            features = adapter.extract_features(images)
            
            assert features.shape[0] == batch_size
            assert features.shape[1] == 512  # CLIP base model output size
            
        except ImportError:
            # Skip if transformers not available
            pytest.skip("transformers library not available")


def test_data_loader_creation():
    """Test data loader creation function."""
    config = {
        'data': {
            'train_data_dir': '/tmp/train',
            'val_data_dir': '/tmp/val', 
            'test_data_dir': '/tmp/test',
            'train_annotations': '/tmp/train.json',
            'val_annotations': '/tmp/val.json',
            'test_annotations': '/tmp/test.json',
            'target_size': [224, 224],
            'dataset_type': 'cadb'
        },
        'batch_size': 16,
        'num_workers': 2
    }
    
    # This would fail with actual missing files, but tests function signature
    try:
        train_loader, val_loader, test_loader = create_data_loaders(config)
    except (FileNotFoundError, ValueError):
        # Expected when files don't exist
        pass


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running data pipeline robustness tests...")
    
    # Test transforms
    transforms = get_composition_transforms('train')
    print("✓ Training transforms created successfully")
    
    transforms = get_composition_transforms('val')
    print("✓ Validation transforms created successfully")
    
    print("Data pipeline basic functionality verified!")
