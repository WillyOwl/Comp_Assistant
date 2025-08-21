"""
Dataset preparation and loading for composition analysis training.

This module handles loading and preprocessing of CADB and AVA datasets with
composition-specific annotations for training the hybrid CNN-ViT models.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class CompositionDataset(Dataset):
    """
    Custom dataset for composition analysis training.
    
    Supports CADB (Composition Assessment Database) and AVA (Aesthetic Visual Analysis)
    datasets with composition-specific annotations.
    """
    
    def __init__(self, 
                 data_dir: str,
                 annotations_file: str,
                 dataset_type: str = 'cadb',
                 split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the composition dataset.
        
        Args:
            data_dir: Directory containing the image files
            annotations_file: Path to annotations CSV/JSON file
            dataset_type: Type of dataset ('cadb', 'ava', 'mixed')
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_size: Target image size (width, height)
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        self.image_paths = list(self.annotations.keys())
        
        logger.info(f"Loaded {len(self.image_paths)} samples for {split} split")
        
    def _load_annotations(self, annotations_file: str) -> Dict:
        """Load and parse annotations file."""
        annotations_path = Path(annotations_file)
        
        if annotations_path.suffix == '.json':
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
        elif annotations_path.suffix == '.csv':
            df = pd.read_csv(annotations_path)
            annotations = self._process_csv_annotations(df)
        else:
            raise ValueError(f"Unsupported annotation format: {annotations_path.suffix}")
            
        return annotations
    
    def _process_csv_annotations(self, df: pd.DataFrame) -> Dict:
        """Process CSV annotations into the required format."""
        annotations = {}
        
        for _, row in df.iterrows():
            image_path = row['image_path']
            
            # Extract composition annotations
            composition_data = {
                'rule_of_thirds': {
                    'score': row.get('rule_of_thirds_score', 0.0),
                    'points': self._parse_grid_points(row.get('rot_grid_points', '')),
                    'strength': row.get('rot_strength', 0.0)
                },
                'leading_lines': {
                    'score': row.get('leading_lines_score', 0.0),
                    'lines': self._parse_lines(row.get('leading_lines', '')),
                    'strength': row.get('lines_strength', 0.0)
                },
                'symmetry': {
                    'score': row.get('symmetry_score', 0.0),
                    'type': row.get('symmetry_type', 'none'),
                    'axis': row.get('symmetry_axis', 0.0)
                },
                'depth': {
                    'score': row.get('depth_score', 0.0),
                    'layers': row.get('depth_layers', 1),
                    'foreground_ratio': row.get('foreground_ratio', 0.0)
                },
                'overall_quality': row.get('overall_score', 0.0),
                'aesthetic_score': row.get('aesthetic_score', 0.0)
            }
            
            annotations[image_path] = composition_data
            
        return annotations
    
    def _parse_grid_points(self, points_str: str) -> List[Tuple[float, float]]:
        """Parse rule of thirds grid points from string."""

        if not points_str or pd.isna(points_str):
            return []
        
        try:
            # Expected format: "x1,y1,x2,y2,x3,y3"

            points = []

            for point in str(points_str).split(';'):
                if point.strip():
                    x, y = map(float, point.split(','))
                    points.append((x, y))

            return points
        
        except (ValueError, IndexError):
            return []
        
    def _parse_lines(self, lines_str: str) -> List[Dict]:
        """Parse leading lines from string."""

        if not lines_str or pd.isna(lines_str):
            return []
        
        try:
            # Expected format: "x1,y1,x2,y2,strength;..."

            lines = []

            for line in str(lines_str).split(';'):
                if line.strip():
                    parts = line.split(',')

                    if len(parts) >= 4:
                        lines.append({
                            'start_x': float(parts[0]),
                            'start_y': float(parts[1]),
                            'end_x': float(parts[2]),
                            'end_y': float(parts[3]),
                            'strength': float(parts[4]) if len(parts) > 4 else 1.0
                        })
            
            return lines
        
        except (ValueError, IndexError):
            return []
        
    def __len__(self) -> int:
        """Return the size of the dataset"""

        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image tensor and composition targets
        
        """

        image_path = self.image_paths[idx]
        full_path = self.data_dir / image_path

        # Load image

        try:
            image = Image.open(full_path).convert('RGB')
            image = np.array(image)

        except Exception as e:
            logger.warning(f"Error loading image {full_path}: {e}")

            # Return a blank image if loading fails

            image = np.zeros((224, 224, 3), dtype = np.uint8)

        # Get annotations

        annotations = self.annotations[image_path]

        # Apply transforms

        if self.transform:
            transformed = self.transform(image = image)
            image = transformed['image']

        else:
            # Default transform if none provided
            transform = A.Compose([
                A.Resize(*self.target_size),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

            transformed = transform(image = image)
            image = transformed['image']

        # Prepare targets
        targets = self._prepare_targets(annotations)

        return {
            'image': image,
            'image_path': image_path,
            **targets
        }
    
    def _prepare_targets(self, annotations: Dict) -> Dict[str, torch.Tensor]:
        """Convert annotations to model targets."""
        targets = {}

        # Rule of thirds (9-point grid confidences)
        rot_points = annotations.get('rule_of_thirds', {}).get('points', [])
        rot_target = torch.zeros(9, dtype=torch.float32)

        for i, (x, y) in enumerate(rot_points[:9]):  # Limit to 9 points
            # Map to grid positions (3x3) with bounds checking
            grid_x = int(np.clip(x * 3, 0, 2))
            grid_y = int(np.clip(y * 3, 0, 2))
            strength = annotations.get('rule_of_thirds', {}).get('strength', 0.0)
            rot_target[grid_y * 3 + grid_x] = max(0.0, min(1.0, strength))

        targets['rule_of_thirds'] = rot_target

        # Leading lines target (line parameters + confidence)
        lines = annotations.get('leading_lines', {}).get('lines', [])

        if lines:
            # Use the strongest line
            strongest_line = max(lines, key=lambda x: x.get('strength', 0))
            
            # Validate and clip coordinates
            lines_target = torch.tensor([
                np.clip(strongest_line.get('start_x', 0), 0, 1),
                np.clip(strongest_line.get('start_y', 0), 0, 1),
                np.clip(strongest_line.get('end_x', 0), 0, 1),
                np.clip(strongest_line.get('end_y', 0), 0, 1),
                np.clip(strongest_line.get('strength', 0), 0, 1)
            ], dtype=torch.float32)
        else:
            lines_target = torch.zeros(5, dtype=torch.float32)

        targets['leading_lines'] = lines_target

        # Symmetry target (type probabilities)
        symmetry_type = annotations.get('symmetry', {}).get('type', 'none')
        symmetry_target = torch.zeros(4, dtype=torch.float32)  # none, horizontal, vertical, radial

        type_mapping = {'none': 0, 'horizontal': 1, 'vertical': 2, 'radial': 3}
        if symmetry_type in type_mapping:
            symmetry_target[type_mapping[symmetry_type]] = 1.0

        targets['symmetry'] = symmetry_target

        # Depth target (single value) with validation
        depth_score = annotations.get('depth', {}).get('score', 0.0)
        depth_target = torch.tensor([np.clip(depth_score, 0.0, 1.0)], dtype=torch.float32)
        targets['depth'] = depth_target

        # Overall quality (for auxiliary loss) with validation
        overall_score = annotations.get('overall_quality', 0.0)
        overall_target = torch.tensor([np.clip(overall_score, 0.0, 1.0)], dtype=torch.float32)
        targets['overall_quality'] = overall_target

        return targets
    
def get_composition_transforms(split: str = 'train',
                                   target_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        
        """
        Get albumentations transform pipeline for composition training.
    
        Args:
            split: Dataset split ('train', 'val', 'test')
            target_size: Target image size
        
        Returns:
            Albumentations compose object
        """

        if split == 'train':
            # Training augmentations that preserve composition
            transforms = A.Compose([
                A.Resize(*target_size),

                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                ], p=0.5),
            
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.3),

                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ], p=0.2),

            # Spatial transforms that maintain composition
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                    border_mode=0, p=0.3
                ),

                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        else:
            # Validation and test transforms (no augmentation)
            transforms = A.Compose([
                A.Resize(*target_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

        return transforms
    
def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        config: Full training configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    data_config = config['data']
    training_config = config['training']
    
    # Create datasets
    train_dataset = CompositionDataset(
        data_dir=data_config['train_data_dir'],
        annotations_file=data_config['train_annotations'],
        dataset_type=data_config.get('dataset_type', 'cadb'),
        split='train',
        transform=get_composition_transforms('train', 
                                        tuple(data_config['target_size'])),
        target_size=tuple(data_config['target_size'])
    )

    val_dataset = CompositionDataset(
        data_dir=data_config['val_data_dir'],
        annotations_file=data_config['val_annotations'],
        dataset_type=data_config.get('dataset_type', 'cadb'),
        split='val',
        transform=get_composition_transforms('val', 
                                        tuple(data_config['target_size'])),
        target_size=tuple(data_config['target_size'])
    )

    test_dataset = CompositionDataset(
        data_dir=data_config['test_data_dir'],
        annotations_file=data_config['test_annotations'],
        dataset_type=data_config.get('dataset_type', 'cadb'),
        split='test',
        transform=get_composition_transforms('test',
                                        tuple(data_config['target_size'])),
        target_size=tuple(data_config['target_size'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

class CLIPDatasetAdapter:
    """
    Adapter for using CLIP pretrained features in composition training.

    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP adapter."""
        from transformers import CLIPProcessor, CLIPVisionModel
        
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.vision_model.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP vision features from images.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            CLIP vision features [B, feature_dim]
        """
        with torch.no_grad():
            # Convert to PIL for CLIP processor
            pil_images = []
            for img in images:
                # Denormalize if needed
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Process through CLIP
            inputs = self.processor(images=pil_images, return_tensors="pt")
            outputs = self.vision_model(**inputs)
            
            return outputs.pooler_output  # [B, 512] for base model