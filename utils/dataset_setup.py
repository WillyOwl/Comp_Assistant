#!/usr/bin/env python3
"""
Dataset setup script for composition analysis training.

This script helps download, organize, and prepare the CADB and AVA datasets
for training composition analysis models.
"""

import argparse
import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manager for downloading and organizing composition datasets."""

    def __init__(self, data_root: str = "./datasets"):
        """
        Initialize dataset manager.
        
        Args:
            data_root: Root directory for datasets
        """

        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'cadb': {
                'name': 'Composition Assessment Database',
                'url': 'https://github.com/bcmi/CADB-dataset',
                'description': '9,497 images with composition annotations',
                'splits': ['train', 'val', 'test'],
                'annotation_format': 'csv'
            },
            'ava': {
                'name': 'AVA (Aesthetic Visual Analysis)',
                'url': 'https://github.com/mtobeiyf/ava_downloader',
                'description': '250K+ images with aesthetic scores',
                'splits': ['train', 'val', 'test'],
                'annotation_format': 'csv'
            }
        }

    def setup_cadb_dataset(self, output_dir: Optional[str] = None) -> bool:
        """
        Set up CADB dataset with proper organization.
        
        Args:
            output_dir: Output directory (defaults to data_root/cadb)
            
        Returns:
            True if setup successful
        """

        if output_dir is None:
            output_dir = self.data_root / 'cadb'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Setting up CADB dataset...")
        logger.info("Note: CADB dataset needs to be manually downloaded from:")
        logger.info("https://github.com/bcmi/CADB-dataset")
        
        # Create directory structure
        directories = ['train', 'val', 'test', 'annotations']
        for directory in directories:
            (output_dir / directory).mkdir(exist_ok=True)
        
        # Create sample annotation files
        self._create_sample_cadb_annotations(output_dir / 'annotations')
        
        logger.info(f"CADB dataset structure created at: {output_dir}")
        logger.info("Please download the actual dataset and organize according to the structure.")
        
        return True
    
    def setup_ava_dataset(self, output_dir: Optional[str] = None) -> bool:
        """
        Set up AVA dataset with proper organization.
        
        Args:
            output_dir: Output directory (defaults to data_root/ava)
            
        Returns:
            True if setup successful
        """

        if output_dir is None:
            output_dir = self.data_root / 'ava'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Setting up AVA dataset...")
        logger.info("Note: AVA dataset needs to be manually downloaded from:")
        logger.info("https://github.com/mtobeiyf/ava_downloader")
        
        # Create directory structure
        directories = ['train', 'val', 'test', 'annotations']
        for directory in directories:
            (output_dir / directory).mkdir(exist_ok=True)
        
        # Create sample annotation files
        self._create_sample_ava_annotations(output_dir / 'annotations')
        
        logger.info(f"AVA dataset structure created at: {output_dir}")
        logger.info("Please download the actual dataset and organize according to the structure.")
        
        return True
    
    def _create_sample_cadb_annotations(self, annotations_dir: Path):
        """Create sample CADB annotation files."""
        sample_data = {
            'image_path': [
                'train/sample_001.jpg',
                'train/sample_002.jpg',
                'train/sample_003.jpg'
            ],
            'rule_of_thirds_score': [0.8, 0.6, 0.9],
            'rot_grid_points': [
                '0.33,0.33;0.67,0.33;0.33,0.67',
                '0.25,0.25;0.75,0.75',
                '0.33,0.33;0.67,0.33;0.33,0.67;0.67,0.67'
            ],
            'rot_strength': [0.8, 0.6, 0.9],
            'leading_lines_score': [0.7, 0.5, 0.8],
            'leading_lines': [
                '0.0,0.0,1.0,1.0,0.8',
                '0.2,0.2,0.8,0.8,0.6',
                '0.1,0.1,0.9,0.9,0.9'
            ],
            'lines_strength': [0.7, 0.5, 0.8],
            'symmetry_score': [0.6, 0.4, 0.7],
            'symmetry_type': ['horizontal', 'none', 'vertical'],
            'symmetry_axis': [0.5, 0.0, 0.5],
            'depth_score': [0.8, 0.6, 0.9],
            'depth_layers': [3, 2, 4],
            'foreground_ratio': [0.3, 0.4, 0.2],
            'overall_score': [0.75, 0.55, 0.85],
            'aesthetic_score': [0.8, 0.6, 0.9]
        }
        
        # Create train annotations
        train_df = pd.DataFrame(sample_data)
        train_df.to_csv(annotations_dir / 'train.csv', index=False)
        
        # Create val and test with similar structure (smaller samples)
        val_data = {k: v[:2] for k, v in sample_data.items()}
        val_data['image_path'] = ['val/sample_001.jpg', 'val/sample_002.jpg']
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(annotations_dir / 'val.csv', index=False)
        
        test_data = {k: v[:1] for k, v in sample_data.items()}
        test_data['image_path'] = ['test/sample_001.jpg']
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(annotations_dir / 'test.csv', index=False)
        
        logger.info("Sample CADB annotation files created")

    def _create_sample_ava_annotations(self, annotations_dir: Path):
        """Create sample AVA annotation files."""
        sample_data = {
            'image_id': [1, 2, 3],
            'image_path': [
                'train/1.jpg',
                'train/2.jpg',
                'train/3.jpg'
            ],
            'score1': [1, 2, 0],
            'score2': [2, 3, 1],
            'score3': [5, 4, 2],
            'score4': [8, 6, 3],
            'score5': [12, 8, 5],
            'score6': [15, 12, 8],
            'score7': [20, 15, 12],
            'score8': [18, 20, 15],
            'score9': [10, 15, 18],
            'score10': [8, 12, 20],
            'mean_score': [6.8, 7.2, 8.1],
            'std_score': [1.5, 1.3, 1.8],
            'rule_of_thirds_score': [0.7, 0.8, 0.9],
            'leading_lines_score': [0.6, 0.7, 0.8],
            'symmetry_score': [0.5, 0.6, 0.7],
            'depth_score': [0.8, 0.7, 0.9]
        }
        
        # Create train annotations
        train_df = pd.DataFrame(sample_data)
        train_df.to_csv(annotations_dir / 'train.csv', index=False)
        
        # Create val and test with similar structure
        val_data = {k: v[:2] for k, v in sample_data.items()}
        val_data['image_id'] = [4, 5]
        val_data['image_path'] = ['val/4.jpg', 'val/5.jpg']
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(annotations_dir / 'val.csv', index=False)
        
        test_data = {k: v[:1] for k, v in sample_data.items()}
        test_data['image_id'] = [6]
        test_data['image_path'] = ['test/6.jpg']
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(annotations_dir / 'test.csv', index=False)
        
        logger.info("Sample AVA annotation files created")

    def create_mixed_dataset(self, cadb_dir: str, ava_dir: str, 
                           output_dir: str, split_ratios: Dict[str, float] = None):
        """
        Create a mixed dataset combining CADB and AVA.
        
        Args:
            cadb_dir: CADB dataset directory
            ava_dir: AVA dataset directory
            output_dir: Output directory for mixed dataset
            split_ratios: Train/val/test split ratios
        """
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating mixed dataset from CADB and AVA...")
        
        # Load annotations from both datasets
        cadb_annotations = self._load_cadb_annotations(cadb_dir)
        ava_annotations = self._load_ava_annotations(ava_dir)
        
        # Combine and harmonize annotations
        combined_annotations = self._combine_annotations(cadb_annotations, ava_annotations)
        
        # Split dataset
        train_data, val_data, test_data = self._split_dataset(combined_annotations, split_ratios)
        
        # Save mixed dataset
        self._save_mixed_dataset(output_path, train_data, val_data, test_data)
        
        logger.info(f"Mixed dataset created at: {output_path}")

    def _load_cadb_annotations(self, cadb_dir: str) -> pd.DataFrame:
        """Load CADB annotations."""
        # Implementation would load actual CADB annotations
        # This is a placeholder
        return pd.DataFrame()
    
    def _load_ava_annotations(self, ava_dir: str) -> pd.DataFrame:
        """Load AVA annotations."""
        # Implementation would load actual AVA annotations
        # This is a placeholder
        return pd.DataFrame()
    
    def _combine_annotations(self, cadb_df: pd.DataFrame, ava_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and harmonize annotations from different datasets."""
        # Implementation would harmonize the different annotation formats
        # This is a placeholder
        return pd.DataFrame()
    
    def _split_dataset(self, annotations: pd.DataFrame, 
                      split_ratios: Dict[str, float]) -> tuple:
        """Split dataset into train/val/test."""
        # Implementation would split the dataset
        # This is a placeholder
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def _save_mixed_dataset(self, output_path: Path, 
                           train_data: pd.DataFrame,
                           val_data: pd.DataFrame, 
                           test_data: pd.DataFrame):
        """Save the mixed dataset."""
        # Create directories
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(exist_ok=True)
        
        annotations_dir = output_path / 'annotations'
        annotations_dir.mkdir(exist_ok=True)
        
        # Save annotations
        train_data.to_csv(annotations_dir / 'train.csv', index=False)
        val_data.to_csv(annotations_dir / 'val.csv', index=False)
        test_data.to_csv(annotations_dir / 'test.csv', index=False)
    
    def validate_dataset(self, dataset_dir: str) -> bool:
        """
        Validate dataset structure and annotations.
        
        Args:
            dataset_dir: Dataset directory to validate
            
        Returns:
            True if dataset is valid
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            logger.error(f"Dataset directory does not exist: {dataset_dir}")
            return False
        
        # Check required directories
        required_dirs = ['train', 'val', 'test', 'annotations']
        for dir_name in required_dirs:
            if not (dataset_path / dir_name).exists():
                logger.error(f"Missing required directory: {dir_name}")
                return False
        
        # Check annotation files
        annotation_files = ['train.csv', 'val.csv', 'test.csv']
        annotations_dir = dataset_path / 'annotations'
        
        for file_name in annotation_files:
            file_path = annotations_dir / file_name
            if not file_path.exists():
                logger.error(f"Missing annotation file: {file_name}")
                return False
            
            # Validate annotation format
            try:
                df = pd.read_csv(file_path)
                if 'image_path' not in df.columns:
                    logger.error(f"Missing 'image_path' column in {file_name}")
                    return False
                
                logger.info(f"Validated {file_name}: {len(df)} samples")
                
            except Exception as e:
                logger.error(f"Error reading {file_name}: {e}")
                return False
        
        logger.info(f"Dataset validation successful: {dataset_dir}")
        return True
    
    def generate_dataset_info(self, dataset_dir: str) -> Dict:
        """
        Generate dataset information summary.
        
        Args:
            dataset_dir: Dataset directory
            
        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(dataset_dir)
        info = {
            'dataset_name': dataset_path.name,
            'dataset_path': str(dataset_path),
            'splits': {},
            'total_samples': 0,
            'annotation_columns': []
        }
        
        annotations_dir = dataset_path / 'annotations'
        
        for split in ['train', 'val', 'test']:
            annotation_file = annotations_dir / f'{split}.csv'
            if annotation_file.exists():
                df = pd.read_csv(annotation_file)
                info['splits'][split] = len(df)
                info['total_samples'] += len(df)
                
                if not info['annotation_columns']:
                    info['annotation_columns'] = list(df.columns)
        
        return info
    
def main():
    """Main function for dataset setup script."""
    parser = argparse.ArgumentParser(description='Setup datasets for composition analysis training')
    
    parser.add_argument('--dataset', choices=['cadb', 'ava', 'mixed'], required=True,
                       help='Dataset to setup')
    parser.add_argument('--data-root', default='./datasets',
                       help='Root directory for datasets')
    parser.add_argument('--output-dir', 
                       help='Custom output directory')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing dataset')
    parser.add_argument('--info', action='store_true',
                       help='Generate dataset information')
    
    args = parser.parse_args()
    
    # Create dataset manager
    manager = DatasetManager(args.data_root)
    
    if args.validate:
        dataset_dir = args.output_dir or os.path.join(args.data_root, args.dataset)
        success = manager.validate_dataset(dataset_dir)
        return 0 if success else 1
    
    if args.info:
        dataset_dir = args.output_dir or os.path.join(args.data_root, args.dataset)
        info = manager.generate_dataset_info(dataset_dir)
        print(json.dumps(info, indent=2))
        return 0
    
    # Setup dataset
    if args.dataset == 'cadb':
        success = manager.setup_cadb_dataset(args.output_dir)
    elif args.dataset == 'ava':
        success = manager.setup_ava_dataset(args.output_dir)
    elif args.dataset == 'mixed':
        if not args.output_dir:
            logger.error("Output directory required for mixed dataset")
            return 1
        
        cadb_dir = os.path.join(args.data_root, 'cadb')
        ava_dir = os.path.join(args.data_root, 'ava')
        manager.create_mixed_dataset(cadb_dir, ava_dir, args.output_dir)
        success = True
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())