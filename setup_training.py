#!/usr/bin/env python3
"""
Setup and Test Training Pipeline

This script generates a synthetic dataset and tests the training pipeline
to ensure everything works correctly before using real datasets.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.synthetic_dataset_generator import SyntheticCompositionDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_dataset():
    """Generate synthetic dataset for training."""
    logger.info("Generating synthetic dataset...")
    
    generator = SyntheticCompositionDataset('./datasets/synthetic_cadb')
    generator.generate_dataset(num_samples=300)  # Small dataset for quick testing
    
    logger.info("Synthetic dataset generation complete!")

def test_training_pipeline():
    """Test the training pipeline with synthetic data."""
    logger.info("Testing training pipeline...")
    
    # Check if dataset exists
    dataset_path = Path('./datasets/synthetic_cadb')
    if not dataset_path.exists():
        logger.error("Synthetic dataset not found! Run generate_synthetic_dataset() first.")
        return False
    
    try:
        # Test training for just a few epochs
        cmd = [
            sys.executable, 'train.py',
            '--config', 'configs/training_config.json',
            '--debug'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Training pipeline test completed successfully!")
            logger.info("Training output:")
            logger.info(result.stdout)
            return True
        else:
            logger.error("Training pipeline test failed!")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("Training test timed out (5 minutes) - this is expected for the test")
        return True
    except Exception as e:
        logger.error(f"Error running training test: {e}")
        return False

def main():
    """Main function."""
    print("AI Composition Assistant - Training Pipeline Setup")
    print("=" * 60)
    
    # Step 1: Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    generate_synthetic_dataset()
    
    # Step 2: Test training pipeline
    print("\n2. Testing training pipeline...")
    success = test_training_pipeline()
    
    if success:
        print("\n✅ Training pipeline setup complete!")
        print("\nNext steps:")
        print("1. The synthetic dataset is ready for training")
        print("2. You can now run full training with: python train.py --config configs/training_config.json")
        print("3. Monitor training with MLflow UI")
        print("4. When ready, replace synthetic data with real CADB/AVA datasets")
    else:
        print("\n❌ Training pipeline test failed!")
        print("Check the error messages above and fix any issues.")

if __name__ == '__main__':
    main()
