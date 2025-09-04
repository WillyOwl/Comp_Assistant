# AI Photography Composition Assistant - Project Restoration Guide

This comprehensive guide will help you restore and continue the AI Photography Composition Assistant project from scratch. Follow these steps to recreate the complete development environment and resume training.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.11 or 3.13 (project was developed with 3.13.5)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 50GB+ free space for datasets and models

### Required Software
```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-dev python3-pip build-essential cmake \
    libopencv-dev libgtk-3-dev libboost-all-dev \
    git curl wget unzip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libffi-dev

# Install NVIDIA drivers and CUDA (if using GPU)
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit

# Install Anaconda/Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## 🚀 Step-by-Step Project Setup

### Step 1: Create Project Repository
```bash
# Create project directory
mkdir -p ~/Comp_Assistant
cd ~/Comp_Assistant

# Initialize git repository (if you want version control)
git init
```

### Step 2: Recreate Project Structure
```bash
# Create directory structure
mkdir -p {analysis,api,configs,datasets/{cadb/{annotations,test,train,val},synthetic_cadb/{annotations,images,test,train,val}},demo,models,preprocessing,scripts,tests,training,training_outputs,utils,web,mlruns}

# Create __init__.py files for Python packages
touch analysis/__init__.py models/__init__.py preprocessing/__init__.py training/__init__.py utils/__init__.py api/__init__.py
```

### Step 3: Core Configuration Files

Create the essential configuration files:

#### requirements.txt
```txt
opencv-python==4.8.1.78
torch>=2.0.0
torchvision>=0.15.0
tensorflow==2.20.0rc0
scikit-image>=0.20.0
pillow>=9.5.0
numpy>=1.24.0
transformers>=4.36.0
timm>=0.9.0
albumentations>=1.3.0
scipy>=1.10.0
scikit-learn>=1.2.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.4.0
pytest>=7.0.0
mlflow>=2.0.0
wandb>=0.15.0
optuna>=3.0.0
```

#### configs/training_config.json
```json
{
  "experiment_name": "composition_analysis_training",
  "run_name": "hybrid_cnn_vit_v1",
  "output_dir": "./training_outputs",
  "seed": 42,
  "data": {
    "train_data_dir": "./datasets/synthetic_cadb/train",
    "val_data_dir": "./datasets/synthetic_cadb/val",
    "test_data_dir": "./datasets/synthetic_cadb/test",
    "train_annotations": "./datasets/synthetic_cadb/annotations/train.csv",
    "val_annotations": "./datasets/synthetic_cadb/annotations/val.csv",
    "test_annotations": "./datasets/synthetic_cadb/annotations/test.csv",
    "dataset_type": "synthetic",
    "target_size": [224, 224],
    "num_workers": 4,
    "color_jitter_strength": 0.2,
    "rotation_degrees": 5
  },
  "model": {
    "img_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 384,
    "num_attention_heads": 6,
    "num_hidden_layers": 6,
    "backbone": "resnet50",
    "dropout": 0.1,
    "pretrained_path": null
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "gradient_clip_value": 1.0,
    "save_every_n_epochs": 10,
    "validation_frequency": 5,
    "early_stopping_patience": 20,
    "lr_scheduler": "cosine",
    "warmup_epochs": 5
  },
  "loss": {
    "composition_weight": 1.0,
    "aesthetic_weight": 0.5,
    "technical_weight": 0.3,
    "emd_lambda": 0.1
  },
  "mlflow": {
    "tracking_uri": "./mlruns",
    "experiment_name": "composition_analysis"
  }
}
```

### Step 4: Environment Setup
```bash
# Create conda environment
conda create -n assistant python=3.13.5
conda activate assistant

# Install Python dependencies
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### Step 5: Restore Source Code

You'll need to recreate the main source files. Here are the key files to create:

#### Core Model Files (models/)
- `models/__init__.py`
- `models/hybrid_net.py` - Main hybrid CNN-ViT architecture
- `models/feature_detectors.py` - Feature detection components

#### Training Pipeline (training/)
- `training/__init__.py`
- `training/trainer.py` - Main training loop
- `training/dataset_loader.py` - Data loading utilities
- `training/losses.py` - Custom loss functions
- `training/metrics.py` - Evaluation metrics
- `training/clip_transfer.py` - CLIP transfer learning
- `training/hyperparameter_optimization.py` - Hyperparameter tuning

#### Analysis Components (analysis/)
- `analysis/__init__.py`
- `analysis/composition_analyzer.py` - Main composition analysis
- `analysis/rule_evaluators.py` - Rule of thirds, leading lines, etc.
- `analysis/aesthetic_quality.py` - Aesthetic scoring
- `analysis/scoring_algorithms.py` - Scoring utilities
- `analysis/suggestion_engine.py` - Improvement suggestions

#### Preprocessing (preprocessing/)
- `preprocessing/__init__.py`
- `preprocessing/image_preprocessor.py` - Image preprocessing pipeline

#### Utilities (utils/)
- `utils/__init__.py`
- `utils/synthetic_dataset_generator.py` - Synthetic data generation
- `utils/dataset_setup.py` - Dataset setup utilities
- `utils/validation.py` - Validation utilities
- `utils/validation_api.py` - API validation

#### API (api/)
- `api/__init__.py`
- `api/main.py` - FastAPI application

#### Main Scripts
- `train.py` - Main training script
- `demo_inference.py` - Inference demo
- `setup_training.py` - Training setup script
- `generate_realistic_dataset.py` - Dataset generation

### Step 6: Dataset Setup

#### Option A: Generate Synthetic Dataset (Quick Start)
```bash
# Activate environment
conda activate assistant

# Generate synthetic dataset for testing
python setup_training.py
```

#### Option B: Download Real Datasets (Production Training)

**AVA Dataset (Recommended):**
```bash
# Method 1: Using HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('Iceclear/AVA')
dataset.save_to_disk('./datasets/ava')
"

# Method 2: Manual download from Kaggle
# 1. Go to: https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment
# 2. Download and extract to ./datasets/ava/

# Method 3: MEGA Cloud
# Visit: https://mega.nz/folder/9b520Lzb#2gIa1fgAzr677dcHKxjmtQ
# Download all 64 7z files and extract
```

**CADB Dataset (Alternative):**
```bash
# Note: Original CADB repository is no longer available
# You may need to find alternative sources or use AVA dataset
```

### Step 7: Docker Setup (Optional)

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/models/cache /app/logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/models/cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

#### docker-compose.yml
```yaml
version: '3.8'
services:
  composition-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: composition-assistant-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - MODEL_CACHE_DIR=/app/models/cache
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
      - model_cache:/app/models/cache
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: composition-assistant-nginx
    ports:
      - "80:80"
    volumes:
      - ./web:/usr/share/nginx/html
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - composition-api
    restart: unless-stopped

volumes:
  model_cache:
```

### Step 8: Training and Testing

#### Start Training
```bash
# Activate environment
conda activate assistant

# Start training with synthetic dataset
python train.py --config configs/training_config.json

# Or with real dataset (after downloading AVA)
python train.py --config configs/training_config.json --dataset ava
```

#### Run Tests
```bash
# Run inference demo
python demo_inference.py

# Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run with Docker
docker-compose up -d
```

## 🔧 Development Tools

### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

### Weights & Biases (Optional)
```bash
# Login to wandb
wandb login

# Configure in training config or environment
export WANDB_PROJECT="composition-assistant"
```

## 📚 Key Features to Implement

### Essential Components
1. **Hybrid CNN-ViT Model**: ResNet50 + Vision Transformer
2. **Multi-Rule Detection**: Rule of thirds, leading lines, symmetry
3. **Real-time Processing**: <200ms inference time
4. **Transfer Learning**: CLIP-based pretraining
5. **Multi-task Learning**: Aesthetic + composition scoring

### Training Pipeline
1. **Data Augmentation**: Albumentations-based pipeline
2. **Loss Functions**: Earth Mover's Distance for ordinal scoring
3. **Hyperparameter Optimization**: Optuna integration
4. **Model Quantization**: TensorFlow Lite for mobile deployment

### API and Deployment
1. **FastAPI Backend**: REST API for image analysis
2. **Web Interface**: HTML/CSS/JS frontend
3. **Docker Deployment**: Production-ready containers
4. **Model Serving**: Real-time inference endpoints

## 🚨 Important Notes

### Before Removing Current Project
1. **Backup Configurations**: Save all config files
2. **Export Environment**: `conda env export > environment.yml`
3. **Save Trained Models**: Copy any trained model weights
4. **Document Custom Changes**: Note any modifications made to the codebase

### Performance Optimization
- Use GPU acceleration for training (CUDA)
- Enable mixed precision training for faster performance
- Implement model quantization for deployment
- Use data loading optimization (multiple workers)

### Security Considerations
- Secure API endpoints with authentication
- Validate input images for security
- Implement rate limiting for production deployment
- Use HTTPS in production

## 🎯 Next Steps After Setup

1. **Generate/Download Dataset**: Start with synthetic data, then move to real datasets
2. **Train Initial Model**: Train on small dataset to verify pipeline
3. **Hyperparameter Tuning**: Use Optuna for optimization
4. **Model Evaluation**: Test on validation set
5. **Deploy API**: Set up production environment
6. **Create Web Interface**: Build user-friendly frontend

## 📞 Troubleshooting

### Common Issues
- **CUDA not available**: Install proper NVIDIA drivers and CUDA toolkit
- **Memory errors**: Reduce batch size in training config
- **Package conflicts**: Use conda environment isolation
- **Model loading errors**: Check file paths and permissions

### Performance Issues
- **Slow training**: Enable GPU acceleration, increase batch size
- **Out of memory**: Reduce batch size, use gradient accumulation
- **Slow inference**: Implement model quantization, use TensorRT

This guide provides everything needed to restore and continue your AI Photography Composition Assistant project. Follow the steps sequentially, and you'll have a fully functional development environment ready for training and deployment.
