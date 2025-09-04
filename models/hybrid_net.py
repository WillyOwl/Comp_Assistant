"""
Hybrid CNN-ViT Network for Composition Analysis

This module implements a hybrid architecture combining Convolutional Neural Networks (CNN)
with Vision Transformers (ViT) for advanced feature detection in photographic composition analysis.
"""

import torch
import torch.nn as nn
import timm
from transformers import ViTModel, ViTConfig

class HybridCompositionNet(nn.Module):
    """
    Hybrid network combining CNN backbone with Vision Transformer for composition analysis.
    
    The architecture uses a CNN (ResNet) for local feature extraction followed by a Vision
    Transformer for global context understanding and compositional element detection.
    """
    
    def __init__(self, img_size=224, patch_size=16, num_channels=3,
                 hidden_size=768, num_attention_heads=12,
                 num_hidden_layers=12, backbone='resnet50'):
        """
        Initialize the hybrid network.
        
        Args:
            img_size: Input image size (default: 224)
            patch_size: Size of patches for ViT (default: 16)
            num_channels: Number of input channels (default: 3)
            hidden_size: Hidden size for transformer (default: 768)
            num_attention_heads: Number of attention heads (default: 12)
            num_hidden_layers: Number of transformer layers (default: 12)
            backbone: CNN backbone architecture (default: 'resnet50')
        """
        super(HybridCompositionNet, self).__init__()
        
        # CNN Backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        # Get backbone feature dimensions
        dummy_input = torch.zeros(1, num_channels, img_size, img_size)
        features = self.backbone(dummy_input)
        feature_dims = [f.shape[1] for f in features]
        
        # Feature Pyramid Network for multi-scale feature fusion
        self.fpn = FeaturePyramidNetwork(feature_dims)
        
        # Calculate patch embedding output size
        fpn_output_size = img_size // 32  # ResNet50 total stride
        patch_output_size = max(1, fpn_output_size // patch_size)  # Ensure minimum size of 1
        
        # Patch embedding layer - remove LayerNorm for spatial dimensions
        self.patch_embed = nn.Conv2d(feature_dims[-1], hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Layer norm for features after flattening
        self.patch_norm = nn.LayerNorm(hidden_size)
        
        # Position embeddings
        num_patches = patch_output_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # Transformer encoder (named as vit for compatibility with tests)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # Task-specific heads
        self.rule_of_thirds_head = CompositionHead(hidden_size, 'rule_of_thirds')
        self.leading_lines_head = CompositionHead(hidden_size, 'leading_lines')
        self.symmetry_head = CompositionHead(hidden_size, 'symmetry')
        self.depth_head = CompositionHead(hidden_size, 'depth')
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling to prevent exploding gradients."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Parameter):
                # Small random initialization for parameters like positional embeddings
                nn.init.normal_(module, 0.0, 0.02)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            dict: Dictionary containing predictions for different compositional elements
        """
        # Extract CNN features
        cnn_features = self.backbone(x)
        
        # FPN feature fusion
        fused_features = self.fpn(cnn_features)  # Shape: [B, C, H, W]
        
        # Convert to patches and embed
        patches = self.patch_embed(fused_features)  # Shape: [B, hidden_size, H', W']
        B, C, H, W = patches.shape
        patches = patches.permute(0, 2, 3, 1).reshape(B, H * W, C)  # Shape: [B, N, hidden_size]
        
        # Apply layer norm
        patches = self.patch_norm(patches)
        
        # Add position embeddings
        patches = patches + self.pos_embed[:, :H*W]
        
        # Pass through transformer
        transformer_output = self.vit(patches)  # Shape: [B, N, hidden_size]
        
        # Task-specific predictions
        return {
            'rule_of_thirds': self.rule_of_thirds_head(transformer_output),
            'leading_lines': self.leading_lines_head(transformer_output),
            'symmetry': self.symmetry_head(transformer_output),
            'depth': self.depth_head(transformer_output)
        }


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    """
    
    def __init__(self, feature_dims):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, feature_dims[-1], 1)
            for dim in feature_dims[:-1]
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(feature_dims[-1], feature_dims[-1], 3, padding=1)
            for _ in feature_dims[:-1]
        ])
        
        # Final 1x1 conv to reduce channels
        self.final_conv = nn.Conv2d(len(feature_dims) * feature_dims[-1], feature_dims[-1], 1)
        
    def forward(self, features):
        """
        Forward pass through FPN.
        
        Args:
            features: List of feature maps from CNN backbone
            
        Returns:
            torch.Tensor: Fused feature map
        """
        last_feature = features[-1]
        results = [last_feature]
        
        for feature, lateral_conv, output_conv in zip(
            features[:-1][::-1], self.lateral_convs[::-1], self.output_convs[::-1]
        ):
            # Upsample last feature and add lateral connection
            up_scaled = nn.functional.interpolate(
                last_feature, size=feature.shape[-2:], mode='nearest'
            )
            lateral = lateral_conv(feature)
            last_feature = up_scaled + lateral
            results.insert(0, output_conv(last_feature))
        
        # Fuse all feature maps and reduce channels
        fused = torch.cat([
            nn.functional.interpolate(result, size=features[0].shape[-2:], mode='nearest')
            for result in results
        ], dim=1)
        
        return self.final_conv(fused)  # Reduce channels to match original feature dimension


class CompositionHead(nn.Module):
    """
    Task-specific head for different compositional elements.
    """
    
    def __init__(self, input_dim, task_type):
        """
        Initialize composition head.
        
        Args:
            input_dim: Input feature dimension
            task_type: Type of compositional task
        """
        super(CompositionHead, self).__init__()
        
        self.task_type = task_type
        hidden_dim = input_dim // 2
        
        # Global average pooling for sequence
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific output layers
        if task_type == 'rule_of_thirds':
            # 9 points (3x3 grid) with confidence scores
            self.output = nn.Linear(hidden_dim // 2, 9)
        elif task_type == 'leading_lines':
            # Line parameters (start_x, start_y, end_x, end_y) with confidence
            self.output = nn.Linear(hidden_dim // 2, 5)
        elif task_type == 'symmetry':
            # Symmetry type (horizontal, vertical, radial) with confidence
            self.output = nn.Linear(hidden_dim // 2, 4)
        elif task_type == 'depth':
            # Depth map prediction
            self.output = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x):
        """
        Forward pass through the composition head.
        
        Args:
            x: Input features of shape (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: Task-specific predictions
        """
        # Handle both 2D and 3D input tensors
        if len(x.shape) == 2:
            # Input is (batch_size, hidden_dim), skip pooling
            pass
        elif len(x.shape) == 3:
            # Input is (batch_size, seq_len, hidden_dim), apply pooling
            x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
            x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        else:
            raise ValueError(f"Expected input tensor with 2 or 3 dimensions, got {len(x.shape)}")
        
        # MLP processing
        features = self.mlp(x)
        output = self.output(features)
        
        # Task-specific output processing
        if self.task_type == 'rule_of_thirds':
            return output.float()  # Raw logits for BCEWithLogitsLoss
        elif self.task_type == 'leading_lines':
            return torch.sigmoid(output).float()  # Normalized coordinates and confidence
        elif self.task_type == 'symmetry':
            return torch.softmax(output, dim=-1).float()  # Probability distribution
        elif self.task_type == 'depth':
            return torch.relu(output).float()  # Non-negative depth values