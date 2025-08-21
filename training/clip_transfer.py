"""
CLIP-based transfer learning for aesthetic understanding.

This module implements transfer learning from CLIP models for enhanced
aesthetic understanding and composition analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class CLIPAestheticAdapter(nn.Module):
    """
    Adapter module for incorporating CLIP aesthetic understanding into composition models.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32",
                 freeze_clip: bool = True, aesthetic_dim: int = 256):
        """
        Initialize CLIP aesthetic adapter.
        
        Args:
            clip_model_name: Name of the CLIP model to use
            freeze_clip: Whether to freeze CLIP weights
            aesthetic_dim: Dimension of aesthetic feature space
        """

        super().__init__()

        self.clip_model_name = clip_model_name
        self.freeze_clip = freeze_clip
        self.aesthetic_dim = aesthetic_dim

        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get CLIP feature dimensions
        clip_dim = self.clip_model.config.vision_config.hidden_size

        # Aesthetic projection  layers
        self.aesthetic_projector = nn.Sequential(
            nn.Linear(clip_dim, aesthetic_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(aesthetic_dim * 2, aesthetic_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(aesthetic_dim, aesthetic_dim)
        )

        # Composition-specific projectors
        self.composition_projectors = nn.ModuleDict({
            'rule_of_thirds': nn.Linear(aesthetic_dim, 64),
            'leading_lines': nn.Linear(aesthetic_dim, 64),
            'symmetry': nn.Linear(aesthetic_dim, 64),
            'depth': nn.Linear(aesthetic_dim, 64)
        })

        # Text-image alignment for aesthetic concepts
        self.aesthetic_concepts = [
            "a beautiful photograph",
            "professional photography",
            "artistic composition",
            "award-winning photo",
            "visually striking image",
            "well-composed photograph",
            "aesthetic appeal",
            "visual harmony"
        ]

        # Precompute text embeddings for aesthetic concepts
        self.register_buffer('aesthetic_text_embeddings', 
                           self._compute_text_embeddings())
        
    def _compute_text_embeddings(self) -> torch.Tensor:
        """Precompute text embeddings for aesthetic concepts."""

        with torch.no_grad():
            text_inputs = self.clip_processor(
                text = self.aesthetic_concepts,
                return_tensors = "pt",
                padding = True,
                truncation = True
            )

            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim = -1)

        return text_features
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLIP aesthetic adapter.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Dictionary of aesthetic features and scores
        
        """

        batch_size = images.size(0)

        # Get CLIP vision features
        with torch.set_grad_enabled(not self.freeze_clip):
            vision_features = self.clip_model.get_image_features(images)
            vision_features = F.normalize(vision_features, dim = -1)

        # Project to aesthetic space
        aesthetic_features = self.aesthetic_projector(vision_features)

        # Compute aesthetic alignment scores
        aesthetic_scores = torch.matmul(
            F.normalize(aesthetic_features, dim = -1),
            self.aesthetic_text_embeddings.T
        )  # [B, num_concepts]

        # Overall aesthetic score (max alignment)
        overall_aesthetic = torch.max(aesthetic_scores, dim = 1)[0]

        # Composition-specific features
        composition_features = {}
        for task, projector in self.composition_projectors.items():
            composition_features[task] = projector(aesthetic_features)
        
        return {
            'aesthetic_features': aesthetic_features,
            'aesthetic_scores': aesthetic_scores,
            'overall_aesthetic': overall_aesthetic,
            'composition_features': composition_features
        }
    
class CLIPEnhancedCompositionNet(nn.Module):
    """
    Composition network enhanced with CLIP aesthetic understanding.
    """

    def __init__(self, base_model: nn.Module, clip_adapter: CLIPAestheticAdapter,
                 fusion_method: str = 'attention'):
        
        """
        Initialize CLIP-enhanced composition network.
        
        Args:
            base_model: Base composition model (HybridCompositionNet)
            clip_adapter: CLIP aesthetic adapter
            fusion_method: Method for fusing CLIP and base features ('concat', 'attention', 'gated')
        
        """

        super().__init__()

         
        self.base_model = base_model
        self.clip_adapter = clip_adapter
        self.fusion_method = fusion_method
        
        # Get feature dimensions
        self.base_hidden_size = base_model.vit.d_model if hasattr(base_model, 'vit') else 768
        self.clip_aesthetic_dim = clip_adapter.aesthetic_dim

        # Fusion layers
        if fusion_method == 'attention':
            self.fusion_layers = self._create_attention_fusion()
        elif fusion_method == 'gated':
            self.fusion_layers = self._create_gated_fusion()
        elif fusion_method == 'concat':
            self.fusion_layers = self._create_concat_fusion()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
    # Enhanced task heads with CLIP features
        self.enhanced_heads = nn.ModuleDict({
            'rule_of_thirds': EnhancedCompositionHead(
                self.base_hidden_size, 64, 'rule_of_thirds'
            ),
            'leading_lines': EnhancedCompositionHead(
                self.base_hidden_size, 64, 'leading_lines'
            ),
            'symmetry': EnhancedCompositionHead(
                self.base_hidden_size, 64, 'symmetry'
            ),
            'depth': EnhancedCompositionHead(
                self.base_hidden_size, 64, 'depth'
            )
        })

    def _create_attention_fusion(self) -> nn.ModuleDict:
        """Create attention-based fusion layers."""
        return nn.ModuleDict({
            'cross_attention': nn.MultiheadAttention(
                embed_dim=self.base_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ),
            'layer_norm': nn.LayerNorm(self.base_hidden_size),
            'ffn': nn.Sequential(
                nn.Linear(self.base_hidden_size, self.base_hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.base_hidden_size * 2, self.base_hidden_size)
            )
        })
    
    def _create_gated_fusion(self) -> nn.ModuleDict:
        """Create gated fusion layers."""

        return nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(self.base_hidden_size + self.clip_aesthetic_dim, self.base_hidden_size),
                nn.Sigmoid()
            ),
            'transform': nn.Linear(self.clip_aesthetic_dim, self.base_hidden_size)
        })
    
    def _create_conat_fusion(self) -> nn.ModuleDict:
        """Create concatenation-based fusion layers."""

        return nn.ModuleDict({
            'projection': nn.Linear(
                self.base_hidden_size + self.clip_aesthetic_dim, 
                self.base_hidden_size
            ),
            'layer_norm': nn.LayerNorm(self.base_hidden_size)
        })
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced composition network.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Dictionary of composition predictions with aesthetic enhancement
        """

        # Get base model features
        base_outputs = self.base_model(images)
        
        # Get CLIP aesthetic features
        clip_outputs = self.clip_adapter(images)
        
        # Extract base features (before task heads)
        # This requires modifying the base model to return intermediate features
        base_features = self._extract_base_features(images)
        
        # Fuse features
        fused_features = self._fuse_features(base_features, clip_outputs)
        
        # Enhanced predictions
        enhanced_predictions = {}
        for task in ['rule_of_thirds', 'leading_lines', 'symmetry', 'depth']:
            clip_task_features = clip_outputs['composition_features'][task]
            enhanced_predictions[task] = self.enhanced_heads[task](
                fused_features, clip_task_features
            )
        
        # Add aesthetic scores to outputs
        enhanced_predictions['aesthetic_scores'] = clip_outputs['aesthetic_scores']
        enhanced_predictions['overall_aesthetic'] = clip_outputs['overall_aesthetic']
        
        return enhanced_predictions
    
    def _extract_base_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from base model."""

        with torch.no_grad():
            cnn_features = self.base_model.backbone(images)
            fused_features = self.base_model.fpn(cnn_features)
            patches = self.base_model.patch_embed(fused_features)
            B, C, H, W = patches.shape
            patches = patches.permute(0, 2, 3, 1).reshape(B, H * W, C)
            patches = self.base_model.patch_norm(patches)
            patches = patches + self.base_model.pos_embed[:, :H*W]
            transformer_output = self.base_model.vit(patches)
            
            # Global average pooling
            base_features = torch.mean(transformer_output, dim=1)  # [B, hidden_size]
            
        return base_features
    
    def _fuse_features(self, base_features: torch.Tensor, 
                      clip_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse base and CLIP features."""
        aesthetic_features = clip_outputs['aesthetic_features']
        
        if self.fusion_method == 'attention':
            # Cross-attention between base and aesthetic features
            base_features_expanded = base_features.unsqueeze(1)  # [B, 1, hidden_size]
            aesthetic_features_expanded = aesthetic_features.unsqueeze(1)  # [B, 1, aesthetic_dim]
            
            # Project aesthetic features to base dimension
            aesthetic_proj = self.fusion_layers['transform'](aesthetic_features_expanded)
            
            # Cross-attention
            attended_features, _ = self.fusion_layers['cross_attention'](
                base_features_expanded, aesthetic_proj, aesthetic_proj
            )
            
            # Residual connection and normalization
            fused = self.fusion_layers['layer_norm'](
                base_features_expanded + attended_features
            )
            
            # Feed-forward
            fused = fused + self.fusion_layers['ffn'](fused)
            
            return fused.squeeze(1)  # [B, hidden_size]
        
        elif self.fusion_method == 'gated':
            # Gated fusion
            aesthetic_transformed = self.fusion_layers['transform'](aesthetic_features)
            
            # Compute gate
            gate_input = torch.cat([base_features, aesthetic_features], dim=-1)
            gate = self.fusion_layers['gate'](gate_input)
            
            # Apply gate
            fused = gate * base_features + (1 - gate) * aesthetic_transformed
            
            return fused
        
        elif self.fusion_method == 'concat':
            # Simple concatenation and projection
            concatenated = torch.cat([base_features, aesthetic_features], dim=-1)
            fused = self.fusion_layers['projection'](concatenated)
            fused = self.fusion_layers['layer_norm'](fused)
            
            return fused
        
class EnhancedCompositionHead(nn.Module):
    """
    Enhanced composition head with CLIP aesthetic features.
    """
    
    def __init__(self, base_dim: int, clip_dim: int, task_type: str):
        """
        Initialize enhanced composition head.
        
        Args:
            base_dim: Base feature dimension
            clip_dim: CLIP feature dimension
            task_type: Type of composition task
        """
        super().__init__()
        
        self.task_type = task_type
        self.base_dim = base_dim
        self.clip_dim = clip_dim
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(base_dim + clip_dim, base_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific output layers (same as original CompositionHead)
        if task_type == 'rule_of_thirds':
            self.output = nn.Linear(base_dim, 9)
        elif task_type == 'leading_lines':
            self.output = nn.Linear(base_dim, 5)
        elif task_type == 'symmetry':
            self.output = nn.Linear(base_dim, 4)
        elif task_type == 'depth':
            self.output = nn.Linear(base_dim, 1)
    
    def forward(self, base_features: torch.Tensor, 
                clip_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced head.
        
        Args:
            base_features: Base model features [B, base_dim]
            clip_features: CLIP features [B, clip_dim]
            
        Returns:
            Task-specific predictions
        """
        # Fuse features
        combined = torch.cat([base_features, clip_features], dim=-1)
        fused = self.fusion(combined)
        
        # Task-specific output
        output = self.output(fused)
        
        # Apply task-specific activations
        if self.task_type == 'rule_of_thirds':
            return torch.sigmoid(output)
        elif self.task_type == 'leading_lines':
            return torch.sigmoid(output)
        elif self.task_type == 'symmetry':
            return torch.softmax(output, dim=-1)
        elif self.task_type == 'depth':
            return torch.relu(output)
        
class CLIPAestheticLoss(nn.Module):
    """
    Loss function incorporating CLIP aesthetic understanding.
    """
    
    def __init__(self, aesthetic_weight: float = 0.1):
        """
        Initialize CLIP aesthetic loss.
        
        Args:
            aesthetic_weight: Weight for aesthetic loss component
        """
        super().__init__()
        self.aesthetic_weight = aesthetic_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute aesthetic-enhanced loss.
        
        Args:
            predictions: Model predictions including aesthetic scores
            targets: Target values
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Standard composition losses (handled by base loss function)
        # This would typically be computed by the main multi-task loss
        
        # Aesthetic alignment loss
        if 'aesthetic_scores' in predictions and 'overall_quality' in targets:
            # Encourage high aesthetic scores for high-quality compositions
            quality_scores = targets['overall_quality'].squeeze()
            aesthetic_alignment = predictions['overall_aesthetic']
            
            # Normalize quality scores to [0, 1] if needed
            if quality_scores.max() > 1.0:
                quality_scores = quality_scores / quality_scores.max()
            
            aesthetic_loss = self.mse_loss(aesthetic_alignment, quality_scores)
            total_loss += self.aesthetic_weight * aesthetic_loss
        
        return total_loss

def create_clip_enhanced_model(base_model: nn.Module, 
                             clip_model_name: str = "openai/clip-vit-base-patch32",
                             freeze_clip: bool = True,
                             fusion_method: str = 'attention') -> CLIPEnhancedCompositionNet:
    """
    Create a CLIP-enhanced composition model.
    
    Args:
        base_model: Base composition model
        clip_model_name: CLIP model to use
        freeze_clip: Whether to freeze CLIP weights
        fusion_method: Feature fusion method
        
    Returns:
        CLIP-enhanced composition model
    """
    # Create CLIP adapter
    clip_adapter = CLIPAestheticAdapter(
        clip_model_name=clip_model_name,
        freeze_clip=freeze_clip
    )
    
    # Create enhanced model
    enhanced_model = CLIPEnhancedCompositionNet(
        base_model=base_model,
        clip_adapter=clip_adapter,
        fusion_method=fusion_method
    )
    
    return enhanced_model


def fine_tune_with_aesthetic_data(model: CLIPEnhancedCompositionNet,
                                aesthetic_dataset: torch.utils.data.Dataset,
                                num_epochs: int = 10,
                                learning_rate: float = 1e-5) -> None:
    """
    Fine-tune model with aesthetic-specific data.
    
    Args:
        model: CLIP-enhanced composition model
        aesthetic_dataset: Dataset with aesthetic annotations
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
    """
    optimizer = torch.optim.AdamW(
        model.clip_adapter.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    dataloader = torch.utils.data.DataLoader(
        aesthetic_dataset, batch_size=32, shuffle=True
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            images = batch['image']
            aesthetic_scores = batch['aesthetic_score']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Aesthetic loss
            aesthetic_loss = F.mse_loss(
                outputs['overall_aesthetic'],
                aesthetic_scores
            )
            
            aesthetic_loss.backward()
            optimizer.step()
            
            total_loss += aesthetic_loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Aesthetic fine-tuning epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    logger.info("Aesthetic fine-tuning completed")