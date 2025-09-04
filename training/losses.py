"""
Loss functions for composition analysis training.

This module implements specialized loss functions including Earth Mover's Distance
loss for ordinal composition scoring and multi-task losses for joint training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

class EarthMoversDistance(nn.Module):
    """
    Earth Mover's Distance (EMD) loss for ordinal regression.
    
    Particularly effective for composition scoring where the ordinal nature
    of quality ratings is important.
    
    """

    def __init__(self, num_classes: int = 10, normalize: bool = True):
        """
        Initialize EMD loss.
        
        Args:
            num_classes: Number of quality score classes (e.g., 1-10)
            normalize: Whether to normalize the loss
        
        """

        super().__init__()
        self.num_classes = num_classes
        self.normalize = normalize

        # Create distance matrix for EMD computation

        self.register_buffer('distance_matrix', self._create_distance_matrix())

    def _create_distance_matrix(self) -> torch.Tensor:
        """Create distance matrix for EMD computation."""

        distances = torch.zeros(self.num_classes, self.num_classes)

        for i in range (self.num_classes):
            for j in range (self.num_classes):
                distances[i, j] = abs(i - j)

        return distances
    
    def forward(self, pred_probs: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute EMD loss.
        
        Args:
            pred_probs: Predicted probability distribution [B, num_classes]
            target_scores: Target scores [B] (will be converted to distribution)
            
        Returns:
            EMD loss value
        
        """

        batch_size = pred_probs.size(0)

        # Convert target scores to probability distributions

        target_probs = self._scores_to_distribution(target_scores)

        # Compute cumulative distributions

        pred_cdf = torch.cumsum(pred_probs, dim = 1)
        target_cdf = torch.cumsum(target_probs, dim = 1)

        # Compute EMD as sum of absolute differences between CDFs

        emd_loss = torch.sum(torch.abs(pred_cdf - target_cdf), dim = 1)

        if self.normalize:
            emd_loss = emd_loss / self.num_classes

        return emd_loss.mean()
    
    def _scores_to_distribution(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous scores to probability distributions.
        
        Args:
            scores: Continuous scores [B]
            
        Returns:
            Probability distributions [B, num_classes]
        
        """

        batch_size = scores.size(0)

        # Clamp scores to valid range

        scores = torch.clamp(scores, 0, self.num_classes - 1)

        # Create soft distributions around the target scores

        distributions = torch.zeros(batch_size, self.num_classes, devices = scores.device)

        for i, score in enumerate(scores):
            # Create a Gaussian-like distribution centered on the score

            sigma = 0.5 # Controls the spread

            for j in range(self.num_classes):
                dist = abs(j - score.item())
                distributions[i, j] = torch.exp(-0.5 * (dist / sigma) ** 2)

            # Normalize to make it a probability distribution

            distributions[i] = distributions[i] / distributions[i].sum()

        return distributions
    
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in composition analysis.
    
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        
        """

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [B, C]
            targets: Target labels [B]

        Returns:
            Focal loss value
        
        """

        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        
        elif self.reduction == 'sum':
            return focal_loss.sum()
        
        else:
            return focal_loss
        
class CompositionMultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint training of all composition rules.
    
    """

    def __init__(self, task_weights: Optional[Dict[str, float]] = None,
                 uncertainty_weighting: bool = True):
        """
        Initialize multi-task loss.

        Args:
            task_weights: Manual weights for each task
            uncertainty_weighting: Use learnable uncertainty-based weighting
        
        """

        super().__init__()

        # Default task weights

        self.task_weights = task_weights or {
            'rule_of_thirds': 1.0,
            'leading_lines': 1.0,
            'symmetry': 1.0,
            'depth': 1.0,
            'overall_quality': 0.5
        }

        self.uncertainty_weighting = uncertainty_weighting

        # Learnable uncertainty parameters (log variance)

        if uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1, dtype=torch.float32))

                for task in self.task_weights.keys()
            })

        # Individual loss functions

        self.bce_loss = nn.BCEWithLogitsLoss()  # Safe for mixed precision training
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.emd_loss = EarthMoversDistance(num_classes = 10)

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            
        Returns:
            Dictionary containing individual and total losses
        
        """

        losses = {}
        total_loss = 0.0

        # Rule of thirds loss (BCE for grid points)

        if 'rule_of_thirds' in predictions:
            # Add small epsilon to prevent numerical issues
            pred_rot = torch.clamp(predictions['rule_of_thirds'], min=-10, max=10)
            rot_loss = self.bce_loss(pred_rot, targets['rule_of_thirds'])
            
            # Check for NaN and replace with zero if found
            if torch.isnan(rot_loss):
                rot_loss = torch.tensor(0.0, device=pred_rot.device, dtype=torch.float32)
            
            losses['rule_of_thirds'] = rot_loss

        # Leading lines loss (MSE for line parameters)

        if 'leading_lines' in predictions:
            pred_lines = torch.clamp(predictions['leading_lines'], min=0, max=1)
            lines_loss = self.mse_loss(pred_lines, targets['leading_lines'])
            
            # Check for NaN and replace with zero if found
            if torch.isnan(lines_loss):
                lines_loss = torch.tensor(0.0, device=pred_lines.device, dtype=torch.float32)
            
            losses['leading_lines'] = lines_loss

        # Symmetry loss (Cross entropy for type classification)

        if 'symmetry' in predictions:
            # Convert one-hot targets to class indices
            symmetry_targets = targets['symmetry'].argmax(dim = 1)
            symmetry_loss = self.ce_loss(predictions['symmetry'], symmetry_targets)
            
            # Check for NaN and replace with zero if found
            if torch.isnan(symmetry_loss):
                symmetry_loss = torch.tensor(0.0, device=predictions['symmetry'].device, dtype=torch.float32)
            
            losses['symmetry'] = symmetry_loss

        # Depth loss (MSE for depth scores)

        if 'depth' in predictions:
            pred_depth = torch.clamp(predictions['depth'], min=0, max=1)
            depth_loss = self.mse_loss(pred_depth, targets['depth'])
            
            # Check for NaN and replace with zero if found
            if torch.isnan(depth_loss):
                depth_loss = torch.tensor(0.0, device=pred_depth.device, dtype=torch.float32)

            losses['depth'] = depth_loss

        # Overall quality loss (EMD for ordinal scoring)

        if 'overall_quality' in targets:
            # Convert continuous scores to probability distribution for EMD

            quality_scores = targets['overall_quality'].squeeze()

            # For this, we'll use MSE as a simpler alternative to EMD

            if 'overall_quality' in predictions:
                quality_loss = self.mse_loss(predictions['overall_quality'], 
                                             targets['overall_quality'])
                
            else:
                # If no direct overall quality prediction, use weighted sum of task scores

                weighted_pred = (
                    predictions.get('rule_of_thirds', torch.zeros_like(
                        targets['overall_quality'])).mean(dim = 1, keepdim = True) * 0.25 +
                    predictions.get('leading_lines', torch.zeros_like(
                        targets['overall_quality']))[:, -1:] * 0.25 +
                    predictions.get('symmetry', torch.zeros_like(
                        targets['overall_quality'])).max(dim = 1, keepdim = True)[0] * 0.25 +
                    predictions.get('depth', torch.zeros_like(
                        targets['overall_quality'])) * 0.25
                )

                quality_loss = self.mse_loss(weighted_pred, targets['overall_quality'])

            losses['overall_quality'] = quality_loss

        # Combine losses with weighting
        # Get device from first available loss tensor
        device = next(iter(losses.values())).device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for task, loss in losses.items():
            # Ensure loss is float32 and check for NaN/inf
            loss = loss.float()
            
            # Skip NaN or infinite losses
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Skipping {task} loss due to NaN/inf: {loss}")
                continue
            
            if self.uncertainty_weighting and task in self.log_vars:
                # Uncertainty-based weighting: L = sum_i (1/2σ²)L_i + log(σ²)
                # Ensure tensors are on the same device and use float32
                log_var = self.log_vars[task].to(device).float()
                
                # Clamp log_var to prevent extreme values
                log_var = torch.clamp(log_var, min=-10, max=10)
                precision = torch.exp(-log_var)
                
                # Scale the loss to prevent explosion
                weighted_loss = (0.5 * precision * loss + 0.5 * log_var).float()

            else:
                # Manual weighting with scaling
                weight = torch.tensor(self.task_weights.get(task, 1.0), 
                                    device=device, dtype=torch.float32)
                weighted_loss = (weight * loss).float()

            total_loss = total_loss + weighted_loss
            
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Total loss is NaN/inf, replacing with zero")
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            
        losses['total'] = total_loss
        return losses

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning composition similarity.
    
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for contrastive learning
            temperature: Temperature parameter for softmax
        
        """

        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features: torch.Tensor, composition_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss based on composition similarity.
        
        Args:
            features: Feature representations [B, D]
            composition_scores: Overall composition scores [B]
            
        Returns:
            Contrastive loss value
        
        """

        batch_size = features.size(0)

        # Normalize features
        features = F.normalize(features, dim = 1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create labels based on composition score similarity
        score_diff = torch.abs(composition_scores.unsqueeze(0) - composition_scores.unsqueeze(1))
        similar_pairs = (score_diff < 0.01).float() # Threshold for similar compositions

        # Contrastive loss
        pos_mask = similar_pairs
        neg_mask = 1 - similar_pairs

        # Remove diagonal (self-similarity)
        mask = torch.eye(batch_size, device = features.device)
        pos_mask = pos_mask * (1 - mask)
        neg_mask = neg_mask * (1 - mask)

        # Compute positive and negative similarities
        pos_sim = similarity_matrix * pos_mask
        neg_sim = similarity_matrix * neg_mask

        # Contrastive loss computation
        pos_loss = -torch.log(torch.exp(pos_sim) + 1e-8).sum() / (pos_mask.sum() + 1e-8)
        neg_loss = torch.log(torch.exp(neg_sim) + 1e-8).sum() / (neg_mask.sum() + 1e-8)

        return pos_loss + neg_loss
    
class PercepturalLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features for composition understanding.
    
    """

    def __init__(self, feature_layers: List[int] = [3, 8, 15, 22]):
        """
        Initialize perceptual loss.

        Args:
            feature_layers: VGG layers to use for feature extraction
        
        """

        super().__init__()

        # Load pretrained VGG16
        from torchvision.models import vgg16
        vgg = vgg16(pretrained = True).features

        self.feature_extractors = nn.ModuleList()
        self.feature_layers = feature_layers

        # Create feature extractors for specified layers
        current_layer = 0
        current_extractor = nn.Sequential()

        for i, layer in enumerate(vgg):
            current_extractor.add_module(str(i), layer)

            if i in feature_layers:
                self.feature_extractors.append(current_extractor)
                current_extractor = nn.Sequential()

        # Freeze parameters
        for extractor in self.feature_extractors:
            for param in extractor.parameters():
                param.requires_grad = False

        
    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between input and target images.
        
        Args:
            input_img: Input image [B, C, H, W]
            target_img: Target image [B, C, H, W]
            
        Returns:
            Perceptual loss value
        
        """

        perceptual_loss = 0.0

        for extractor in self.feature_extractors:
            input_features = extractor(input_img)
            target_features = extractor(target_img)

            perceptual_loss += F.mse_loss(input_features, target_features)

        return perceptual_loss / len(self.feature_extractors)
    
def get_loss_function(config: Dict) -> nn.Module:
        """
        Factory function to create loss function based on configuration.
    
        Args:
            config: Loss configuration dictionary
        
        Returns:
            Configured loss function
        
        """

        loss_type = config.get('type', 'multi-task')

        if loss_type == 'multi_task':
            return CompositionMultiTaskLoss(
                task_weights = config.get('task_weights'),
                uncertainty_weighting = config.get('uncertainty_weighting', True)
            )
        
        elif loss_type == 'emd':
            return EarthMoversDistance(
                num_classes = config.get('num_classes', 10),
                normalize = config.get('normalize', True)
            )
        
        elif loss_type == 'focal':
            return FocalLoss(
                alpha = config.get('alpha', 1.0),
                gamma = config.get('gamma', 2.0)
            )
        
        elif loss_type == 'contrastive':
            return ContrastiveLoss(
                margin = config.get('margin', 1.0),
                temperature = config.get('temperature', 0.07)
            )
        
        elif loss_type == 'perceptual':
            return PercepturalLoss(
                feature_layers = config.get('feature_layers', [3, 8, 15, 22])
            )
        
        else:
            raise ValueError("Unknown loss type: {loss_type}")