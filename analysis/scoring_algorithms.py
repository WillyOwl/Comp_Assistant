#!/usr/bin/env python3
"""
Composition Scoring Algorithms

This module implements advanced scoring algorithms for compositional analysis
including Earth Mover's Distance loss and multi-task scoring approaches.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

class EMDLoss(nn.Module):
    """
    Earth Mover's Distance Loss for Composition Quality Assessment.
    
    Implements EMD loss to measure distribution differences for ordinal
    quality ratings, addressing the ordinal nature of composition scores.
    """

    def __init__(self, num_classes: int = 5, device: Optional[torch.device] = None):
        """
        Initialize EMD Loss.
        
        Args:
            num_classes: Number of quality classes (e.g., 1-5 rating scale)
            device: PyTorch device for computation

        """

        super(EMDLoss, self).__init__()

        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create cost matrix for EMD computation
        self.register_buffer('cost_matrix', self._create_cost_matrix())

    def _create_cost_matrix(self) -> torch.Tensor:
        """Create cost matrix for EMD computation."""

        cost_matrix = torch.zeros(self.num_classes, self.num_classes)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                cost_matrix[i, j] = abs(i - j)
        
        return cost_matrix
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute EMD loss between predicted and target distributions.
        
        Args:
            predicted: Predicted probability distribution (batch_size, num_classes)
            target: Target probability distribution (batch_size, num_classes)
            
        Returns:
            EMD loss value
        """

        batch_size = predicted.size(0)

        # Ensure distribution sum to 1
        predicted = F.softmax(predicted, dim = 1)

        # Compute cumulative distributions
        predicted_cdf = torch.cumsum(predicted, dim = 1)
        target_cdf = torch.cumsum(target, dim = 1)

        # Compute EMD as L1 distance between CDFs
        emd_loss = torch.mean(torch.sum(torch.abs(predicted_cdf - target_cdf), dim = 1))

        return emd_loss
    
class CompositionScorer:
    """
    Main composition scoring engine.
    
    Provides unified interface for calculating scores across different
    compositional rules and quality metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize composition scorer.

        Args:
            config: Configuration dictionary for scording parameters

        """

        self.config = config or self._get_default_config()

        # Initialize scoring components
        self.rule_weights = self.config.get('rule_weights', {})
        self.use_emd = self.config.get('use_emd_loss', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

        logger.info("CompositionScorer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default scoring configuration."""

        return {
            'rule_weights': {
                'rule_of_thirds': 1.0,
                'leading_lines': 1.0,
                'symmetry': 1.0,
                'depth_layering': 1.0,
                'color_harmony': 1.0
            },
            'use_emd_loss': True,
            'confidence_threshold': 0.7,
            'quality_threshold': 0.5,
            'normalization_method': 'min_max'  # 'min_max', 'z_score', 'robust'
        }

    def calculate_rule_score(self, rule_name: str, rule_result: Dict[str, Any]) -> float:
        
        """
        Calculate normalized score for a specific compositional rule.
        
        Args:
            rule_name: Name of the compositional rule
            rule_result: Result dictionary from rule evaluator
            
        Returns:
            Normalized score between 0 and 1
        """

        try:
            # Get base score from rule result
            base_score = rule_result.get('score', 0.0)

            # Apply rule-specific scoring logic
            if rule_name == 'rule_of_thirds':
                score = self._score_rule_of_thirds(rule_result)
            
            elif rule_name == 'leading_lines':
                score = self._score_leading_lines(rule_result)

            elif rule_name == 'symmetry':
                score = self._score_symmetry(rule_result)

            elif rule_name == 'depth_layering':
                score = self._score_depth_layering(rule_result)

            elif rule_name == 'color_harmony':
                score = self._score_color_harmony(rule_result)

            else:
                score = base_score

            # Apply confidence weighting
            confidence = rule_result.get('confidence', 1.0)
            
            if confidence < self.confidence_threshold:
                score *= confidence / self.confidence_threshold

            return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.warning(f"Score calculation failed for {rule_name}: {str(e)}")
            return 0.0
        
    def _score_rule_of_thirds(self, result: Dict[str, Any]) -> float:
        """Score rule of thirds evaluation result."""

        base_score = result.get('score', 0.0)

        # Bonus for good intersection alignment
        intersection_scores = result.get('intersection_scores', [])
        if intersection_scores:
            intersection_bonus = min(0.2, np.mean(intersection_scores) * 0.2)
        
        else:
            intersection_bonus = 0.0

        # Bonus for visual weight distribution
        weight_dist = result.get('weight_distribution', {})
        balance_score = weight_dist.get('balance_score', 0.0)
        balance_bonus = min(0.1, balance_score * 0.1)

        return base_score + intersection_bonus + balance_bonus
    
    def _score_leading_lines(self, result: Dict[str, Any]) -> float:
        """Score leading lines evaluation result."""

        base_score = result.get('score', 0.0)

        # Bonus for vanishing points
        vanishing_points = result.get('vanishing_points', [])
        vp_bonus = min(0.2, len(vanishing_points) * 0.1)

        # Bonus for line diversity
        line_analysis = result.get('line_analysis', {})
        line_types = line_analysis.get('line_types', {})
        if sum(line_types.values()) > 0:
            diversity = len([t for t in line_types.values() if t > 0])
            diversity_bonus = min(0.1, diversity * 0.05)
        
        else:
            diversity_bonus = 0.0

        return base_score + vp_bonus + diversity_bonus
    
    def _score_symmetry(self, result: Dict[str, Any]) -> float:
        """Score symmetry evaluation result."""

        # Take the best symmetry type score
        horizontal = result.get('horizontal_score', 0.0)
        vertical = result.get('vertical_score', 0.0)
        radial = result.get('radial_score', 0.0)

        best_score = max(horizontal, vertical, radial)

        # Bonus for multiple symmetry types
        symmetry_count = sum(1 for score in [horizontal, vertical, radial] if score > 0.3)

        multi_symmetry_bonus = min(0.1, (symmetry_count - 1) * 0.05) if symmetry_count > 1 else 0.0

        return best_score + multi_symmetry_bonus
    
    def _score_depth_layering(self, result: Dict[str, Any]) -> float:
        """Score depth layering evaluation result."""

        base_score = result.get('score', 0.0)

        # Bonus for optimal layer count
        layers = result.get('layers', [])
        num_layers = len(layers)

        if 3 <= num_layers <= 4:
            layer_bonus = 0.1
        
        elif num_layers >= 2:
            layer_bonus = 0.05
        
        else:
            layer_bonus = 0.0

        # Bonus for focal point distribution
        focal_points = result.get('focal_points', [])

        if len(focal_points) >= 2:
            focal_depths = [fp['depth'] for fp in focal_points]
            focal_variance = np.var(focal_depths)
            focal_bonus = min(0.1, focal_variance * 0.2)

        else:
            focal_bonus = 0.0

        return base_score + layer_bonus + focal_bonus
    
    def _score_color_harmony(self, result: Dict[str, Any]) -> float:
        """Score color harmony evaluation result."""

        base_score = result.get('score', 0.0)

        # Bonus for specific harmony types
        harmony_scores = result.get('harmony_scores', {})

        # Complementary harmony bonus
        comp_bonus = min(0.1, harmony_scores.get('complementary', 0.0) * 0.1)

        # Analogous harmony bonus
        analog_bonus = min(0.1, harmony_scores.get('analogous', 0.0) * 0.1)

        # Balance bonuses
        sat_balance = harmony_scores.get('saturation_balance', 0.0)
        bright_balance = harmony_scores.get('brightness_balance', 0.0)
        balance_bonus = min(0.1, (sat_balance + bright_balance) * 0.05)

        return base_score + comp_bonus + analog_bonus + balance_bonus
    
    def calculate_weighted_score(self, rule_scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score from individual rule scores.
        
        Args:
            rule_scores: Dictionary of rule names to scores
            
        Returns:
            Weighted overall composition score
        """
        try:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for rule_name, score in rule_scores.items():
                weight = self.rule_weights.get(rule_name, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Weighted score calculation failed: {str(e)}")
            return 0.0
        
    def calculate_confidence_weighted_score(self, rule_results: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculate score weighted by confidence levels.
        
        Args:
            rule_results: Dictionary of rule evaluation results
            
        Returns:
            Tuple of (weighted_score, overall_confidence)
        """
        try:
            total_confidence_weight = 0.0
            confidence_weighted_sum = 0.0
            confidences = []
            
            for rule_name, result in rule_results.items():
                score = result.get('score', 0.0)
                confidence = result.get('confidence', 1.0)
                weight = self.rule_weights.get(rule_name, 1.0)
                
                confidence_weighted_sum += score * confidence * weight
                total_confidence_weight += confidence * weight
                confidences.append(confidence)
            
            if total_confidence_weight > 0:
                weighted_score = confidence_weighted_sum / total_confidence_weight
            else:
                weighted_score = 0.0
            
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            return weighted_score, overall_confidence
            
        except Exception as e:
            logger.warning(f"Confidence weighted score calculation failed: {str(e)}")
            return 0.0, 0.0

class MultiTaskScorer(nn.Module):
    """
    Multi-task neural network scorer for composition analysis.
    
    Jointly predicts aesthetic scores, composition quality, and technical
    attributes using shared representations.
    
    """

    def __init__(self, device: Optional[torch.device] = None):
        super(MultiTaskScorer, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(512, 256),  # Input from feature vector
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.aesthetic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5-class aesthetic rating
            nn.Softmax(dim=1)
        )
        
        self.composition_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Regression for composition score
            nn.Sigmoid()
        )
        
        self.technical_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Sharpness, contrast, exposure
        )
        
        # EMD Loss for aesthetic rating
        self.emd_loss = EMDLoss(num_classes=5, device=self.device)
        
        self.to(self.device)
        logger.info(f"MultiTaskScorer initialized on {self.device}")

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task network.
        
        Args:
            features: Input feature tensor (batch_size, 512)
            
        Returns:
            Dictionary of task predictions
        
        """

        # Shared feature extraction
        shared_features = self.shared_layers(features)

        # Task-specific predictions
        aesthetic_pred = self.aesthetic_head(shared_features)
        composition_pred = self.composition_head(shared_features)
        technical_pred = self.technical_head(shared_features)

        return {
            'aesthetic': aesthetic_pred,
            'composition': composition_pred,
            'technical': technical_pred
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Aesthetic loss (EMD)
        if 'aesthetic' in targets:
            losses['aesthetic'] = self.emd_loss(predictions['aesthetic'], targets['aesthetic'])
        
        # Composition loss (MSE)
        if 'composition' in targets:
            losses['composition'] = F.mse_loss(predictions['composition'], targets['composition'])
        
        # Technical loss (MSE for multiple attributes)
        if 'technical' in targets:
            losses['technical'] = F.mse_loss(predictions['technical'], targets['technical'])
        
        # Combined loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
class AdaptiveScorer:
    """
    Adaptive scoring system that learns from user feedback.
    
    Adjusts scoring weights and thresholds based on user preferences
    and feedback over time.
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize adaptive scorer.
        
        Args:
            initial_weights: Initial rule weights
        """
        self.weights = initial_weights or {
            'rule_of_thirds': 0.25,
            'leading_lines': 0.25,
            'symmetry': 0.20,
            'depth_layering': 0.15,
            'color_harmony': 0.15
        }
        
        self.feedback_history = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 10  # Minimum feedback samples for adaptation
        
        logger.info("AdaptiveScorer initialized")

    def update_from_feedback(self, rule_scores: Dict[str, float],
                             user_rating: float, user_preferences: Optional[Dict[str, float]] = None):
        """
        Update scoring weights based on user feedback.
        
        Args:
            rule_scores: Individual rule scores for the image
            user_rating: User's overall rating (0-1)
            user_preferences: Optional user preference weights
        """

        try:
            # Store feedback
            feedback_entry = {
                'rule_scores': rule_scores.copy(),
                'user_rating': user_rating,
                'predicted_rating': self.calculate_score(rule_scores),
                'preferences': user_preferences
            }

            self.feedback_history.append(feedback_entry)

            # Adapt weights if sufficient feedback

            if len(self.feedback_history) >= self.adaptation_threshold:
                self._adapt_weights()

            logger.debug(f"Feedback updated: user = {user_rating: .3f}, predicted = {feedback_entry['predicted_rating']:.3f}")

        except Exception as e:
            logger.warning(f"Feedback update failed: {str(e)}")

    def _adapt_weights(self):
        """Adapt scoring weights based on feedback history."""

        try:
            # Calculate prediction errors for each rule
            rule_correlations = {}

            for rule_name in self.weights.keys():
                rule_scores = [entry['rule_scores'].get(rule_name, 0.0) for entry in self.feedback_history]

                user_ratings = [entry['user_rating'] for entry in self.feedback_history]

                if len(rule_scores) > 1:
                    correlation = np.corrcoef(rule_scores, user_ratings)[0, 1]
                    rule_correlations[rule_name] = correlation if not np.isnan(correlation) else 0.0

                else:
                    rule_correlations[rule_name] = 0.0

            # Adjust weights based on correlations
            total_correlation = sum(abs(corr) for corr in rule_correlations.values())

            if total_correlation > 0:
                for rule_name, correlation in rule_correlations.items():
                    # Higher correlation = higher weight
                    new_weight = abs(correlation) / total_correlation

                    # Smooth update using learning rate
                    self.weights[rule_name] = (
                        (1 - self.learning_rate) * self.weights[rule_name] +
                        self.learning_rate * new_weight
                    )

            logger.info(f"Weights adapted: {self.weights}")

        except Exception as e:
            logger.warning(f"Weight adaptation failed: {str(e)}")
        
    def calculate_score(self, rule_scores: Dict[str, float]) -> float:
        """
        Calculate adaptive weighted score.
        
        Args:
            rule_scores: Individual rule scores
            
        Returns:
            Adaptive weighted overall score
        
        """

        try:
            total_weight = 0.0
            weighted_sum = 0.0

            for rule_name, score in rule_scores.items():
                weight = self.weights.get(rule_name, 0.0)
                weighted_sum += score * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"Adaptive score calculation failed: {str(e)}")
            return 0.0
        
    def get_personalized_suggestions(self, rule_scores: Dict[str, float]) -> List[str]:
        """
        Generate personalized suggestions based on learned preferences.
        
        Args:
            rule_scores: Current rule scores
            
        Returns:
            List of personalized improvement suggestions
        
        """

        suggestions = []

        try:
            # Identify rules with low scores and high user importance
            for rule_name, score in rule_scores.items():
                weight = self.weights.get(rule_name, 0.0)
                importance = weight / max(self.weights.values()) if self.weights.values() else 0.0
                
                if score < 0.5 and importance > 0.5:
                    suggestions.append(f"Focus on improving {rule_name.replace('_', ' ')} (high importance for you)")
                elif score < 0.3:
                    suggestions.append(f"Consider enhancing {rule_name.replace('_', ' ')}")
            
            return suggestions[:3]  # Limit to top 3 suggestions
            
        except Exception as e:
            logger.warning(f"Personalized suggestions generation failed: {str(e)}")
            return ["Unable to generate personalized suggestions."]
        
    def calculate_composition_distribution_score(self, rule_scores: Dict[str, float],
                                                 target_distribution: Optional[np.ndarray] = None) -> float:
        """
        Calculate composition score using distribution-based approach.
    
    Args:
        rule_scores: Individual rule scores
        target_distribution: Target score distribution (optional)
        
    Returns:
        Distribution-based composition score
        
        """
    
        try:
            scores_array = np.array(list(rule_scores.values()))

            if target_distribution is None:
                # Default target: balanced high scores
                target_distribution = np.ones_like(scores_array) * 0.7

            # Use Wasserstein distance as distribution metric
            distribution_score = 1.0 - wasserstein_distance(scores_array, target_distribution)

            return max(0.0, min(1.0, distribution_score))
        
        except Exception as e:
            logger.warning(f"Distribution score calculation failed: {str(e)}")
            return 0.0
        
    def ensemble_scoring(self, rule_scores: Dict[str, float], 
                         scoring_methods: List[str] = None) -> Dict[str, float]:
        """
        Ensemble multiple scoring methods for robust evaluation.
    
    Args:
        rule_scores: Individual rule scores
        scoring_methods: List of scoring methods to ensemble
        
    Returns:
        Dictionary of ensemble scores
        
        """
    
        if scoring_methods is None:
            scoring_methods = ['weighted_average', 'geometric_mean', 'harmonic_mean', 'distribution_based']
    
        ensemble_scores = {}
        scores_array = np.array(list(rule_scores.values()))
    
        try:
            # Weighted average (traditional)
            if 'weighted_average' in scoring_methods:
                weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])[:len(scores_array)]
                ensemble_scores['weighted_average'] = np.average(scores_array, weights=weights)
        
            # Geometric mean (emphasizes balance)
            if 'geometric_mean' in scoring_methods:
                # Add small epsilon to avoid zero values
                scores_positive = scores_array + 1e-8
                ensemble_scores['geometric_mean'] = np.exp(np.mean(np.log(scores_positive)))
        
            # Harmonic mean (emphasizes weak points)
            if 'harmonic_mean' in scoring_methods:
                scores_positive = scores_array + 1e-8
                ensemble_scores['harmonic_mean'] = len(scores_array) / np.sum(1.0 / scores_positive)
        
            # Distribution-based
            if 'distribution_based' in scoring_methods:
                ensemble_scores['distribution_based'] = self.calculate_composition_distribution_score(rule_scores)
        
            # Ensemble average
            if len(ensemble_scores) > 1:
                ensemble_scores['ensemble_average'] = np.mean(list(ensemble_scores.values()))
        
            return ensemble_scores
        
        except Exception as e:
            logger.warning(f"Ensemble scoring failed: {str(e)}")
            return {'ensemble_average': np.mean(scores_array) if len(scores_array) > 0 else 0.0}