"""
Evaluation metrics for composition analysis training.

This module implements comprehensive metrics for evaluating composition models
against professional photography standards and aesthetic quality measures.

"""

import torch
import time
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

class CompositionMetrics:
    """
    Comprehensive metrics for composition analysis evaluation.
    
    """

    def __init__(self, professional_standards: bool = True):
        """
        Initialize composition metrics.
        
        Args:
            professional_standards: Whether to include professional photography standards
        
        """
        
        self.professional_standards = professional_standards

        # Professional photography score thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }

    def compute_metrics(self, predictions: List[Dict[str, torch.Tensor]],
                        targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        
        """
         Compute comprehensive evaluation metrics.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of computed metrics
        
        """

        metrics = {}

        # Concatenate all predictions and targets
        all_preds = self._concatenate_batch_outputs(predictions)
        all_targets = self._concatenate_batch_outputs(targets)

        # Rule of thirds metrics
        if 'rule_of_thirds' in all_preds:
            rot_metrics = self._compute_rule_of_thirds_metrics(
                all_preds['rule_of_thirds'],
                all_targets['rule_of_thirds']
            )

            metrics.update({f'rot_{k}': v for k, v in rot_metrics.items()})

        # Leading lines metrics
        if 'leading_lines' in all_preds:
            lines_metrics = self._compute_leading_lines_metrics(
                all_preds['leading_lines'], 
                all_targets['leading_lines']
            )
            metrics.update({f'lines_{k}': v for k, v in lines_metrics.items()})
        
        # Symmetry metrics
        if 'symmetry' in all_preds:
            symmetry_metrics = self._compute_symmetry_metrics(
                all_preds['symmetry'], 
                all_targets['symmetry']
            )
            metrics.update({f'symmetry_{k}': v for k, v in symmetry_metrics.items()})
        
        # Depth metrics
        if 'depth' in all_preds:
            depth_metrics = self._compute_depth_metrics(
                all_preds['depth'], 
                all_targets['depth']
            )
            metrics.update({f'depth_{k}': v for k, v in depth_metrics.items()})
        
        # Overall quality metrics
        if 'overall_quality' in all_targets:
            overall_metrics = self._compute_overall_quality_metrics(
                all_preds, all_targets
            )
            metrics.update(overall_metrics)

        # Aesthetic correlation metrics
        aesthetic_metrics = self._compute_aesthetic_correlation_metrics(
           all_preds, all_targets
        )
        metrics.update(aesthetic_metrics)

        # Professional standards evaluation
        if self.professional_standards:
            professional_metrics = self._compute_professional_standards_metrics(
                all_preds, all_targets
            )
            metrics.update(professional_metrics)

        return metrics
    
    def _concatenate_batch_outputs(self, batch_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Concatenate outputs from multiple batches."""

        if not batch_list:
            return {}
        
        concatenated = {}
        for key in batch_list[0].keys():
            if isinstance(batch_list[0][key], torch.Tensor):
                concatenated[key] = torch.cat([batch[key] for batch in batch_list], dim = 0)

        return concatenated
    
    def _compute_rule_of_thirds_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute rule of thirds specific metrics.
        
        Args:
            predictions: Predicted grid point confidences [N, 9]
            targets: Target grid point confidences [N, 9]
            
        Returns:
            Dictionary of rule of thirds metrics
        
        """

        # Convert to numpy for easier computation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        # Binary classification metrics (above threshold)
        threshold = 0.5
        pred_binary = (pred_np > threshold).astype(int)
        target_binary = (target_np > threshold).astype(int)

        # Flatten for sklearn metrics
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()

        # Basic classification metrics
        accuracy = accuracy_score(target_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average = 'binary', zero_division = 0 
            )
        
        # Grid point detection accuracy (per image)
        grid_accuracy = np.mean([
            np.mean(pred_binary[i] == target_binary[i])
            for i in range(len(pred_binary))
        ])

        # Intersection over Union for grid points
        intersection = np.sum(pred_binary * target_binary, axis = 1)
        union = np.sum((pred_binary + target_binary) > 0, axis = 1)
        iou = np.mean(intersection / (union + 1e-8))

        # Continuous correlation
        correlation, _ = pearsonr(pred_np.flatten(), target_np.flatten())

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'grid_accuracy': grid_accuracy,
            'iou': iou,
            'correlation': correlation
        }
    
    def _compute_leading_lines_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute leading lines specific metrics.
        
        Args:
            predictions: Predicted line parameters [N, 5] (x1, y1, x2, y2, strength)
            targets: Target line parameters [N, 5]
            
        Returns:
            Dictionary of leading lines metrics
        
        """

        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        # Endpoint distance error
        pred_endpoints = pred_np[:, :4] # x1, y1, x2, y2
        target_endpoints = target_np[:, :4]

        endpoint_mse = np.mean((pred_endpoints - target_endpoints) ** 2)
        endpoint_mae = np.mean(np.abs(pred_endpoints - target_endpoints))

        # Strength correlation
        pred_strength = pred_np[:, 4]
        target_strength = target_np[:, 4]

        strength_correlation, _ = pearsonr(pred_strength, target_strength)
        strength_mse = np.mean((pred_strength - target_strength) ** 2)

        # Line angle accuracy
        pred_angles = self._compute_line_angles(pred_endpoints)
        target_angles = self._compute_line_angles(target_endpoints)

        # Handle angle wraparound
        angle_diff = np.abs(pred_angles - target_angles)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        mean_angle_error = np.mean(angle_diff)

        return {
            'endpoint_mse': endpoint_mse,
            'endpoint_mae': endpoint_mae,
            'strength_correlation': strength_correlation,
            'strength_mse': strength_mse,
            'angle_error': mean_angle_error
        }
    
    def _compute_line_angles(self, endpoints: np.ndarray) -> np.ndarray:
        """Compute angles from line endpoints."""

        x1, y1, x2, y2 = endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], endpoints[:, 3]

        return np.arctan2(y2 - y1, x2 - x1)
    
    def _compute_symmetry_metrics(self, 
                                predictions: torch.Tensor, 
                                targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute symmetry classification metrics.
        
        Args:
            predictions: Predicted symmetry type probabilities [N, 4]
            targets: Target symmetry type (one-hot) [N, 4]
            
        Returns:
            Dictionary of symmetry metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Convert to class predictions
        pred_classes = np.argmax(pred_np, axis=1)
        target_classes = np.argmax(target_np, axis=1)
        
        # Classification metrics
        accuracy = accuracy_score(target_classes, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_classes, pred_classes, average='macro', zero_division=0
        )
        
        # Per-class metrics
        class_names = ['none', 'horizontal', 'vertical', 'radial']
        per_class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            class_mask = (target_classes == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    target_classes[class_mask], 
                    pred_classes[class_mask]
                )
                per_class_metrics[f'{class_name}_accuracy'] = class_acc
        
        # Confidence calibration
        max_probs = np.max(pred_np, axis=1)
        correct_predictions = (pred_classes == target_classes)
        confidence_correlation, _ = pearsonr(max_probs, correct_predictions.astype(float))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confidence_correlation': confidence_correlation,
            **per_class_metrics
        }
    
    def _compute_depth_metrics(self, 
                             predictions: torch.Tensor, 
                             targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute depth estimation metrics.
        
        Args:
            predictions: Predicted depth scores [N, 1]
            targets: Target depth scores [N, 1]
            
        Returns:
            Dictionary of depth metrics
        """
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Regression metrics
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        rmse = np.sqrt(mse)
        
        # Correlation metrics
        pearson_corr, _ = pearsonr(pred_np, target_np)
        spearman_corr, _ = spearmanr(pred_np, target_np)
        
        # Relative error
        relative_error = np.mean(np.abs(pred_np - target_np) / (target_np + 1e-8))
        
        # Threshold accuracy (within 10% of target)
        threshold_acc = np.mean(np.abs(pred_np - target_np) < 0.1 * target_np)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'relative_error': relative_error,
            'threshold_accuracy': threshold_acc
        }
    
    def _compute_overall_quality_metrics(self, 
                                       predictions: Dict[str, torch.Tensor], 
                                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute overall composition quality metrics.
        
        Args:
            predictions: Dictionary of all predictions
            targets: Dictionary of all targets
            
        Returns:
            Dictionary of overall quality metrics
        """
        # Combine individual composition scores to overall score
        overall_pred = self._compute_overall_score(predictions)
        overall_target = targets['overall_quality'].detach().cpu().numpy().flatten()
        overall_pred = overall_pred.detach().cpu().numpy().flatten()
        
        # Regression metrics
        mse = np.mean((overall_pred - overall_target) ** 2)
        mae = np.mean(np.abs(overall_pred - overall_target))
        
        # Correlation with ground truth
        pearson_corr, _ = pearsonr(overall_pred, overall_target)
        spearman_corr, _ = spearmanr(overall_pred, overall_target)
        
        # Quality level classification accuracy
        pred_levels = self._score_to_quality_level(overall_pred)
        target_levels = self._score_to_quality_level(overall_target)
        level_accuracy = accuracy_score(target_levels, pred_levels)
        
        return {
            'overall_mse': mse,
            'overall_mae': mae,
            'overall_pearson': pearson_corr,
            'overall_spearman': spearman_corr,
            'overall_score': pearson_corr,  # Main metric for model selection
            'quality_level_accuracy': level_accuracy
        }
    
    def _compute_overall_score(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute overall composition score from individual predictions."""
        scores = []
        
        if 'rule_of_thirds' in predictions:
            rot_score = torch.mean(predictions['rule_of_thirds'], dim=1, keepdim=True)
            scores.append(rot_score)
        
        if 'leading_lines' in predictions:
            lines_score = predictions['leading_lines'][:, -1:] / 1.0  # Strength component
            scores.append(lines_score)
        
        if 'symmetry' in predictions:
            # Higher confidence in any symmetry type = higher score
            symmetry_score = torch.max(predictions['symmetry'], dim=1, keepdim=True)[0]
            scores.append(symmetry_score)
        
        if 'depth' in predictions:
            scores.append(predictions['depth'])
        
        if scores:
            overall_score = torch.mean(torch.cat(scores, dim=1), dim=1, keepdim=True)
        else:
            overall_score = torch.zeros(predictions[list(predictions.keys())[0]].size(0), 1)
        
        return overall_score
    
    def _score_to_quality_level(self, scores: np.ndarray) -> np.ndarray:
        """Convert continuous scores to quality levels."""
        levels = np.zeros_like(scores, dtype=int)
        
        levels[scores >= self.quality_thresholds['excellent']] = 3
        levels[(scores >= self.quality_thresholds['good']) & 
               (scores < self.quality_thresholds['excellent'])] = 2
        levels[(scores >= self.quality_thresholds['fair']) & 
               (scores < self.quality_thresholds['good'])] = 1
        # levels below 'fair' remain 0
        
        return levels
    
    def _compute_aesthetic_correlation_metrics(self, 
                                             predictions: Dict[str, torch.Tensor], 
                                             targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute correlation with aesthetic quality measures.
        
        Args:
            predictions: Dictionary of all predictions  
            targets: Dictionary of all targets
            
        Returns:
            Dictionary of aesthetic correlation metrics
        """
        # Get overall predicted and target scores
        overall_pred = self._compute_overall_score(predictions)
        overall_pred = overall_pred.detach().cpu().numpy().flatten()
        
        if 'aesthetic_score' in targets:
            aesthetic_target = targets['aesthetic_score'].detach().cpu().numpy().flatten()
        else:
            # Use overall quality as proxy for aesthetic score
            aesthetic_target = targets['overall_quality'].detach().cpu().numpy().flatten()
        
        # Aesthetic correlation
        aesthetic_corr, _ = pearsonr(overall_pred, aesthetic_target)
        aesthetic_spearman, _ = spearmanr(overall_pred, aesthetic_target)
        
        # Ranking correlation (important for aesthetic assessment)
        ranking_corr = self._compute_ranking_correlation(overall_pred, aesthetic_target)
        
        return {
            'aesthetic_pearson': aesthetic_corr,
            'aesthetic_spearman': aesthetic_spearman,
            'aesthetic_ranking': ranking_corr
        }
    
    def _compute_ranking_correlation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute ranking correlation using normalized discounted cumulative gain."""
        # Sort by predictions and check if order matches target order
        pred_order = np.argsort(pred)[::-1]  # Descending order
        target_sorted = target[pred_order]
        
        # Compute NDCG-like metric
        dcg = np.sum(target_sorted / np.log2(np.arange(2, len(target_sorted) + 2)))
        
        # Ideal DCG
        ideal_order = np.argsort(target)[::-1]
        ideal_sorted = target[ideal_order]
        ideal_dcg = np.sum(ideal_sorted / np.log2(np.arange(2, len(ideal_sorted) + 2)))
        
        return dcg / (ideal_dcg + 1e-8)
    
    def _compute_professional_standards_metrics(self, 
                                              predictions: Dict[str, torch.Tensor], 
                                              targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute metrics based on professional photography standards.
        
        Args:
            predictions: Dictionary of all predictions
            targets: Dictionary of all targets
            
        Returns:
            Dictionary of professional standards metrics
        """
        overall_pred = self._compute_overall_score(predictions)
        overall_pred = overall_pred.detach().cpu().numpy().flatten()
        overall_target = targets['overall_quality'].detach().cpu().numpy().flatten()
        
        # Professional grade classification (>= 0.7 is professional)
        professional_threshold = 0.7
        pred_professional = (overall_pred >= professional_threshold).astype(int)
        target_professional = (overall_target >= professional_threshold).astype(int)
        
        professional_accuracy = accuracy_score(target_professional, pred_professional)
        professional_precision, professional_recall, professional_f1, _ = \
            precision_recall_fscore_support(target_professional, pred_professional, 
                                           average='binary', zero_division=0)
        
        # Award-worthy classification (>= 0.9 is award-worthy)
        award_threshold = 0.9
        pred_award = (overall_pred >= award_threshold).astype(int)
        target_award = (overall_target >= award_threshold).astype(int)
        
        award_accuracy = accuracy_score(target_award, pred_award)
        
        # Portfolio ranking correlation
        # Simulate portfolio selection (top 20% of images)
        top_k = max(1, len(overall_pred) // 5)
        pred_top_k = np.argsort(overall_pred)[-top_k:]
        target_top_k = np.argsort(overall_target)[-top_k:]
        
        portfolio_overlap = len(set(pred_top_k) & set(target_top_k)) / top_k
        
        return {
            'professional_accuracy': professional_accuracy,
            'professional_precision': professional_precision,
            'professional_recall': professional_recall,
            'professional_f1': professional_f1,
            'award_accuracy': award_accuracy,
            'portfolio_overlap': portfolio_overlap
        }

class PerformanceBenchmark:
    """
    Performance benchmarking for composition models.
    """
    
    def __init__(self):
        """Initialize performance benchmark."""
        self.inference_times = []
        self.memory_usage = []
    
    def benchmark_inference(self, model: torch.nn.Module, 
                          input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            input_tensor: Input tensor for inference
            num_runs: Number of inference runs
            
        Returns:
            Dictionary of performance metrics
        """
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }