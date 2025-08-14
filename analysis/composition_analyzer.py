#!/usr/bin/env python3
"""
Main Composition Analyzer

This module provides the core CompositionAnalyzer class that orchestrates all
compositional analysis components for Stage Three of the AI Composition Assistant.

"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime

from .rule_evaluators import (
    RuleOfThirdsEvaluator,
    LeadingLinesEvaluator,
    SymmetryEvaluator,
    DepthLayeringEvaluator,
    ColorHarmonyEvaluator
)

from .scoring_algorithms import CompositionScorer, MultiTaskScorer
from .aesthetic_quality import AestheticQualityAssessor
from .suggestion_engine import SuggestionEngine

logger = logging.getLogger(__name__)

@dataclass
class CompositionResults:
    """

    Comprehensive composition analysis results.

    Contains scores, evaluations, and suggestions for all compositional elements.
    
    """

    overall_score: float
    rule_scores: Dict[str, float]
    aesthetic_score: float
    technical_score: float
    detailed_analysis: Dict[str, Any]
    suggestions: List[str]
    confidence: float
    processing_time: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """ Convert results to dictionary format. """

        return {
            'overall_score': self.overall_score,
            'rule_scores': self.rule_scores,
            'aesthetic_score': self.aesthetic_score,
            'technical_score': self.technical_score,
            'detailed_analysis': self.detailed_analysis,
            'suggestions': self.suggestions,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }

class CompositionAnalyzer:
    """

    Main compositional analysis orchestrator for Stage Three.

    This class coordinates all compositional analysis components including
    rule evaluation, scoring, aesthetic assessment, and suggestion generation.
    
    """

    def __init__(self, device: Optional[torch.device] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the composition analyzer.

        Args:
            device: PyTorch device for computation
            config: Configuration dictionary for analyzer settings
        """

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config or self._get_default_config()

        # Initialize rule evaluators

        self.rule_evaluators = {
            'rule_of_thirds': RuleOfThirdsEvaluator(),
            'leading_lines': LeadingLinesEvaluator(),
            'symmetry': SymmetryEvaluator(),
            'depth_layering': DepthLayeringEvaluator(),
            'color_harmony': ColorHarmonyEvaluator()
        }

        # Initialize scoring system
        self.scorer = CompositionScorer(self.config.get('scoring', {}))
        self.multi_task_scorer = MultiTaskScorer(self.device)

        # Initialize aesthetic quality assessor
        self.aesthetic_assessor = AestheticQualityAssessor(self.device)

        # Initialize suggestion engine
        self.suggestion_engine = SuggestionEngine(self.config.get('suggestions', {}))

        logger.info(f"CompositionAnalyzer initialized on {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the analyzer"""

        return {
            'rule_weights': {
                'rule_of_thirds': 0.25,
                'leading_lines': 0.25,
                'symmetry': 0.20,
                'depth_layering': 0.15,
                'color_harmony': 0.15
            },

            'scoring': {
                'use_emd_loss': True,
                'confidence_threshold': 0.7,
                'quality_threshold': 0.5
            },

            'suggestions': {
                'max_suggestions': 5,
                'prioritize_major_issues': True,
                'include_technical_suggestions': True
            },

            'analysis_depth': 'comprehensive' # 'basic', 'standard', 'comprehensive'
        }
    
    def analyze(self, image: np.ndarray,
                features: Optional[Dict[str, Any]] = None,
                return_visualizations: bool = False) -> CompositionResults:
        
        """
        Perform comprehensive compositional analysis on an image.

        Args:
            image: Input image (H, W, C) in BGR format
            features: Pre-extracted features from Stage Two (Optional)
            return_visualizations: Whether to include visualization data

        Returns:
            CompositionResults containing complete analysis
        """

        start_time = datetime.now()

        try:
            # Step 1: Run individual rule evaluations
            logger.debug("Running rule evaluations...")
            rule_results = self._evaluate_rules(image, features)

            # Step 2: Calculate rule scores
            logger.debug("Calculating rule scores...")
            rule_scores = self._calculate_rule_scores(rule_results)

            # Step 3: Assess aesthetic quality
            logger.debug("Assessing aesthetic quality...")
            aesthetic_score = self._assess_aesthetic_quality(image, rule_results)

            # Step 4: Calculate technical quality
            logger.debug("Calculating technical quality...")
            technical_score = self._calculate_technical_quality(image, features)

            # Step 5: Calculate overall composition score
            logger.debug("Calculating overall score...")
            overall_score = self._calculate_overall_score(
                rule_scores, aesthetic_score, technical_score
            )

            # Step 6: Generate suggestions
            logger.debug("Generating suggestions...")
            suggestions = self._generate_suggestions(
                rule_results, rule_scores, aesthetic_score, technical_score
            )

            # Step 7: Calculate confidence
            confidence = self._calculate_confidence(rule_results, rule_scores)

            # Step 8: Prepare detailed analysis
            detailed_analysis = self._prepare_detailed_analysis(
                rule_results, return_visualizations
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            results = CompositionResults(
                overall_score = overall_score,
                rule_scores = rule_scores,
                aesthetic_score = aesthetic_score,
                technical_score = technical_score,
                detailed_analysis = detailed_analysis,
                suggestions = suggestions,
                confidence = confidence,
                processing_time = processing_time,
                timestamp = start_time
            )

            logger.info(f"Analysis completed in {processing_time:.3f}s - Score: {overall_score:.3f}")

            return results
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _evaluate_rules(self, image: np.ndarray,
                        features: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Run all rule evaluators on the rule."""

        rule_results = {}

        for rule_name, evaluator in self.rule_evaluators.items():
            try:
                if rule_name == 'rule_of_thirds':
                    result = evaluator.evaluate(image, features)

                elif rule_name == 'leading_lines':
                    result = evaluator.evaluate(image, features)

                elif rule_name == 'symmetry':
                    result = evaluator.evaluate(image)

                elif rule_name == 'depth_layering':
                    result = evaluator.evaluate(image, features)
                
                elif rule_name == 'color_harmony':
                    result = evaluator.evaluate(image)

                else:
                    result = evaluator.evaluate(image, features)

                rule_results[rule_name] = result
                logger.debug(f"{rule_name} evaluation completed")

            except Exception as e:
                logger.warning(f"Rule evaluation failed for {rule_name}: {str(e)}")
                rule_results[rule_name] = self.get_empty_rule_result()

        return rule_results
    
    def _calculate_rule_scores(self, rule_results: Dict[str, Dict[str, Any]]) ->Dict[str, float]:
        """Calculate normalized scores for each compositional rule."""

        rule_scores = {}

        for rule_name, result in rule_results.items():
            try:
                score = self.scorer.calculate_rule_score(rule_name, result)
                rule_scores[rule_name] = max(0.0, min(1.0, score)) # Clamp to [0, 1]

            except Exception as e:
                logger.warning(f"Score calculation failed for {rule_name}: {str(e)}")

                rule_scores[rule_name] = 0.0

        return rule_scores
    
    def _assess_aesthetic_quality(self, image: np.ndarray, rule_results: Dict[str, Any]) -> float:
        """Assess overall aesthetic quality of the image."""

        try:
            return self.aesthetic_assessor.assess(image, rule_results)
        except Exception as e:
            logger.warning(f"Aesthetic assessment failed: {str(e)}")
            return 0.5 # Neutral score as fallback
        
    def _calculate_technical_quality(self, image: np.ndarray, feature: Optional[Dict[str, Any]]) -> float:
        """Calculate technical quality metrics"""

        try:
            # Basic technical quality metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 1000.0) # Normalize

            # Contrast (RMS contrast)
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 64.0) # Normalize

            # Brightness balance
            brightness = gray.mean()
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5

            # Combine technical scores
            technical_score = sharpness_score * 0.4 + contrast_score * 0.4 + brightness_score * 0.2

            return max(0.0, min(1.0, technical_score))
        
        except Exception as e:
            logger.warning(f"Technical quality calculation failed: {str(e)}")
            return 0.5
        
    def _calculate_overall_score(self, rule_scores: Dict[str, float],
                                 aesthetic_score: float, technical_score: float) -> float:
        """Calculate weighted overall composition score."""

        try:
            # Weighted rule scores
            rule_weight_sum = 0.0
            weighted_rule_score = 0.0

            for rule_name, score in rule_scores.items():
                weight = self.config['rule_weights'].get(rule_name, 0.0)
                weighted_rule_score += score * weight
                rule_weight_sum += weight

            # Normalize rule score if weights don't sum to 1
            if rule_weight_sum > 0:
                weighted_rule_score /= rule_weight_sum

            # Combine with aesthetic and technical scores
            overall_score = (
                weighted_rule_score * 0.6 + # Compositional rules
                aesthetic_score * 0.3 +     # Aesthetic quality
                technical_score * 0.1       # Technical quality
            )

            return max(0.0, min(1.0, overall_score))
        
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {str(e)}")
            return 0.5
        
    def _generate_suggestions(self,
                              rule_results: Dict[str, Dict[str, Any]],
                              rule_scores: Dict[str, float],
                              aesthetic_score: float,
                              technical_score: float) -> List[str]:
        """Generate improvement suggestions based on analysis results"""

        try:
            return self.suggestion_engine.generate_suggestions(
                rule_results, rule_scores, aesthetic_score, technical_score
            )
        
        except Exception as e:
            logger.warning(f"Suggestion generation failed: {str(e)}")
            return ["Unable to generate suggestions due to analysis error."]
        
    def _calculate_confidence(self, 
                            rule_results: Dict[str, Dict[str, Any]],
                            rule_scores: Dict[str, float]) -> float:
        """Calculate confidence in the analysis results."""
        try:
            confidences = []
            
            # Rule-specific confidence calculations
            for rule_name, result in rule_results.items():
                if 'confidence' in result:
                    confidences.append(result['confidence'])
                else:
                    # Estimate confidence based on detection quality
                    if rule_name == 'rule_of_thirds' and 'points' in result:
                        confidence = min(1.0, len(result['points']) / 4.0)
                    elif rule_name == 'leading_lines' and 'lines' in result:
                        confidence = min(1.0, len(result['lines']) / 3.0)
                    else:
                        confidence = 0.5  # Neutral confidence
                    confidences.append(confidence)
            
            # Overall confidence as average
            overall_confidence = np.mean(confidences) if confidences else 0.5
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
        
    def _prepare_detailed_analysis(self, rule_results: Dict[str, Dict[str, Any]],
                                   include_visualizations: bool) -> Dict[str, Any]:
        """Prepare detailed analysis data"""

        detailed = {
            'rule_analysis': {},
            'metadata': {
                'analyzer_version': '1.0.0',
                'analysis_depth': self.config['analysis_depth'],
                'device': str(self.device)
            }
        }

        for rule_name, result in rule_results.items():
            rule_detail = result.copy()

            # Remove Large visualization data if not requested

            if not include_visualizations:
                for key in ['visualization', 'debug_image', 'overlay']:
                    rule_detail.pop(key, None)

            detailed['rule_analysis'][rule_name] = rule_detail
        
        return detailed
    
    def _get_empty_rule_result(self) -> Dict[str, Any]:
        """Get empty result structure for failed rule evaluation."""

        return {
            'score': 0.0,
            'confidence': 0.0,
            'elements_detected': 0,
            'analysis_success': False
        }
    
    def batch_analyze(self, images: List[np.ndarray],
                      features_list: Optional[List[Dict[str, Any]]] = None) -> List[CompositionResults]:
        """

        Analyze mutiple images in batch for efficiency.

        Args:
            images: List of input images
            features_list: List of pre-extracted features (optional)

        Returns:
            List of CompositionResults for each image
        """

        results = []
        features_list = features_list or [None] * len(images)

        logger.info(f"Starting batch analysis of {len(images)} images")

        for i, (image, features) in enumerate(zip(images, features_list)):
            try:
                result = self.analyze(image, features)
                results.append(result)
                logger.debug(f"Batch analysis {i+1} / {len(images)} completed")

            except Exception as e:
                logger.error(f"Batch analysis failed for image {i+1}: {str(e)}")
                # Create error result

                error_result = CompositionResults(
                    overall_score=0.0,
                    rule_scores={},
                    aesthetic_score=0.0,
                    technical_score=0.0,
                    detailed_analysis={'error': str(e)},
                    suggestions=["Analysis failed due to processing error."],
                    confidence=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )

                results.append(error_result)

        logger.info(f"Batch analysis completed: {len(results)} results")
        return results