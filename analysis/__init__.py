"""
Compositional Analysis Module

This module contains the Stage Three implementation of the AI Composition Assistant,
providing comprehensive compositional analysis, scoring, and suggestion generation.
"""

from .composition_analyzer import CompositionAnalyzer
from .rule_evaluators import (
    RuleOfThirdsEvaluator,
    LeadingLinesEvaluator,
    SymmetryEvaluator,
    DepthLayeringEvaluator,
    ColorHarmonyEvaluator
)
from .scoring_algorithms import (
    CompositionScorer,
    EMDLoss,
    MultiTaskScorer
)
from .aesthetic_quality import AestheticQualityAssessor
from .suggestion_engine import SuggestionEngine

__all__ = [
    'CompositionAnalyzer',
    'RuleOfThirdsEvaluator',
    'LeadingLinesEvaluator', 
    'SymmetryEvaluator',
    'DepthLayeringEvaluator',
    'ColorHarmonyEvaluator',
    'CompositionScorer',
    'EMDLoss',
    'MultiTaskScorer',
    'AestheticQualityAssessor',
    'SuggestionEngine'
]

__version__ = "1.0.0"
__author__ = "Willy Zuo"