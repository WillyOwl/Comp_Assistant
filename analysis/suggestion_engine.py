#!/usr/bin/env python3
"""
Suggestion Engine for Composition Improvement

This module generates intelligent suggestions for improving photographic composition
based on analysis results and compositional best practices.

"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)


class SuggestionPriority(Enum):
    """Priority levels for composition suggestions."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SuggestionCategory(Enum):
    """Categories of composition suggestions."""
    COMPOSITION = "composition"
    COLOR = "color"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    FRAMING = "framing"


@dataclass
class CompositionSuggestion:
    """
    Individual composition improvement suggestion.
    
    Contains detailed information about a specific improvement
    recommendation including rationale and implementation guidance.
    """
    id: str
    title: str
    description: str
    category: SuggestionCategory
    priority: SuggestionPriority
    current_score: float
    potential_improvement: float
    rationale: str
    implementation_tips: List[str]
    relevant_rules: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion to dictionary format."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'current_score': self.current_score,
            'potential_improvement': self.potential_improvement,
            'rationale': self.rationale,
            'implementation_tips': self.implementation_tips,
            'relevant_rules': self.relevant_rules
        }


class SuggestionEngine:
    """
    Main suggestion generation engine.
    
    Analyzes composition evaluation results and generates actionable
    improvement suggestions with prioritization and implementation guidance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize suggestion engine.
        
        Args:
            config: Configuration dictionary for suggestion parameters
        """
        self.config = config or self._get_default_config()
        
        # Load suggestion templates and knowledge base
        self.suggestion_templates = self._load_suggestion_templates()
        self.implementation_guides = self._load_implementation_guides()
        
        logger.info("SuggestionEngine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for suggestion engine."""
        return {
            'max_suggestions': 5,
            'prioritize_major_issues': True,
            'include_technical_suggestions': True,
            'include_creative_suggestions': True,
            'min_improvement_threshold': 0.1,
            'personalization_enabled': False
        }
    
    def generate_suggestions(self, 
                           rule_results: Dict[str, Dict[str, Any]],
                           rule_scores: Dict[str, float],
                           aesthetic_score: float,
                           technical_score: float) -> List[str]:
        """
        Generate improvement suggestions based on analysis results.
        
        Args:
            rule_results: Detailed rule evaluation results
            rule_scores: Individual rule scores
            aesthetic_score: Overall aesthetic quality score
            technical_score: Technical quality score
            
        Returns:
            List of improvement suggestion strings
        """
        try:
            # Generate detailed suggestions
            detailed_suggestions = self.generate_detailed_suggestions(
                rule_results, rule_scores, aesthetic_score, technical_score
            )
            
            # Convert to simple string format for backward compatibility
            suggestion_strings = []
            for suggestion in detailed_suggestions:
                suggestion_strings.append(suggestion.description)
            
            return suggestion_strings[:self.config['max_suggestions']]
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}")
            return ["Unable to generate suggestions due to analysis error."]
    
    def generate_detailed_suggestions(self,
                                    rule_results: Dict[str, Dict[str, Any]],
                                    rule_scores: Dict[str, float],
                                    aesthetic_score: float,
                                    technical_score: float) -> List[CompositionSuggestion]:
        """
        Generate detailed composition suggestions with full metadata.
        
        Args:
            rule_results: Detailed rule evaluation results
            rule_scores: Individual rule scores
            aesthetic_score: Overall aesthetic quality score
            technical_score: Technical quality score
            
        Returns:
            List of CompositionSuggestion objects
        """
        try:
            suggestions = []
            
            # Generate rule-specific suggestions
            rule_suggestions = self._generate_rule_suggestions(rule_results, rule_scores)
            suggestions.extend(rule_suggestions)
            
            # Generate technical suggestions
            if self.config.get('include_technical_suggestions', True):
                tech_suggestions = self._generate_technical_suggestions(technical_score, rule_results)
                suggestions.extend(tech_suggestions)
            
            # Generate aesthetic suggestions
            aesthetic_suggestions = self._generate_aesthetic_suggestions(aesthetic_score, rule_results)
            suggestions.extend(aesthetic_suggestions)
            
            # Generate creative suggestions
            if self.config.get('include_creative_suggestions', True):
                creative_suggestions = self._generate_creative_suggestions(rule_results, rule_scores)
                suggestions.extend(creative_suggestions)
            
            # Prioritize and filter suggestions
            filtered_suggestions = self._prioritize_and_filter_suggestions(suggestions)
            
            return filtered_suggestions[:self.config['max_suggestions']]
            
        except Exception as e:
            logger.error(f"Detailed suggestion generation failed: {str(e)}")
            return []
    
    def _generate_rule_suggestions(self, 
                                 rule_results: Dict[str, Dict[str, Any]],
                                 rule_scores: Dict[str, float]) -> List[CompositionSuggestion]:
        """Generate suggestions based on compositional rule analysis."""
        suggestions = []
        
        try:
            # Rule of Thirds suggestions
            if 'rule_of_thirds' in rule_results:
                rot_suggestions = self._generate_rule_of_thirds_suggestions(
                    rule_results['rule_of_thirds'], rule_scores.get('rule_of_thirds', 0.0)
                )
                suggestions.extend(rot_suggestions)
            
            # Leading Lines suggestions
            if 'leading_lines' in rule_results:
                lines_suggestions = self._generate_leading_lines_suggestions(
                    rule_results['leading_lines'], rule_scores.get('leading_lines', 0.0)
                )
                suggestions.extend(lines_suggestions)
            
            # Symmetry suggestions
            if 'symmetry' in rule_results:
                symmetry_suggestions = self._generate_symmetry_suggestions(
                    rule_results['symmetry'], rule_scores.get('symmetry', 0.0)
                )
                suggestions.extend(symmetry_suggestions)
            
            # Depth Layering suggestions
            if 'depth_layering' in rule_results:
                depth_suggestions = self._generate_depth_suggestions(
                    rule_results['depth_layering'], rule_scores.get('depth_layering', 0.0)
                )
                suggestions.extend(depth_suggestions)
            
            # Color Harmony suggestions
            if 'color_harmony' in rule_results:
                color_suggestions = self._generate_color_suggestions(
                    rule_results['color_harmony'], rule_scores.get('color_harmony', 0.0)
                )
                suggestions.extend(color_suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Rule suggestions generation failed: {str(e)}")
            return []
    
    def _generate_rule_of_thirds_suggestions(self, 
                                           result: Dict[str, Any], 
                                           score: float) -> List[CompositionSuggestion]:
        """Generate Rule of Thirds specific suggestions."""
        suggestions = []
        
        try:
            if score < 0.4:
                # Poor rule of thirds adherence
                suggestions.append(CompositionSuggestion(
                    id="rot_placement",
                    title="Improve Subject Placement",
                    description="Position key subjects along rule of thirds grid lines or at intersection points",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.HIGH,
                    current_score=score,
                    potential_improvement=0.3,
                    rationale="Strong visual elements placed at rule of thirds intersections create more dynamic and engaging compositions",
                    implementation_tips=[
                        "Imagine dividing your frame into 9 equal sections with 2 horizontal and 2 vertical lines",
                        "Place important subjects at the intersection points of these lines",
                        "Align horizons with the upper or lower horizontal line",
                        "Position vertical elements like trees or buildings along the vertical lines"
                    ],
                    relevant_rules=["rule_of_thirds"]
                ))
            
            elif score < 0.6:
                # Moderate improvement possible
                suggestions.append(CompositionSuggestion(
                    id="rot_refinement",
                    title="Fine-tune Rule of Thirds Alignment",
                    description="Adjust framing to better align key elements with grid intersections",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=score,
                    potential_improvement=0.2,
                    rationale="Small adjustments to element positioning can significantly improve visual balance",
                    implementation_tips=[
                        "Move slightly to reposition subjects closer to intersection points",
                        "Crop the image to improve alignment with grid lines",
                        "Consider the visual weight distribution across grid sections"
                    ],
                    relevant_rules=["rule_of_thirds"]
                ))
            
            # Check for specific issues
            intersection_scores = result.get('intersection_scores', [])
            if intersection_scores and max(intersection_scores) < 0.3:
                suggestions.append(CompositionSuggestion(
                    id="rot_intersections",
                    title="Utilize Intersection Points",
                    description="Place focal points at rule of thirds intersection points for stronger impact",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.HIGH,
                    current_score=max(intersection_scores) if intersection_scores else 0.0,
                    potential_improvement=0.4,
                    rationale="Intersection points naturally draw the viewer's eye and create visual tension",
                    implementation_tips=[
                        "Identify the most important element in your scene",
                        "Position it at one of the four intersection points",
                        "Use the intersection closest to the natural focal point"
                    ],
                    relevant_rules=["rule_of_thirds"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Rule of thirds suggestions generation failed: {str(e)}")
            return []
    
    def _generate_leading_lines_suggestions(self, 
                                          result: Dict[str, Any], 
                                          score: float) -> List[CompositionSuggestion]:
        """Generate Leading Lines specific suggestions."""
        suggestions = []
        
        try:
            lines = result.get('lines', [])
            vanishing_points = result.get('vanishing_points', [])
            
            if score < 0.3 or len(lines) < 2:
                suggestions.append(CompositionSuggestion(
                    id="lines_create",
                    title="Incorporate Leading Lines",
                    description="Add or emphasize lines that guide the viewer's eye through the composition",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.HIGH,
                    current_score=score,
                    potential_improvement=0.4,
                    rationale="Leading lines create visual flow and guide attention to important subjects",
                    implementation_tips=[
                        "Look for natural lines: roads, rivers, fences, shadows",
                        "Use architectural elements: stairs, railings, building edges",
                        "Create implied lines with arranged objects or directional lighting",
                        "Ensure lines lead toward your main subject"
                    ],
                    relevant_rules=["leading_lines"]
                ))
            
            elif not vanishing_points and len(lines) >= 2:
                suggestions.append(CompositionSuggestion(
                    id="lines_convergence",
                    title="Create Line Convergence",
                    description="Arrange lines to converge toward a vanishing point for stronger visual impact",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=score,
                    potential_improvement=0.2,
                    rationale="Converging lines create depth and draw attention to the convergence point",
                    implementation_tips=[
                        "Angle your camera to make parallel lines converge",
                        "Use perspective to create natural convergence",
                        "Position the vanishing point strategically in your frame"
                    ],
                    relevant_rules=["leading_lines"]
                ))
            
            line_analysis = result.get('line_analysis', {})
            line_types = line_analysis.get('line_types', {})
            
            if sum(line_types.values()) > 0 and line_types.get('diagonal', 0) == 0:
                suggestions.append(CompositionSuggestion(
                    id="lines_diagonal",
                    title="Add Diagonal Lines",
                    description="Incorporate diagonal lines for more dynamic composition",
                    category=SuggestionCategory.CREATIVE,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=score,
                    potential_improvement=0.15,
                    rationale="Diagonal lines add energy and movement to compositions",
                    implementation_tips=[
                        "Tilt your camera slightly for diagonal horizons",
                        "Look for naturally diagonal elements",
                        "Use shadows or lighting to create diagonal patterns"
                    ],
                    relevant_rules=["leading_lines"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Leading lines suggestions generation failed: {str(e)}")
            return []
    
    def _generate_symmetry_suggestions(self, 
                                     result: Dict[str, Any], 
                                     score: float) -> List[CompositionSuggestion]:
        """Generate Symmetry specific suggestions."""
        suggestions = []
        
        try:
            h_score = result.get('horizontal_score', 0.0)
            v_score = result.get('vertical_score', 0.0)
            r_score = result.get('radial_score', 0.0)
            
            best_score = max(h_score, v_score, r_score)
            
            if best_score < 0.3:
                suggestions.append(CompositionSuggestion(
                    id="symmetry_create",
                    title="Enhance Symmetrical Elements",
                    description="Look for or create symmetrical arrangements in your composition",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=best_score,
                    potential_improvement=0.3,
                    rationale="Symmetry creates visual balance and can be very pleasing to the eye",
                    implementation_tips=[
                        "Center your subject for vertical symmetry",
                        "Use reflections in water or glass for natural symmetry",
                        "Arrange objects symmetrically in still life photography",
                        "Look for architectural symmetry in buildings"
                    ],
                    relevant_rules=["symmetry"]
                ))
            
            # Specific symmetry type suggestions
            if v_score > h_score and v_score > r_score and v_score < 0.6:
                suggestions.append(CompositionSuggestion(
                    id="symmetry_vertical",
                    title="Improve Vertical Symmetry",
                    description="Center your composition better for stronger vertical symmetry",
                    category=SuggestionCategory.FRAMING,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=v_score,
                    potential_improvement=0.2,
                    rationale="Perfect vertical centering enhances symmetrical impact",
                    implementation_tips=[
                        "Use your camera's grid lines to center subjects",
                        "Check that equal space exists on left and right sides",
                        "Ensure vertical elements are truly vertical"
                    ],
                    relevant_rules=["symmetry"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Symmetry suggestions generation failed: {str(e)}")
            return []
    
    def _generate_depth_suggestions(self, 
                                  result: Dict[str, Any], 
                                  score: float) -> List[CompositionSuggestion]:
        """Generate Depth Layering specific suggestions."""
        suggestions = []
        
        try:
            layers = result.get('layers', [])
            focal_points = result.get('focal_points', [])
            depth_stats = result.get('depth_statistics', {})
            
            if len(layers) < 2:
                suggestions.append(CompositionSuggestion(
                    id="depth_layers",
                    title="Create Distinct Depth Layers",
                    description="Include foreground, middle ground, and background elements",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.HIGH,
                    current_score=score,
                    potential_improvement=0.3,
                    rationale="Multiple depth layers create a sense of three-dimensionality",
                    implementation_tips=[
                        "Include close objects in the foreground",
                        "Position your main subject in the middle ground",
                        "Ensure an interesting background",
                        "Use depth of field to separate layers"
                    ],
                    relevant_rules=["depth_layering"]
                ))
            
            if len(focal_points) < 2:
                suggestions.append(CompositionSuggestion(
                    id="depth_focal_points",
                    title="Add Multiple Focal Points",
                    description="Create visual interest at different depths in your scene",
                    category=SuggestionCategory.COMPOSITION,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=score,
                    potential_improvement=0.2,
                    rationale="Multiple focal points guide the eye through the three-dimensional space",
                    implementation_tips=[
                        "Place interesting objects at different distances",
                        "Use lighting to highlight elements at various depths",
                        "Create a visual path from foreground to background"
                    ],
                    relevant_rules=["depth_layering"]
                ))
            
            depth_range = depth_stats.get('depth_range', 0.0)
            if depth_range < 0.3:
                suggestions.append(CompositionSuggestion(
                    id="depth_range",
                    title="Increase Depth Variation",
                    description="Enhance the contrast between near and far elements",
                    category=SuggestionCategory.TECHNICAL,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=depth_range,
                    potential_improvement=0.25,
                    rationale="Greater depth variation makes the image more three-dimensional",
                    implementation_tips=[
                        "Use a wider aperture for shallow depth of field",
                        "Get closer to foreground objects",
                        "Include very distant background elements",
                        "Use overlapping elements to show depth relationships"
                    ],
                    relevant_rules=["depth_layering"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Depth suggestions generation failed: {str(e)}")
            return []
    
    def _generate_color_suggestions(self, 
                                  result: Dict[str, Any], 
                                  score: float) -> List[CompositionSuggestion]:
        """Generate Color Harmony specific suggestions."""
        suggestions = []
        
        try:
            harmony_scores = result.get('harmony_scores', {})
            temperature_analysis = result.get('temperature_analysis', {})
            dominant_colors = result.get('dominant_colors', [])
            
            if score < 0.4:
                suggestions.append(CompositionSuggestion(
                    id="color_harmony",
                    title="Improve Color Harmony",
                    description="Create more cohesive color relationships in your composition",
                    category=SuggestionCategory.COLOR,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=score,
                    potential_improvement=0.3,
                    rationale="Harmonious colors create visual unity and emotional coherence",
                    implementation_tips=[
                        "Use complementary colors for high contrast",
                        "Try analogous colors for peaceful harmony",
                        "Limit your color palette to 3-5 main colors",
                        "Consider the emotional impact of color choices"
                    ],
                    relevant_rules=["color_harmony"]
                ))
            
            # Specific harmony type suggestions
            comp_score = harmony_scores.get('complementary', 0.0)
            analog_score = harmony_scores.get('analogous', 0.0)
            
            if comp_score < 0.2 and analog_score < 0.2:
                suggestions.append(CompositionSuggestion(
                    id="color_scheme",
                    title="Establish a Color Scheme",
                    description="Choose either complementary or analogous colors for better harmony",
                    category=SuggestionCategory.COLOR,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=max(comp_score, analog_score),
                    potential_improvement=0.25,
                    rationale="A clear color scheme creates visual coherence",
                    implementation_tips=[
                        "Complementary: Use colors opposite on the color wheel",
                        "Analogous: Use colors next to each other on the color wheel",
                        "Monochromatic: Use different shades of the same color",
                        "Consider post-processing to enhance color relationships"
                    ],
                    relevant_rules=["color_harmony"]
                ))
            
            # Color balance suggestions
            sat_balance = harmony_scores.get('saturation_balance', 0.0)
            bright_balance = harmony_scores.get('brightness_balance', 0.0)
            
            if sat_balance < 0.4:
                suggestions.append(CompositionSuggestion(
                    id="color_saturation",
                    title="Balance Color Saturation",
                    description="Adjust the intensity of colors for better visual balance",
                    category=SuggestionCategory.COLOR,
                    priority=SuggestionPriority.LOW,
                    current_score=sat_balance,
                    potential_improvement=0.15,
                    rationale="Balanced saturation prevents any one color from overwhelming the composition",
                    implementation_tips=[
                        "Avoid having all colors at maximum saturation",
                        "Use one highly saturated accent color",
                        "Balance bright colors with muted tones",
                        "Consider the mood you want to convey"
                    ],
                    relevant_rules=["color_harmony"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Color suggestions generation failed: {str(e)}")
            return []
    
    def _generate_technical_suggestions(self, 
                                      technical_score: float,
                                      rule_results: Dict[str, Dict[str, Any]]) -> List[CompositionSuggestion]:
        """Generate technical quality suggestions."""
        suggestions = []
        
        try:
            if technical_score < 0.5:
                suggestions.append(CompositionSuggestion(
                    id="technical_overall",
                    title="Improve Technical Quality",
                    description="Enhance sharpness, contrast, and exposure for better image quality",
                    category=SuggestionCategory.TECHNICAL,
                    priority=SuggestionPriority.HIGH,
                    current_score=technical_score,
                    potential_improvement=0.3,
                    rationale="Good technical quality is the foundation of compelling photography",
                    implementation_tips=[
                        "Use a tripod for sharper images",
                        "Check your focus point carefully",
                        "Ensure proper exposure using histogram",
                        "Shoot in good lighting conditions"
                    ],
                    relevant_rules=["technical_quality"]
                ))
            
            # Check for specific technical issues
            if technical_score < 0.4:
                suggestions.append(CompositionSuggestion(
                    id="technical_exposure",
                    title="Optimize Exposure",
                    description="Adjust exposure settings for better highlight and shadow detail",
                    category=SuggestionCategory.TECHNICAL,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=technical_score,
                    potential_improvement=0.2,
                    rationale="Proper exposure preserves detail and enhances overall image quality",
                    implementation_tips=[
                        "Use exposure compensation to fine-tune brightness",
                        "Avoid blown highlights and blocked shadows",
                        "Consider HDR for high contrast scenes",
                        "Use graduated filters for uneven lighting"
                    ],
                    relevant_rules=["technical_quality"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Technical suggestions generation failed: {str(e)}")
            return []
    
    def _generate_aesthetic_suggestions(self, 
                                      aesthetic_score: float,
                                      rule_results: Dict[str, Dict[str, Any]]) -> List[CompositionSuggestion]:
        """Generate aesthetic quality suggestions."""
        suggestions = []
        
        try:
            if aesthetic_score < 0.5:
                suggestions.append(CompositionSuggestion(
                    id="aesthetic_overall",
                    title="Enhance Visual Appeal",
                    description="Improve the overall aesthetic quality and emotional impact",
                    category=SuggestionCategory.CREATIVE,
                    priority=SuggestionPriority.MEDIUM,
                    current_score=aesthetic_score,
                    potential_improvement=0.25,
                    rationale="Strong aesthetic appeal engages viewers emotionally",
                    implementation_tips=[
                        "Consider the emotional story you want to tell",
                        "Pay attention to mood and atmosphere",
                        "Look for unique perspectives and angles",
                        "Simplify your composition to focus attention"
                    ],
                    relevant_rules=["aesthetic_quality"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Aesthetic suggestions generation failed: {str(e)}")
            return []
    
    def _generate_creative_suggestions(self, 
                                     rule_results: Dict[str, Dict[str, Any]],
                                     rule_scores: Dict[str, float]) -> List[CompositionSuggestion]:
        """Generate creative and artistic suggestions."""
        suggestions = []
        
        try:
            # Creative suggestions based on what's already working well
            best_rule = max(rule_scores, key=rule_scores.get) if rule_scores else None
            best_score = rule_scores.get(best_rule, 0.0) if best_rule else 0.0
            
            if best_score > 0.7:
                if best_rule == 'rule_of_thirds':
                    suggestions.append(CompositionSuggestion(
                        id="creative_rot_enhance",
                        title="Strengthen Your Rule of Thirds",
                        description="Since you're already using rule of thirds well, consider enhancing it further",
                        category=SuggestionCategory.CREATIVE,
                        priority=SuggestionPriority.LOW,
                        current_score=best_score,
                        potential_improvement=0.1,
                        rationale="Building on your strengths can create exceptional compositions",
                        implementation_tips=[
                            "Add secondary subjects at other intersection points",
                            "Use the grid lines to organize multiple elements",
                            "Consider breaking the rule intentionally for artistic effect"
                        ],
                        relevant_rules=[best_rule]
                    ))
                
                elif best_rule == 'symmetry':
                    suggestions.append(CompositionSuggestion(
                        id="creative_symmetry_enhance",
                        title="Explore Symmetry Variations",
                        description="Experiment with different types of symmetry or intentional asymmetry",
                        category=SuggestionCategory.CREATIVE,
                        priority=SuggestionPriority.LOW,
                        current_score=best_score,
                        potential_improvement=0.1,
                        rationale="Mastering symmetry opens up creative possibilities",
                        implementation_tips=[
                            "Try radial symmetry for different effects",
                            "Introduce small asymmetrical elements for interest",
                            "Use symmetry in patterns and textures"
                        ],
                        relevant_rules=[best_rule]
                    ))
            
            # Random creative suggestion
            creative_ideas = [
                "Try shooting from an unusual angle or perspective",
                "Experiment with negative space to create minimalist compositions",
                "Use framing elements to create depth and focus",
                "Consider the golden hour for warm, dramatic lighting",
                "Look for patterns and textures that create visual rhythm"
            ]
            
            if len(suggestions) < 2:  # Add a random creative suggestion if we don't have many
                random_idea = random.choice(creative_ideas)
                suggestions.append(CompositionSuggestion(
                    id="creative_random",
                    title="Creative Enhancement",
                    description=random_idea,
                    category=SuggestionCategory.CREATIVE,
                    priority=SuggestionPriority.LOW,
                    current_score=0.5,
                    potential_improvement=0.2,
                    rationale="Creative experimentation leads to unique and compelling images",
                    implementation_tips=[
                        "Don't be afraid to try something different",
                        "Take multiple shots with different approaches",
                        "Study work by photographers you admire"
                    ],
                    relevant_rules=["creative"]
                ))
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Creative suggestions generation failed: {str(e)}")
            return []
    
    def _prioritize_and_filter_suggestions(self, 
                                         suggestions: List[CompositionSuggestion]) -> List[CompositionSuggestion]:
        """Prioritize and filter suggestions based on impact and configuration."""
        try:
            # Filter by minimum improvement threshold
            min_improvement = self.config.get('min_improvement_threshold', 0.1)
            filtered = [s for s in suggestions if s.potential_improvement >= min_improvement]
            
            # Sort by priority and potential improvement
            def sort_key(suggestion):
                priority_weight = {
                    SuggestionPriority.CRITICAL: 4,
                    SuggestionPriority.HIGH: 3,
                    SuggestionPriority.MEDIUM: 2,
                    SuggestionPriority.LOW: 1
                }
                return (priority_weight[suggestion.priority], suggestion.potential_improvement)
            
            filtered.sort(key=sort_key, reverse=True)
            
            # Apply prioritization settings
            if self.config.get('prioritize_major_issues', True):
                # Ensure high-priority suggestions are included first
                high_priority = [s for s in filtered if s.priority in [SuggestionPriority.CRITICAL, SuggestionPriority.HIGH]]
                other_priority = [s for s in filtered if s.priority in [SuggestionPriority.MEDIUM, SuggestionPriority.LOW]]
                
                # Combine with high priority first
                filtered = high_priority + other_priority
            
            return filtered
            
        except Exception as e:
            logger.warning(f"Suggestion prioritization failed: {str(e)}")
            return suggestions
    
    def _load_suggestion_templates(self) -> Dict[str, Any]:
        """Load suggestion templates for different scenarios."""
        # This would typically load from configuration files or database
        # For now, return empty dict as templates are embedded in the generation methods
        return {}
    
    def _load_implementation_guides(self) -> Dict[str, List[str]]:
        """Load implementation guides for different techniques."""
        # This would typically load from configuration files or database
        return {
            'rule_of_thirds': [
                "Use camera grid lines as guides",
                "Focus on intersection points for key subjects",
                "Align horizons with grid lines"
            ],
            'leading_lines': [
                "Look for natural lines in the environment",
                "Use architectural elements effectively",
                "Create diagonal compositions for energy"
            ],
            'symmetry': [
                "Center subjects carefully",
                "Use reflections when available",
                "Balance elements on both sides"
            ]
        }