#!/usr/bin/env python3
"""
Rule Evaluators for Compositional Analysis

This module contains specialized evaluators for each compositional rule
including rule of thirds, leading lines, symmetry, depth layering, 
and color harmony.

"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy import ndimage
import colorsys

logger = logging.getLogger(__name__)

class BaseRuleEvaluator(ABC):
    """
    Abstract base class for all rule evaluators.

    Provides common interface and utility methods for compositional rule evaluation.

    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the rule evaluator with configuration"""

        self.config = config or {}

    @abstractmethod

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the compositional rule on the given image.

        Args:
            image: Input image(H, W, C) in BGR format
            features: optional pre-extracted features

        Returns:
            Dictionary containing evaluation results
        
        """

        pass

    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""

        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))
    
class RuleOfThirdsEvaluator(BaseRuleEvaluator):
    """
    Evaluates adherence to the Rule of Thirds compositional principle.

    Analyzes placement of strong visual elements along rule of thirds grid lines
    and intersections for optimal composition balance.
    
    """

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate rule of thirds compliance.

        Args:
            image: Input image
            features: Pre-extracted features (corners, edges, etc...)

        Returns:
            Dictionary with rule of thirds analysis results
        """

        try:
            h, w = image.shape[:2]

            # Define rule of thirds grid

            third_w, third_h = w // 3, h // 3
            v_lines = [third_w, 2 * third_w]
            h_lines = [third_h, 2 * third_h]
            intersections = [(x, y) for x in v_lines for y in h_lines]

            # Extract or detect strong points
            strong_points = self._get_strong_points(image, features)

            # Calculate alignment scores
            intersection_scores = self._calculate_intersection_alignment(strong_points, intersections)

            line_scores = self._calculate_line_alignment(strong_points, v_lines, h_lines, w, h)

            # Calculate overall rule of thirds score
            overall_score = self._calculate_overall_rot_score(intersection_scores, line_scores)

            # Analyze visual weight distribution
            weight_analysis = self._analyze_visual_weight_distribution(image, v_lines, h_lines)

            # Calculate confidence based on number of detected elements
            confidence = min(1.0, len(strong_points) / 10.0) # Normalize by expected points

            return {
                'score': overall_score,
                'intersection_scores': intersection_scores,
                'line_alignment_scores': line_scores,
                'strong_points': strong_points,
                'intersections': intersections,
                'grid_lines': {'vertical': v_lines, 'horizontal': h_lines},
                'weight_distribution': weight_analysis,
                'confidence': confidence,
                'elements_detected': len(strong_points),
                'analysis_success': True
            }
        
        except Exception as e:
            logger.error(f"Rule of thirds evaluation failed: {str(e)}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'elements_detected': 0,
                'analysis_success': False,
                'error': str(e)
            }
    
    def _get_strong_points(self, image: np.ndarray, features: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Extract or detect strong visual points in the image."""

        strong_points = []

        # Use pre-extracted features if available
        if features and 'corners' in features and features['corners'] is not None:
            corners = features['corners']
            if len(corners) > 0:
                strong_points.extend([(int(pt[0][0]), int(pt[0][1])) for pt in corners])

        # Detect additional strong points using corner detection
         
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners = 20, qualityLevel = 0.01, minDistance = 30
            )

            if corners is not None:
                additional_points = [(int(x), int(y)) for [[x, y]] in corners]
                strong_points.extend(additional_points)

        except Exception as e:
            logger.warning(f"Corner detection failed: {str(e)}")

        # Remove duplicates and return
        unique_points = list(set(strong_points))
        return unique_points[:15] # Limit to avoid noise
    
    def _calculate_intersection_alignment(self, strong_points: List[Tuple[int, int]],
                                          intersections: List[Tuple[int, int]]) -> List[float]:
        """Calculate alignment scores for rule of thirds intersections."""

        scores = []

        for intersection in intersections:
            distances = [np.sqrt((pt[0] - intersection[0]) ** 2 + (pt[1] - intersection[1]) ** 2)
                         for pt in strong_points]
            
            if distances:
                min_distance = min(distances)
                # Score based on proximity (closer = higher score)
                proximity_score = max(0.0, 1.0 - min_distance / 100.0)
                scores.append(proximity_score)

            else:
                scores.append(0.0)

        return scores
    
    def _calculate_line_alignment(self, strong_points: List[Tuple[int, int]],
                                  v_lines: List[int], h_lines: List[int],
                                  w: int, h: int) -> Dict[str, List[float]]:
        
        """Calculate alignment scores for rule of thirds lines."""
        v_scores = []
        h_scores = []

        # Vertical line alignment

        for v_line in v_lines:
            distances = [abs(pt[0] - v_line) for pt in strong_points]
            if distances:
                min_distance = min(distances)
                alignment_score = max(0.0, 1.0 - min_distance / 50.0)
                v_scores.append(alignment_score)

            else:
                v_scores.append(0.0)

        # Horizontal line alignment

        for h_line in h_lines:
            distances = [abs(pt[1] - h_line) for pt in strong_points]
            if distances:
                min_distance = min(distances)
                alignment_score = max(0.0, 1.0 - min_distance / 50.0)
                h_scores.append(alignment_score)
            
            else:
                h_scores.append(0.0)

        return {'vertical': v_scores, 'horizontal': h_scores}
    
    def _calculate_overall_rot_score(self, intersection_scores: List[float],
                                     line_scores: Dict[str, List[float]]) -> float:
        
        """Calculate overall rule of thirds score."""
        # Intersection alignment (40% weight)
        intersection_score = np.mean(intersection_scores) if intersection_scores else 0.0

        # Line alignment (60% weight)
        all_line_scores = line_scores['vertical'] + line_scores['horizontal']
        line_score = np.mean(all_line_scores) if all_line_scores else 0.0

        overall_score = intersection_score * 0.4 + line_score * 0.6

        return self._normalize_score(overall_score)
    
    def _analyze_visual_weight_distribution(self, image: np.ndarray,
                                            v_lines: List[int], h_lines: List[int]) -> Dict[str, float]:
        """Analyze visual weight distribution across rule of thirds grid."""

        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Create 9 regions based on rule of thirds grid
            regions = []
            y_dividers = [0] + h_lines + [h]
            x_dividers = [0] + v_lines + [w]

            for i in range(3):
                for j in range(3):
                    region = gray[y_dividers[i]: y_dividers[i+1],
                                  x_dividers[j]: x_dividers[j+1]]
                    
                    # Calculate visual weight as combination of brightness and variance

                    weight = region.mean() * 0.5 + region.std() * 0.5
                    regions.append(weight)

            # Calculate balance metrics
            total_weight = sum(regions)
            weights_normalized = [w / total_weight for w in regions] if total_weight > 0 else [0] * 9

            # Analyze balance
            center_weight = weights_normalized[4] # Center region
            corner_weights = [weights_normalized[i] for i in [0, 2, 6, 8]]
            edge_weights = [weights_normalized[i] for i in [1, 3, 5, 7]]

            balance_score = 1.0 - abs(0.5 - center_weight) # Prefer off-center weighting

            return {
                    'region_weights': weights_normalized,
                    'center_weights': center_weight,
                    'corner_weight_avg': np.mean(corner_weights),
                    'edge_weight_avg': np.mean(edge_weights),
                    'balance_score': balance_score
            }
        
        except Exception as e:
            logger.warning(f"Visual weight analysis failed: {str(e)}")
            return {'balance_score': 0.5}
        
class LeadingLinesEvaluator(BaseRuleEvaluator):
    """
    Evaluate leading lines composition principle.

    Analyzes lines that guide the viewer's eye through the composition
    toward important subjects or vanishing points.
    
    """

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate leading lines in the composition.

        Args:
            image: Input image
            features: Pre-extracted features (lines, edges, etc...)

        Returns:
            Dictionary with leading lines analysis results
        
        """

        try:
            # Extract or detect lines
            lines = self._get_lines(image, features)

            # Analyze line properties
            line_analysis = self.analyze_line_properties(lines, image.shape[:2])

            # Find vanishing points
            vanishing_points = self._find_vanishing_points(lines, image.shape[:2])

            # Score line effectiveness
            line_scores = self._score_line_effectiveness(lines, vanishing_points, image.shape[:2])

            # Calculate overall score
            overall_score = self._calculate_overall_lines_score(line_analysis, line_scores, vanishing_points)

            # Calculate confidence
            confidence = min(1.0, len(lines) / 5.0) # Normalize by expected lines

            return {
                'score': overall_score,
                'lines': lines,
                'line_scores': line_scores,
                'vanishing_points': vanishing_points,
                'line_analysis': line_analysis,
                'confidence': confidence,
                'elements_detected': len(lines),
                'analysis_success': True
            }
        
        except Exception as e:
            logger.error(f"Leading lines evaluation failed: {str(e)}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'elements_detected': 0,
                'analysis_success': False,
                'error': str(e)
            }
        
    def _get_lines(self, image: np.ndarray, features: Optional[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
        """Extract or detect lines in the image."""

        lines = []

        # Use pre-extracted features if available
        if features and 'lines' in features and features['lines'] is not None:
            feature_lines = features['lines']
            if len(feature_lines) > 0:
                for line in feature_lines:
                    if len(line[0]) > 4:
                        x1, y1, x2, y2 = line[0][:4]
                        lines.append((int(x1), int(x2), int(y1), int(y2)))

        # Detect additional lines using HoughLinesP

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

            detected_lines = cv2.HoughLinesP(
                edges, rho = 1, theta = np.pi / 180, threshold = 50,
                minLineLength = 30, maxLineGap = 10
            )

            if detected_lines is not None:
                for line in detected_lines:
                    x1, y1, x2, y2 = line[0]
                    lines.append((int(x1), int(x2), int(y1), int(y2)))
        
        except Exception as e:
            logger.warning(f"Line detection failed: {str(e)}")

        # Filter and return significant lines
        return self._filter_significant_lines(lines, image.shape[:2])
    
    def _filter_significant_lines(self, lines: List[Tuple[int, int, int, int]],
                                  image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        
        """Filter out short or insignificant lines."""

        h, w  = image_shape
        min_length = min(w, h) * 0.1 # Minimum 10% of image dimension

        significant_lines = []

        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if length >= min_length:
                significant_lines.append(line)
        
        return significant_lines[:10] # Limit to avoid noise
    
    def analyze_line_properties(self, lines: List[Tuple[int, int, int, int]],
                                 image_shape: Tuple[int, int]) -> Dict[str, Any]:
        
        """Analyze properties of detected lines"""

        if not lines:
            return {'avg_length': 0, 'angle_distribution': [], 'line_types': {}}
        
        h, w = image_shape
        lengths = []
        angles = []

        for line in lines:
            x1, y1, x2, y2 = line
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            lengths.append(length)
            angles.append(angle)

        # Classify line types
        line_types = {
            'horizontal': sum(1 for a in angles if abs(a) < 15 or abs(a) > 165),
            'vertical': sum(1 for a in angles if 75 < abs(a) < 105),
            'diagonal': sum(1 for a in angles if 15 <= abs(a) <= 75 or 105 <= abs(a) <= 165)
        }

        return {
            'avg_length': np.mean(lengths),
            'length_std': np.std(lengths),
            'angle_distribution': angles,
            'line_types': line_types,
            'total_lines': len(lines)
        }
    
    def _find_vanishing_points(self, lines: List[Tuple[int, int, int, int]],
                               image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        
        """Find vanishing points where lines converge."""

        if len(lines) < 2:
            return []
        
        h, w = image_shape
        vanishing_points = []

        # Find intersections of line pairs
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                intersection = self._line_intersection(lines[i], lines[j])
                
                if intersection:
                    x, y = intersection
                    # Consider points within extended image bounds

                    if -w <= x <= 2 * w and -h <= y <= 2 * h:
                        vanishing_points.append((x, y))

        # Cluster nearby vanishing points

        if len(vanishing_points) > 1:
            clustered_points = self._cluster_vanishing_points(vanishing_points)

            return clustered_points[:3]
        
        return vanishing_points
    
    def _line_intersection(self, line1: Tuple[int, int, int, int],
                           line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        
        """Calculate intersection points of two lines"""

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) * (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6: # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)
    
    def _cluster_vanishing_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Cluster nearby vanishing points."""

        if len(points) <= 1:
            return points
        
        points_array = np.array(points)

        # Use K-means clustering with adaptive number of clusters

        max_clusters = min(3, len(points))

        best_clusters = []

        for n_clusters in range(1, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10)
                labels = kmeans.fit_predict(points_array)
                centers = kmeans.cluster_centers_

                # Calculate cluster quality (silhouette-like score)

                if n_clusters > 1:
                    distances = pdist(points_array)
                    avg_distance = np.mean(distances)
                    cluster_quality = avg_distance / n_clusters

                else:
                    cluster_quality = float('inf')

                best_clusters = [(center[0], center[1]) for center in centers]

            except Exception:
                continue
        
        return best_clusters
    
    def _score_line_effectiveness(self, lines: List[Tuple[int, int, int, int]],
                                  vanishing_points: List[Tuple[float, float]],
                                  image_shape: Tuple[int, int]) -> List[float]:
        
        """Score the effectiveness of each line for composition"""

        scores = []
        h, w = image_shape

        for line in lines:
            x1, y1, x2, y2 = line

            # Length score (Longer lines are generally better)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            length_score = min(1.0, length / (min(w, h) * 0.5 ))

            # Position score (Lines starting from edges are better)
            edge_distance = min(x1, y1, w - x1, h - y1, x2, y2, w - x2, h - y2)
            position_score = max(0.0, 1.0 - edge_distance / 50.0)

            # Vanishing point convergence score
            convergence_score = 0.0
            if vanishing_points:
                for vp in vanishing_points:
                    vp_x, vp_y = vp

                    # Check if line points toward vanishing point
                    line_vec = np.array([x2 - x1, y2 - y1])
                    to_vp_vec = np.array([vp_x - x1, vp_y - y1])

                    if np.linalg.norm(line_vec) > 0 and np.linalg.norm(to_vp_vec) > 0:
                        cos_angle = np.dot(line_vec, to_vp_vec) / (np.linalg.norm(line_vec) * np.linalg.norm(to_vp_vec))

                        convergence_score = max(convergence_score, abs(cos_angle))

            # Combine scores
            overall_line_score = length_score * 0.4 + position_score * 0.3 + convergence_score * 0.3

            scores.append(overall_line_score)
        
        return scores
    
    def _calculate_overall_lines_score(self, line_analysis: Dict[str, Any],
                                       line_scores: List[float],
                                       vanishing_points: List[Tuple[float, float]]) -> float:
        
        """Calculate overall leading lines score."""

        if not line_scores:
            return 0.0
        
        # Average line effectiveness
        avg_line_score = np.mean(line_scores)

        # Bonus for vanishing points
        vp_bonus = min(0.3, len(vanishing_points) * 0.1)

        # Bonus for line diversity
        line_types = line_analysis.get('line_types', {})
        diversity_bonus = 0.0
        if sum(line_types.values()) > 0:
            diversity = len([t for t in line_types.values() if t > 0])
            diversity_bonus = min(0.2, diversity * 0.1)

        overall_score = avg_line_score + vp_bonus + diversity_bonus

        return self._normalize_score(overall_score)
    
class SymmetryEvaluator(BaseRuleEvaluator):
    """
    Evaluates symmetry in composition.

    Analyzes horizontal, vertical, and radial symmetry to assess
    compositional balance and visual harmony.
    
    """

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate symmetry in the composition.

        Args:
            image: Input image
            features: Optional pre-extracted features

        Returns:
            Dictionary with symmetry analysis results
        
        """

        try:
            # Analyze different types of symmetry
            horizontal_score = self._analyze_horizontal_symmetry(image)
            vertical_score = self._analyze_vertical_symmetry(image)
            radial_score = self._analyze_radial_symmetry(image)

            # Determine dominant symmetry type
            scores = {
                'horizontal': horizontal_score,
                'vertical': vertical_score,
                'radial': radial_score
            }

            if scores:
                dominant_type = max(scores, key=scores.get)
                dominant_score = scores[dominant_type]
            else:
                dominant_type = None
                dominant_score = 0.0

            # Calculate overall symmetry score
            overall_score = max(horizontal_score, vertical_score, radial_score)

            # Calculate confidence based on strength of symmetry
            confidence = dominant_score

            return {
                'score': overall_score,
                'horizontal_score': horizontal_score,
                'vertical_score': vertical_score,
                'radial_score': radial_score,
                'dominant_type': dominant_type if dominant_score > 0.3 else None,
                'confidence': confidence,
                'elements_detected': 1 if overall_score > 0.2 else 0,
                'analysis_success': True
            }
        
        except Exception as e:
            logger.error(f"Symmetry evaluation failed: {str(e)}")
            return {
                'score': 0.0,
                'horizontal_score': 0.0,
                'vertical_score': 0.0,
                'radial_score': 0.0,
                'dominant_type': None,
                'confidence':  0.0,
                'elements_detected': 0,
                'analysis_success': False,
                'error': str(e)
            }
        
    def _analyze_horizontal_symmetry(self, image: np.ndarray) -> float:
        """Analyze horizontal symmetry (reflection across horizontal axis)."""

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Split image into top and bottom halves
            mid_h = h // 2
            top_half = gray[:mid_h, :]
            bottom_half = gray[mid_h: mid_h + top_half.shape[0], :]

            # Flip bottom half vertically for comparison
            bottom_flipped = np.flipud(bottom_half)

            # Calculate similarity using normalized cross-correlation
            similarity = cv2.matchTemplate(top_half, bottom_flipped, cv2.TM_CCOEFF_NORMED)[0, 0]

            return self._normalize_score(similarity, -1.0, 1.0)
        
        except Exception as e:
            logger.warning(f"Horizontal symmetry analysis failed: {str(e)}")
            return 0.0
        
    def _analyze_vertical_symmetry(self, image: np.ndarray) -> float:
        """Analyze vertical symmetry (reflection across vertical axis)."""

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Split image into left and right halves
            mid_w = w // 2
            left_half = gray[:, :mid_w]
            right_half = gray[:, mid_w: mid_w + left_half.shape[1]]

            # Flip right half horizontally for comparison
            right_flipped = np.fliplr(right_half)

            # Calculate similarity using normalized cross-correlation
            similarity = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)[0, 0]

            return self._normalize_score(similarity, -1.0, 1.0)
        
        except Exception as e:
            logger.warning(f"Vertical symmetry analysis failed: {str(e)}")
            return 0.0
        
    def _analyze_radial_symmetry(self, image: np.ndarray) -> float:
        """Analyze radial symmetry (rotation around center point)."""

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            center = (w // 2, h // 2)

            # Create polar transformation
            max_radius = min(w, h) // 2
            polar_img = cv2.linearPolar(gray, center, max_radius, cv2.WARP_FILL_OUTLIERS)

            # Analyze symmetry in different rotational angles
            symmetry_scores = []
            for angle in [90, 180, 270]:
                rows, cols = polar_img.shape
                rotation_matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
                rotated = cv2.warpAffine(polar_img, rotation_matrix, (cols, rows))

                # Calculate similarity
                similarity = cv2.matchTemplate(polar_img, rotated, cv2.TM_CCOEFF_NORMED)[0, 0]
                symmetry_scores.append(similarity)

            # Return best symmetry score
            best_score = max(symmetry_scores) if symmetry_scores else 0.0
            
            return self._normalize_score(best_score, -1.0, 1.0)
        
        except Exception as e:
            logger.warning(f"Radial symmetry analysis failed: {str(e)}")
            return 0.0
        
class DepthLayeringEvaluator(BaseRuleEvaluator):
    """
    Evaluate depth layering in the composition.

    Analyzes foreground, middle ground, and background separation
    to assess three-dimensional depth perception
    
    """

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate depth layering in the composition.

        Args:
            image: Input image
            features: Optional pre-extracted features (depth map, etc...)

        Returns:
            Dictionary with depth layering analysis results
        
        """

        try:
            # Estimate depth map
            depth_map = self._estimate_depth_map(image, features)

            # Analyze depth layers
            layers = self._analyze_depth_layers(depth_map)

            # Find focal points at different depths
            focal_points = self._find_depth_focal_points(image, depth_map)

            # Calculate depth statistics
            depth_stats = self._calculate_depth_statistics(depth_map)

            # Score depth effectiveness
            depth_score = self._score_depth_layering(layers, focal_points, depth_stats)

            # Calculate confidence
            confidence = min(1.0, len(layers) / 3.0) # Normalize by expected layers

            return {
                'score': depth_score,
                'layers': layers,
                'focal_points': focal_points,
                'depth_map': depth_map,
                'depth_statistics': depth_stats,
                'confidence': confidence,
                'elements_detected': len(layers),
                'analysis_success': True
            }
        
        except Exception as e:
            logger.warning(f"Depth layering evaluation failed: {str(e)}")
            return {
                'score': 0.0,
                'layers': [],
                'focal_points': [],
                'depth_map': np.zeros((100, 100)),
                'depth_statistics': {'std_depth': 1.0},
                'confidence': 0.0,
                'elements_detected': 0,
                'analysis_success': False,
                'error': str(e)
            }
        
    def _estimate_depth_map(self, image: np.ndarray, features: Optional[Dict[str, Any]]) -> np.ndarray:
        
        """Estimate depth map using various techniques"""

        # Use pre-extracted depth map if available
        if features and 'depth_map' in features:
            return features['depth_map']
        
        # Simple depth estimation using blur and contrast
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge-based depth estimation
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.blur(edges.astype(np.float32), (15, 15))

            # Contrast-based depth estimation
            contrast = cv2.Laplacian(gray, cv2.CV_64F)
            contrast_map = cv2.blur(np.abs(contrast), (15, 15))

            # Combine metrics (higher values = closer / foreground)
            depth_map = edge_density * 0.6 + contrast_map * 0.4

            # Normalize to [0, 1] range
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            return depth_map
        
        except Exception as e:
            logger.warning(f"Depth estimation failed: {str(e)}")
            # Return uniform depth map as fallback
            return np.ones(image.shape[:2], dtype = np.float32) * 0.5
        
    def _analyze_depth_layers(self, depth_map: np.ndarray) -> List[Dict[str, Any]]:
        
        """Analyze distince depth layers in the image."""

        try:
            # Quantize depth into layers
            n_layers = 5
            quantized = np.floor(depth_map * (n_layers - 1)).astype(np.uint8)

            layers = []
            for layer_idx in range(n_layers):
                mask = (quantized == layer_idx)
                if np.sum(mask) > depth_map.size * 0.05: # At least 5% of image
                    # Calculate layer properties
                    layer_area = np.sum(mask) / depth_map.size
                    avg_depth = layer_idx / (n_layers - 1)

                    # Find connected components
                    labeled, num_components = ndimage.label(mask)

                    layer_info = {
                        'depth_level': avg_depth,
                        'area_ratio': layer_area,
                        'num_regions': num_components,
                        'mask': mask
                    }
                    layers.append(layer_info)
            
            return sorted(layers, key = lambda x: x['depth_level'])
        
        except Exception as e:
            logger.warning(f"Depth layer analysis failed: {str(e)}")
            return []
        
    def _find_depth_focal_points(self, image: np.ndarray, depth_map: np.ndarray) -> List[Dict[str, Any]]:

        """Find focal points at different depth levels."""

        try:
            focal_points = []

            # Use corner detection weighted by depth
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners = 20,
                                              qualityLevel = 0.01, minDistance = 30)
            
            if corners is not None:
                for corner in corners:
                    x, y = int(corner[0][0]), int(corner[0][1])
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth_value = depth_map[y, x]

                        focal_point = {
                            'position': (x, y),
                            'depth': depth_value,
                            'strength': float(cv2.cornerHarris(gray, 2, 3, 0.04)[y, x])
                        }
                        focal_points.append(focal_point)

            # Sort by depth (foreground first)
            return sorted(focal_points, key = lambda x: x['depth'], reverse = True)
        
        except Exception as e:
            logger.warning(f"Focal point detection failed: {str(e)}")
            return []
        
    def _calculate_depth_statistics(self, depth_map: np.ndarray) -> Dict[str, float]:
        
        """Calculate statistical measures of depth distribution"""

        try:
            return {
                'mean_depth': float(np.mean(depth_map)),
                'std_depth': float(np.std(depth_map)),
                'depth_range': float(np.max(depth_map) - np.min(depth_map)),
                'depth_entropy': float(-np.sum(np.histogram(depth_map, bins = 10,
                                                            density = True)[0] * np.log(np.histogram(depth_map, bins = 10, density = True)[0] + 1e-8)))
            }
        
        except Exception as e:
            logger.warning(f"Depth statistics calculation failed: {str(e)}")
            return {
                'mean_depth': 0.5, 'std_depth': 0.1, 'depth_range': 0.2, 'depth_entropy': 1.0
            }
        
    def _score_depth_layering(self, layers: List[Dict[str, Any]],
                              focal_points: List[Dict[str, Any]],
                              depth_stats: Dict[str, float]) -> float:
        
        """Score the effectiveness of depth layering"""

        score_components = []

        # Layer count score (3-4 layers is  optimal)
        num_layers = len(layers)
        if num_layers >= 3:
            layer_score = min(1.0, num_layers / 4.0)

        else:
            layer_score = num_layers / 3.0
        score_components.append(layer_score * 0.3)

        # Depth variation score
        depth_variation = depth_stats.get('std_depth', 0.0)
        variation_score = min(1.0, depth_variation * 3.0) # Scale appropriately
        score_components.append(variation_score * 0.3)

        # Focal point distribution score
        if len(focal_points) >= 2:
            depth_values = [fp['depth'] for fp in focal_points]
            focal_variation = np.std(depth_values)
            focal_score = min(1.0, focal_variation * 2.0)

        else:
            focal_score = 0.0
        score_components.append(focal_score * 0.2)

        # Depth range score
        depth_range = depth_stats.get('depth_range', 0.0)
        range_score = min(1.0, depth_range * 2.0)
        score_components.append(range_score * 0.2)

        return sum(score_components)
    
class ColorHarmonyEvaluator(BaseRuleEvaluator):
    """
    Evaluates color harmony in composition

    Analyzes color relationships, palette coherence, and emotional impact
    based on color theory principles
    """

    def evaluate(self, image: np.ndarray, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate color harmony in the composition.

        Args:
            image: Input image
            features: Optional pre-extracted features

        Returns:
            Dictionary with color harmony analysis results
        """

        try:
            # Extract dominant colors
            dominant_colors = self._extract_dominant_colors(image)

            # Analyze color relationships
            color_relationships = self._analyze_color_relationships(dominant_colors)

            # Calculate color harmony scores
            harmony_scores = self._calculate_harmony_scores(dominant_colors, color_relationships)

            # Analyze color temperature and mood
            temperature_analysis = self._analyze_color_temperature(image, dominant_colors)

            # Calculate overall color score
            overall_score = self._calculate_overall_color_score(harmony_scores, temperature_analysis)

            # Calculate confidence
            confidence = min(1.0, len(dominant_colors) / 5.0)

            return {
                'score': overall_score,
                'dominant_colors': dominant_colors,
                'color_relationships': color_relationships,
                'harmony_scores': harmony_scores,
                'temperature_analysis': temperature_analysis,
                'confidence': confidence,
                'elements_detected': len(dominant_colors),
                'analysis_success': True
            }
        
        except Exception as e:
            logger.error(f"Color harmony evaluation failed: {str(e)}")
            return {
                'score': 0.0,
                'dominant_colors': [],
                'color_relationships': {},
                'temperature_analysis': {},
                'confidence': 0.0,
                'elements_detected': 0,
                'analysis_success': False,
                'error': str(e)
            }
        
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:

        """Extract dominant colors using K-means clustering."""

        try:
            # Reshape image for clustering
            pixels = image.reshape(-1, 3).astype(np.float32)

            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                            cv2.KMEANS_RANDOM_CENTERS)
            
            # Calculate color statistics
            dominant_colors = []
            for i, center in enumerate(centers):
                mask = (labels.flatten() == i)
                percentage = np.sum(mask) / len(labels.flatten())

                # Convert BGR to RGB and HSV
                bgr_color = center.astype(np.uint8)
                rgb_color = bgr_color[::-1]
                hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255, rgb_color[1] / 255,
                                                rgb_color[2] / 255)
                
                color_info = {
                    'bgr': tuple(bgr_color),
                    'rgb': tuple(rgb_color),
                    'hsv': hsv_color,
                    'percentage': float(percentage),
                    'cluster_id': i
                }
                dominant_colors.append(color_info)
            
            # Sort by percentage (most dominant first)

            return sorted(dominant_colors, key = lambda x: x['percentage'], reverse = True)
        
        except Exception as e:
            logger.warning(f"Color extraction failed: {str(e)}")
            return []
        
    def _analyze_color_relationships(self, dominant_colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships between dominant colors."""
        if len(dominant_colors) < 2:
            return {}
        
        relationships = {
            'complementary_pairs': [],
            'analogous_groups': [],
            'triadic_groups': [],
            'monochromatic_level': 0.0
        }
        
        try:
            # Check for complementary colors (opposite on color wheel)
            for i in range(len(dominant_colors)):
                for j in range(i + 1, len(dominant_colors)):
                    hue1 = dominant_colors[i]['hsv'][0] * 360
                    hue2 = dominant_colors[j]['hsv'][0] * 360
                    
                    hue_diff = abs(hue1 - hue2)
                    hue_diff = min(hue_diff, 360 - hue_diff)  # Handle circular nature
                    
                    if 150 <= hue_diff <= 210:  # Complementary range
                        relationships['complementary_pairs'].append((i, j))
            
            # Check for analogous colors (adjacent on color wheel)
            analogous_groups = []
            for i in range(len(dominant_colors)):
                group = [i]
                hue_base = dominant_colors[i]['hsv'][0] * 360
                
                for j in range(len(dominant_colors)):
                    if i != j:
                        hue_other = dominant_colors[j]['hsv'][0] * 360
                        hue_diff = abs(hue_base - hue_other)
                        hue_diff = min(hue_diff, 360 - hue_diff)
                        
                        if hue_diff <= 30:  # Analogous range
                            group.append(j)
                
                if len(group) >= 3:
                    analogous_groups.append(group)
            
            relationships['analogous_groups'] = analogous_groups
            
            # Calculate monochromatic level (similar hues)
            if len(dominant_colors) >= 2:
                hues = [color['hsv'][0] * 360 for color in dominant_colors]
                hue_std = np.std(hues)
                relationships['monochromatic_level'] = max(0.0, 1.0 - hue_std / 60.0)
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Color relationship analysis failed: {str(e)}")
            return relationships
    
    def _calculate_harmony_scores(self, dominant_colors: List[Dict[str, Any]], 
                                color_relationships: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various color harmony scores."""
        scores = {
            'complementary': 0.0,
            'analogous': 0.0,
            'monochromatic': 0.0,
            'triadic': 0.0,
            'saturation_balance': 0.0,
            'brightness_balance': 0.0
        }
        
        if not dominant_colors:
            return scores
        
        try:
            # Complementary harmony score
            comp_pairs = color_relationships.get('complementary_pairs', [])
            if comp_pairs:
                # Weight by color percentages
                comp_score = 0.0
                for i, j in comp_pairs:
                    weight = dominant_colors[i]['percentage'] + dominant_colors[j]['percentage']
                    comp_score += weight
                scores['complementary'] = min(1.0, comp_score)
            
            # Analogous harmony score
            analog_groups = color_relationships.get('analogous_groups', [])
            if analog_groups:
                # Score based on largest analogous group
                max_group_size = max(len(group) for group in analog_groups)
                scores['analogous'] = min(1.0, max_group_size / len(dominant_colors))
            
            # Monochromatic score
            scores['monochromatic'] = color_relationships.get('monochromatic_level', 0.0)
            
            # Saturation balance
            saturations = [color['hsv'][1] for color in dominant_colors]
            sat_std = np.std(saturations)
            scores['saturation_balance'] = max(0.0, 1.0 - sat_std * 2.0)
            
            # Brightness balance
            brightnesses = [color['hsv'][2] for color in dominant_colors]
            bright_std = np.std(brightnesses)
            scores['brightness_balance'] = max(0.0, 1.0 - bright_std * 2.0)
            
            return scores
            
        except Exception as e:
            logger.warning(f"Harmony score calculation failed: {str(e)}")
            return scores
        
    def _analyze_color_temperature(self, image: np.ndarray, 
                                 dominant_colors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze color temperature and mood characteristics."""
        if not dominant_colors:
            return {}
        
        try:
            # Calculate average color temperature
            warm_colors = 0
            cool_colors = 0
            
            for color in dominant_colors:
                hue = color['hsv'][0] * 360
                percentage = color['percentage']
                
                # Warm colors: red, orange, yellow (0-60, 300-360)
                if (0 <= hue <= 60) or (300 <= hue <= 360):
                    warm_colors += percentage
                # Cool colors: blue, green, purple (120-300)
                elif 120 <= hue <= 300:
                    cool_colors += percentage
            
            temperature_balance = warm_colors - cool_colors  # -1 (cool) to +1 (warm)
            
            # Calculate overall saturation
            avg_saturation = np.mean([color['hsv'][1] for color in dominant_colors])
            
            # Calculate overall brightness
            avg_brightness = np.mean([color['hsv'][2] for color in dominant_colors])
            
            # Determine mood
            mood = self._determine_color_mood(temperature_balance, avg_saturation, avg_brightness)
            
            return {
                'temperature_balance': temperature_balance,
                'warm_percentage': warm_colors,
                'cool_percentage': cool_colors,
                'average_saturation': avg_saturation,
                'average_brightness': avg_brightness,
                'mood': mood
            }
            
        except Exception as e:
            logger.warning(f"Color temperature analysis failed: {str(e)}")
            return {}
    
    def _determine_color_mood(self, temperature: float, saturation: float, brightness: float) -> str:
        """Determine the mood conveyed by the color palette."""
        if brightness > 0.7 and saturation > 0.6:
            if temperature > 0.2:
                return "energetic"
            else:
                return "vibrant"
        elif brightness < 0.3:
            if saturation > 0.5:
                return "dramatic"
            else:
                return "somber"
        elif saturation < 0.3:
            if brightness > 0.5:
                return "serene"
            else:
                return "muted"
        elif temperature > 0.3:
            return "warm"
        elif temperature < -0.3:
            return "cool"
        else:
            return "balanced"
        
    def _calculate_overall_color_score(self, harmony_scores: Dict[str, float], 
                                     temperature_analysis: Dict[str, Any]) -> float:
        """Calculate overall color harmony score."""
        if not harmony_scores:
            return 0.0
        
        try:
            # Weight different harmony types
            weighted_score = (
                harmony_scores.get('complementary', 0.0) * 0.25 +
                harmony_scores.get('analogous', 0.0) * 0.25 +
                harmony_scores.get('monochromatic', 0.0) * 0.15 +
                harmony_scores.get('saturation_balance', 0.0) * 0.2 +
                harmony_scores.get('brightness_balance', 0.0) * 0.15
            )
            
            # Bonus for balanced temperature
            temp_balance = abs(temperature_analysis.get('temperature_balance', 0.0))
            if temp_balance < 0.3:  # Well balanced
                weighted_score += 0.1
            
            return self._normalize_score(weighted_score)
            
        except Exception as e:
            logger.warning(f"Overall color score calculation failed: {str(e)}")
            return 0.0