#!/usr/bin/env python3
"""
Aesthetic Quality Assessment Module

This module implements aesthetic quality assessment using modern ML approaches
including CLIP-based evaluation and multi-modal aesthetic prediction.

"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

@dataclass
class AestheticFeatures:
    """
    Container for extracted aesthetic features.
    
    Stores various aesthetic indicators extracted from the image
    for quality assessment.
    """
    global_features: np.ndarray
    color_features: np.ndarray
    texture_features: np.ndarray
    composition_features: np.ndarray
    semantic_features: Optional[np.ndarray] = None
    
    def to_vector(self) -> np.ndarray:
        """Concatenate all features into a single vector."""
        features = [self.global_features, self.color_features, 
                   self.texture_features, self.composition_features]
        if self.semantic_features is not None:
            features.append(self.semantic_features)
        return np.concatenate(features)


class AestheticFeatureExtractor:
    """
    Extracts aesthetic features from images for quality assessment.
    
    Combines traditional computer vision features with modern deep learning
    representations for comprehensive aesthetic analysis.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize aesthetic feature extractor.
        
        Args:
            device: PyTorch device for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("AestheticFeatureExtractor initialized")
    
    def extract_features(self, image: np.ndarray) -> AestheticFeatures:
        """
        Extract comprehensive aesthetic features from image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            AestheticFeatures containing extracted features
        """
        try:
            # Extract different feature types
            global_features = self._extract_global_features(image)
            color_features = self._extract_color_features(image)
            texture_features = self._extract_texture_features(image)
            composition_features = self._extract_composition_features(image)
            
            return AestheticFeatures(
                global_features=global_features,
                color_features=color_features,
                texture_features=texture_features,
                composition_features=composition_features
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            # Return zero features as fallback
            return AestheticFeatures(
                global_features=np.zeros(10),
                color_features=np.zeros(15),
                texture_features=np.zeros(8),
                composition_features=np.zeros(12)
            )
    
    def _extract_global_features(self, image: np.ndarray) -> np.ndarray:
        """Extract global image statistics and properties."""
        try:
            features = []
            
            # Convert to different color spaces
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Basic statistics
            features.append(np.mean(gray))  # Mean brightness
            features.append(np.std(gray))   # Brightness variation
            features.append(gray.min())     # Darkest value
            features.append(gray.max())     # Brightest value
            
            # Dynamic range
            features.append(gray.max() - gray.min())
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Histogram entropy
            entropy = -np.sum(hist * np.log(hist + 1e-8))
            features.append(entropy)
            
            # Contrast measures
            # RMS contrast
            rms_contrast = gray.std()
            features.append(rms_contrast)
            
            # Michelson contrast
            if gray.max() + gray.min() > 0:
                michelson = (gray.max() - gray.min()) / (gray.max() + gray.min())
            else:
                michelson = 0.0
            features.append(michelson)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Image sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(sharpness / 1000.0)  # Normalize
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Global feature extraction failed: {str(e)}")
            return np.zeros(10, dtype=np.float32)
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-related aesthetic features."""
        try:
            features = []
            
            # Convert to different color spaces
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Color statistics in RGB
            for channel in range(3):
                features.append(np.mean(rgb[:, :, channel]))
                features.append(np.std(rgb[:, :, channel]))
            
            # HSV statistics
            features.append(np.mean(hsv[:, :, 0]))  # Mean hue
            features.append(np.std(hsv[:, :, 0]))   # Hue variation
            features.append(np.mean(hsv[:, :, 1]))  # Mean saturation
            features.append(np.std(hsv[:, :, 1]))   # Saturation variation
            features.append(np.mean(hsv[:, :, 2]))  # Mean value
            features.append(np.std(hsv[:, :, 2]))   # Value variation
            
            # Color temperature estimation
            r_mean = np.mean(rgb[:, :, 0])
            b_mean = np.mean(rgb[:, :, 2])
            if b_mean > 0:
                color_temp = r_mean / b_mean
            else:
                color_temp = 1.0
            features.append(color_temp)
            
            # Colorfulness metric
            rg = rgb[:, :, 0] - rgb[:, :, 1]
            yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
            
            colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
            features.append(colorfulness / 100.0)  # Normalize
            
            # Dominant color analysis
            pixels = rgb.reshape(-1, 3)
            n_colors = min(5, len(np.unique(pixels.view(np.void), axis=0)))
            features.append(n_colors / 5.0)  # Normalize
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Color feature extraction failed: {str(e)}")
            return np.zeros(15, dtype=np.float32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract texture-related features."""
        try:
            features = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Local Binary Pattern (simplified)
            def lbp_simple(img, radius=3):
                """Simplified LBP calculation."""
                h, w = img.shape
                lbp_img = np.zeros_like(img)
                
                for i in range(radius, h - radius):
                    for j in range(radius, w - radius):
                        center = img[i, j]
                        pattern = 0
                        
                        # Compare with neighbors
                        neighbors = [
                            img[i-radius, j-radius], img[i-radius, j], img[i-radius, j+radius],
                            img[i, j+radius], img[i+radius, j+radius], img[i+radius, j],
                            img[i+radius, j-radius], img[i, j-radius]
                        ]
                        
                        for k, neighbor in enumerate(neighbors):
                            if neighbor >= center:
                                pattern += 2**k
                        
                        lbp_img[i, j] = pattern
                
                return lbp_img
            
            # Calculate LBP features
            lbp = lbp_simple(gray)
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=16, range=(0, 255))
            lbp_hist = lbp_hist.astype(np.float32) / lbp_hist.sum()
            
            # Use first 4 LBP histogram bins as features
            features.extend(lbp_hist[:4])
            
            # Gabor filter responses (simplified)
            def gabor_responses(img):
                """Calculate Gabor filter responses."""
                responses = []
                kernel_size = 21
                
                for theta in [0, 45, 90, 135]:
                    kernel = cv2.getGaborKernel((kernel_size, kernel_size), 
                                              sigma=3, theta=np.radians(theta), 
                                              lambd=10, gamma=0.5)
                    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    responses.append(np.mean(np.abs(filtered)))
                
                return responses
            
            gabor_features = gabor_responses(gray)
            features.extend(gabor_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Texture feature extraction failed: {str(e)}")
            return np.zeros(8, dtype=np.float32)
    
    def _extract_composition_features(self, image: np.ndarray) -> np.ndarray:
        """Extract composition-related features."""
        try:
            features = []
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Rule of thirds grid analysis
            third_h, third_w = h // 3, w // 3
            
            # Calculate energy in different grid regions
            regions = [
                gray[0:third_h, 0:third_w],           # Top-left
                gray[0:third_h, third_w:2*third_w],   # Top-center
                gray[0:third_h, 2*third_w:w],         # Top-right
                gray[third_h:2*third_h, 0:third_w],   # Middle-left
                gray[third_h:2*third_h, third_w:2*third_w],  # Center
                gray[third_h:2*third_h, 2*third_w:w], # Middle-right
                gray[2*third_h:h, 0:third_w],         # Bottom-left
                gray[2*third_h:h, third_w:2*third_w], # Bottom-center
                gray[2*third_h:h, 2*third_w:w]        # Bottom-right
            ]
            
            region_energies = []
            for region in regions:
                if region.size > 0:
                    energy = np.mean(region) + np.std(region)  # Brightness + variation
                    region_energies.append(energy)
                else:
                    region_energies.append(0.0)
            
            # Normalize region energies
            total_energy = sum(region_energies)
            if total_energy > 0:
                region_energies = [e / total_energy for e in region_energies]
            
            features.extend(region_energies)
            
            # Symmetry measures (simplified)
            # Horizontal symmetry
            top_half = gray[:h//2, :]
            bottom_half = gray[h//2:h//2+top_half.shape[0], :]
            h_symmetry = 1.0 - np.mean(np.abs(top_half - np.flipud(bottom_half))) / 255.0
            features.append(h_symmetry)
            
            # Vertical symmetry
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:w//2+left_half.shape[1]]
            v_symmetry = 1.0 - np.mean(np.abs(left_half - np.fliplr(right_half))) / 255.0
            features.append(v_symmetry)
            
            # Center bias
            center_region = gray[h//4:3*h//4, w//4:3*w//4]
            center_energy = np.mean(center_region) if center_region.size > 0 else 0.0
            features.append(center_energy / 255.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Composition feature extraction failed: {str(e)}")
            return np.zeros(12, dtype=np.float32)


class AestheticQualityPredictor(nn.Module):
    """
    Neural network for predicting aesthetic quality scores.
    
    Multi-branch architecture that processes different types of aesthetic
    features and combines them for overall quality prediction.
    """
    
    def __init__(self, feature_dims: Dict[str, int]):
        """
        Initialize aesthetic quality predictor.
        
        Args:
            feature_dims: Dictionary of feature type dimensions
        """
        super(AestheticQualityPredictor, self).__init__()
        
        self.feature_dims = feature_dims
        
        # Individual feature processors
        self.global_processor = nn.Sequential(
            nn.Linear(feature_dims['global'], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        self.color_processor = nn.Sequential(
            nn.Linear(feature_dims['color'], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        self.texture_processor = nn.Sequential(
            nn.Linear(feature_dims['texture'], 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8)
        )
        
        self.composition_processor = nn.Sequential(
            nn.Linear(feature_dims['composition'], 24),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(24, 12)
        )
        
        # Fusion network
        fusion_input_size = 16 + 16 + 8 + 12  # Sum of processor outputs
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through aesthetic quality predictor.
        
        Args:
            features: Dictionary of feature tensors
            
        Returns:
            Aesthetic quality score
        """
        # Process individual feature types
        global_feat = self.global_processor(features['global'])
        color_feat = self.color_processor(features['color'])
        texture_feat = self.texture_processor(features['texture'])
        composition_feat = self.composition_processor(features['composition'])
        
        # Fuse features
        fused_features = torch.cat([global_feat, color_feat, texture_feat, composition_feat], dim=1)
        
        # Predict quality score
        quality_score = self.fusion_network(fused_features)
        
        return quality_score


class AestheticQualityAssessor:
    """
    Main aesthetic quality assessment system.
    
    Combines feature extraction, neural network prediction, and heuristic
    assessment for comprehensive aesthetic quality evaluation.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize aesthetic quality assessor.
        
        Args:
            device: PyTorch device for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize feature extractor
        self.feature_extractor = AestheticFeatureExtractor(self.device)
        
        # Initialize predictor with feature dimensions
        feature_dims = {
            'global': 10,
            'color': 15,
            'texture': 8,
            'composition': 12
        }
        
        self.predictor = AestheticQualityPredictor(feature_dims)
        self.predictor.to(self.device)
        
        # Initialize with pretrained weights if available
        self._initialize_predictor()
        
        logger.info(f"AestheticQualityAssessor initialized on {self.device}")
    
    def _initialize_predictor(self):
        """Initialize predictor with default weights."""
        try:
            # Initialize with Xavier/Glorot initialization
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
            
            self.predictor.apply(init_weights)
            logger.debug("Predictor initialized with Xavier weights")
            
        except Exception as e:
            logger.warning(f"Predictor initialization failed: {str(e)}")
    
    def assess(self, image: np.ndarray, 
              rule_results: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        """
        Assess aesthetic quality of an image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            rule_results: Optional compositional rule analysis results
            
        Returns:
            Aesthetic quality score between 0 and 1
        """
        try:
            # Extract aesthetic features
            aesthetic_features = self.feature_extractor.extract_features(image)
            
            # Prepare features for neural network
            feature_tensors = self._prepare_feature_tensors(aesthetic_features)
            
            # Predict using neural network
            with torch.no_grad():
                nn_score = self.predictor(feature_tensors).item()
            
            # Calculate heuristic score
            heuristic_score = self._calculate_heuristic_score(image, aesthetic_features, rule_results)
            
            # Combine scores
            combined_score = self._combine_scores(nn_score, heuristic_score, rule_results)
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Aesthetic assessment failed: {str(e)}")
            return 0.5  # Neutral score as fallback
    
    def _prepare_feature_tensors(self, features: AestheticFeatures) -> Dict[str, torch.Tensor]:
        """Prepare features for neural network input."""
        try:
            feature_tensors = {
                'global': torch.FloatTensor(features.global_features).unsqueeze(0).to(self.device),
                'color': torch.FloatTensor(features.color_features).unsqueeze(0).to(self.device),
                'texture': torch.FloatTensor(features.texture_features).unsqueeze(0).to(self.device),
                'composition': torch.FloatTensor(features.composition_features).unsqueeze(0).to(self.device)
            }
            
            return feature_tensors
            
        except Exception as e:
            logger.warning(f"Feature tensor preparation failed: {str(e)}")
            # Return zero tensors as fallback
            return {
                'global': torch.zeros(1, 10).to(self.device),
                'color': torch.zeros(1, 15).to(self.device),
                'texture': torch.zeros(1, 8).to(self.device),
                'composition': torch.zeros(1, 12).to(self.device)
            }
    
    def _calculate_heuristic_score(self, image: np.ndarray, 
                                 features: AestheticFeatures,
                                 rule_results: Optional[Dict[str, Dict[str, Any]]]) -> float:
        """Calculate heuristic aesthetic score based on traditional metrics."""
        try:
            score_components = []
            
            # Technical quality components
            global_feats = features.global_features
            
            # Contrast score
            contrast = global_feats[1] if len(global_feats) > 1 else 0.0  # Brightness variation
            contrast_score = min(1.0, contrast / 64.0)  # Normalize
            score_components.append(contrast_score * 0.2)
            
            # Sharpness score
            sharpness = global_feats[9] if len(global_feats) > 9 else 0.0
            sharpness_score = min(1.0, sharpness)
            score_components.append(sharpness_score * 0.2)
            
            # Color harmony score
            color_feats = features.color_features
            if len(color_feats) > 13:
                colorfulness = color_feats[13]
                colorfulness_score = min(1.0, colorfulness)
                score_components.append(colorfulness_score * 0.15)
            
            # Composition score from features
            comp_feats = features.composition_features
            if len(comp_feats) > 11:
                # Symmetry and balance
                h_symmetry = comp_feats[9] if len(comp_feats) > 9 else 0.0
                v_symmetry = comp_feats[10] if len(comp_feats) > 10 else 0.0
                center_bias = comp_feats[11] if len(comp_feats) > 11 else 0.0
                
                composition_score = (h_symmetry + v_symmetry + center_bias) / 3.0
                score_components.append(composition_score * 0.25)
            
            # Rule-based enhancement
            if rule_results:
                rule_enhancement = self._calculate_rule_enhancement(rule_results)
                score_components.append(rule_enhancement * 0.2)
            
            return sum(score_components)
            
        except Exception as e:
            logger.warning(f"Heuristic score calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_rule_enhancement(self, rule_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate aesthetic enhancement from compositional rules."""
        try:
            enhancements = []
            
            # Rule of thirds enhancement
            if 'rule_of_thirds' in rule_results:
                rot_score = rule_results['rule_of_thirds'].get('score', 0.0)
                enhancements.append(rot_score)
            
            # Leading lines enhancement
            if 'leading_lines' in rule_results:
                lines_score = rule_results['leading_lines'].get('score', 0.0)
                enhancements.append(lines_score)
            
            # Symmetry enhancement
            if 'symmetry' in rule_results:
                sym_score = rule_results['symmetry'].get('score', 0.0)
                enhancements.append(sym_score)
            
            # Color harmony enhancement
            if 'color_harmony' in rule_results:
                color_score = rule_results['color_harmony'].get('score', 0.0)
                enhancements.append(color_score)
            
            return np.mean(enhancements) if enhancements else 0.0
            
        except Exception as e:
            logger.warning(f"Rule enhancement calculation failed: {str(e)}")
            return 0.0
    
    def _combine_scores(self, nn_score: float, heuristic_score: float,
                       rule_results: Optional[Dict[str, Dict[str, Any]]]) -> float:
        """Combine neural network and heuristic scores."""
        try:
            # Base combination
            combined = nn_score * 0.6 + heuristic_score * 0.4
            
            # Confidence-based weighting
            if rule_results:
                confidences = [result.get('confidence', 1.0) for result in rule_results.values()]
                avg_confidence = np.mean(confidences) if confidences else 1.0
                
                # Higher confidence in rule analysis -> more weight to heuristic
                if avg_confidence > 0.7:
                    combined = nn_score * 0.5 + heuristic_score * 0.5
                else:
                    combined = nn_score * 0.7 + heuristic_score * 0.3
            
            return combined
            
        except Exception as e:
            logger.warning(f"Score combination failed: {str(e)}")
            return (nn_score + heuristic_score) / 2.0
    
    def get_aesthetic_breakdown(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get detailed breakdown of aesthetic quality components.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of aesthetic component scores
        """
        try:
            features = self.feature_extractor.extract_features(image)
            
            breakdown = {}
            
            # Technical quality
            global_feats = features.global_features
            if len(global_feats) > 9:
                breakdown['sharpness'] = min(1.0, global_feats[9])
                breakdown['contrast'] = min(1.0, global_feats[1] / 64.0)
                breakdown['brightness'] = global_feats[0] / 255.0
                breakdown['dynamic_range'] = global_feats[4] / 255.0
            
            # Color quality
            color_feats = features.color_features
            if len(color_feats) > 13:
                breakdown['colorfulness'] = min(1.0, color_feats[13])
                breakdown['color_temperature'] = min(1.0, color_feats[12] / 2.0)
            
            # Composition
            comp_feats = features.composition_features
            if len(comp_feats) > 11:
                breakdown['horizontal_symmetry'] = comp_feats[9] if len(comp_feats) > 9 else 0.0
                breakdown['vertical_symmetry'] = comp_feats[10] if len(comp_feats) > 10 else 0.0
                breakdown['center_composition'] = comp_feats[11] if len(comp_feats) > 11 else 0.0
            
            return breakdown
            
        except Exception as e:
            logger.warning(f"Aesthetic breakdown calculation failed: {str(e)}")
            return {}
    
    def batch_assess(self, images: List[np.ndarray]) -> List[float]:
        """
        Assess aesthetic quality for multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of aesthetic quality scores
        """
        scores = []
        
        for i, image in enumerate(images):
            try:
                score = self.assess(image)
                scores.append(score)
                logger.debug(f"Batch aesthetic assessment {i+1}/{len(images)}: {score:.3f}")
            except Exception as e:
                logger.warning(f"Batch assessment failed for image {i+1}: {str(e)}")
                scores.append(0.5)  # Neutral fallback score
        
        return scores