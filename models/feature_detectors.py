"""
Feature Detectors for Composition Analysis

This module implements specialized detectors for various compositional elements
including rule of thirds, leading lines, symmetry, and depth analysis.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
# SIFT will be initialized in the class using cv2.SIFT_create()
from skimage.measure import ransac, LineModelND  # Changed import location for ransac


class RuleOfThirdsDetector:
    """
    Advanced Rule of Thirds detector using saliency maps and CNN predictions.
    """
    
    def __init__(self, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Rule of Thirds detector.
        
        Args:
            model: Pre-trained model for saliency prediction (optional)
            device: Device to run the model on
        """
        self.model = model
        self.device = device
        
    def detect(self, image):
        """
        Detect Rule of Thirds points in an image.
        
        Args:
            image: Input image (numpy array or torch tensor)
            
        Returns:
            dict: Dictionary containing detected points and confidence scores
        """
        # Convert image to torch tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image = image.float() / 255.0
        
        # Get saliency map from model or compute basic saliency
        if self.model is not None:
            with torch.no_grad():
                saliency = self.model(image.to(self.device))
        else:
            # Fallback to basic saliency computation
            if isinstance(image, torch.Tensor):
                img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = image
            
            # Convert to grayscale for saliency
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Fallback saliency computation using gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize gradient magnitude
            if np.max(gradient_magnitude) > 0:
                saliency = torch.from_numpy(gradient_magnitude / np.max(gradient_magnitude)).unsqueeze(0)
            else:
                # Ultimate fallback - uniform saliency
                saliency = torch.ones((1, gray.shape[0], gray.shape[1])) * 0.5
        
        # Define Rule of Thirds grid points
        h, w = saliency.shape[-2:]
        grid_points = [
            (w//3, h//3), (w//2, h//3), (2*w//3, h//3),
            (w//3, h//2), (w//2, h//2), (2*w//3, h//2),
            (w//3, 2*h//3), (w//2, 2*h//3), (2*w//3, 2*h//3)
        ]
        
        # Calculate saliency scores for each grid point
        scores = []
        window_size = min(w, h) // 10
        
        for x, y in grid_points:
            x1, x2 = max(0, x-window_size), min(w, x+window_size)
            y1, y2 = max(0, y-window_size), min(h, y+window_size)
            region = saliency[..., y1:y2, x1:x2]
            score = region.mean().item()
            scores.append(score)
        
        # Normalize scores
        scores = torch.tensor(scores)
        scores = F.softmax(scores, dim=0)
        
        return {
            'points': grid_points,
            'scores': scores.tolist(),
            'saliency_map': saliency.cpu().numpy()
        }


class LeadingLinesDetector:
    """
    Advanced leading lines detector using Hough Transform and RANSAC.
    """
    
    def __init__(self, min_line_length=50, max_line_gap=10, threshold=50):
        """
        Initialize the leading lines detector.
        
        Args:
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments
            threshold: Threshold for line detection
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold = threshold
        
    def detect(self, image):
        """
        Detect leading lines in an image.
        
        Args:
            image: Input image
            
        Returns:
            dict: Dictionary containing detected lines and their properties
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Initial line detection using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return {'lines': [], 'scores': [], 'vanishing_points': []}
        
        # Convert lines to point pairs for RANSAC
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        points = np.array(points)
        
        # Use RANSAC to find dominant lines
        model = LineModelND()
        model.estimate(points)
        
        # Find inliers
        inliers = model.residuals(points) < 5.0
        
        # Refine lines using RANSAC results
        refined_lines = []
        scores = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line strength based on edge response
            line_mask = np.zeros_like(edges)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            strength = np.sum(edges * line_mask) / np.sum(line_mask > 0)
            
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Combine metrics for final score
            score = strength * length / (image.shape[0] * image.shape[1])
            
            refined_lines.append([x1, y1, x2, y2])
            scores.append(float(score))
        
        # Find vanishing points
        vanishing_points = self._find_vanishing_points(refined_lines)
        
        return {
            'lines': refined_lines,
            'scores': scores,
            'vanishing_points': vanishing_points
        }
    
    def _find_vanishing_points(self, lines):
        """
        Find vanishing points from detected lines.
        
        Args:
            lines: List of detected lines
            
        Returns:
            list: List of vanishing points
        """
        if len(lines) < 2:
            return []
            
        vanishing_points = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i]
                x3, y3, x4, y4 = lines[j]
                
                # Line vectors
                v1 = np.array([x2-x1, y2-y1])
                v2 = np.array([x4-x3, y4-y3])
                
                # Normalize vectors
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Check if lines are nearly parallel
                if abs(np.dot(v1, v2)) > 0.95:
                    continue
                
                # Find intersection
                A = np.array([[x2-x1, x3-x4], [y2-y1, y3-y4]])
                b = np.array([x3-x1, y3-y1])
                
                try:
                    t = np.linalg.solve(A, b)
                    vp = np.array([x1 + t[0]*(x2-x1), y1 + t[0]*(y2-y1)])
                    vanishing_points.append(vp.tolist())
                except np.linalg.LinAlgError:
                    continue
                
        return vanishing_points


class SymmetryDetector:
    """
    Symmetry detector using SIFT features and ViT attention maps.
    """
    
    def __init__(self, model=None):
        """
        Initialize symmetry detector.
        
        Args:
            model: Pre-trained ViT model for attention maps (optional)
        """
        self.model = model
        self.sift = cv2.SIFT_create()
        
    def detect(self, image):
        """
        Detect symmetry in an image.
        
        Args:
            image: Input image
            
        Returns:
            dict: Dictionary containing symmetry information
        """
        # Convert to grayscale for SIFT
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Extract SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) < 2:
            return {
                'horizontal_score': 0.0,
                'vertical_score': 0.0,
                'radial_score': 0.0,
                'dominant_type': None
            }
        
        # Check horizontal symmetry
        flipped_h = np.fliplr(gray)
        h_score = self._compute_symmetry_score(gray, flipped_h, keypoints, descriptors)
        
        # Check vertical symmetry
        flipped_v = np.flipud(gray)
        v_score = self._compute_symmetry_score(gray, flipped_v, keypoints, descriptors)
        
        # Check radial symmetry
        center = (gray.shape[1]//2, gray.shape[0]//2)
        r_score = self._compute_radial_symmetry(gray, center)
        
        # Get attention-based symmetry if model available
        if self.model is not None:
            attention_scores = self._get_attention_scores(image)
            h_score = (h_score + attention_scores['horizontal']) / 2
            v_score = (v_score + attention_scores['vertical']) / 2
            r_score = (r_score + attention_scores['radial']) / 2
        
        # Determine dominant symmetry type
        scores = {'horizontal': h_score, 'vertical': v_score, 'radial': r_score}
        dominant_type = max(scores, key=scores.get)
        
        return {
            'horizontal_score': float(h_score),
            'vertical_score': float(v_score),
            'radial_score': float(r_score),
            'dominant_type': dominant_type
        }
    
    def _compute_symmetry_score(self, img1, img2, keypoints, descriptors):
        """
        Compute symmetry score between two images using SIFT features.
        """
        # Extract features from flipped image
        kp2, desc2 = self.sift.detectAndCompute(img2, None)
        
        if desc2 is None or len(desc2) < 2:
            return 0.0
        
        # Match features
        matches = []
        for i, desc1 in enumerate(descriptors):
            distances = np.linalg.norm(desc2 - desc1, axis=1)
            best_match = np.argmin(distances)
            if distances[best_match] < 0.7 * np.partition(distances, 1)[1]:
                matches.append((i, best_match))
        
        if not matches:
            return 0.0
        
        # Calculate symmetry score based on match quality
        match_distances = []
        for idx1, idx2 in matches:
            pt1 = np.array(keypoints[idx1].pt)
            pt2 = np.array(kp2[idx2].pt)
            dist = np.linalg.norm(pt1 - pt2)
            match_distances.append(dist)
        
        return 1.0 / (1.0 + np.mean(match_distances))
    
    def _compute_radial_symmetry(self, image, center):
        """
        Compute radial symmetry score.
        """
        y, x = np.indices(image.shape)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        
        # Compute average intensity at each radius
        tbin = np.bincount(r.ravel(), image.ravel())
        nr = np.bincount(r.ravel())
        radial_mean = tbin / nr
        
        # Compute radial symmetry score
        reconstructed = radial_mean[r]
        error = np.abs(image - reconstructed)
        score = 1.0 / (1.0 + np.mean(error))
        
        return score
    
    def _get_attention_scores(self, image):
        """
        Get symmetry scores from ViT attention maps.
        """
        # Convert image to tensor and get attention maps
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image = image.float() / 255.0
        
        with torch.no_grad():
            attention_maps = self.model(image, output_attentions=True).attentions[-1]
            
        # Average attention across heads
        attention = attention_maps.mean(1).mean(0)
        
        # Compute symmetry scores from attention patterns
        h_score = self._attention_symmetry_score(attention, 'horizontal')
        v_score = self._attention_symmetry_score(attention, 'vertical')
        r_score = self._attention_symmetry_score(attention, 'radial')
        
        return {
            'horizontal': h_score,
            'vertical': v_score,
            'radial': r_score
        }
    
    def _attention_symmetry_score(self, attention, symmetry_type):
        """
        Compute symmetry score from attention map.
        """
        if symmetry_type == 'horizontal':
            flipped = torch.flip(attention, [1])
        elif symmetry_type == 'vertical':
            flipped = torch.flip(attention, [0])
        else:  # radial
            center = (attention.shape[1]//2, attention.shape[0]//2)
            y, x = torch.meshgrid(torch.arange(attention.shape[0]),
                                torch.arange(attention.shape[1]))
            r = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
            return 1.0 / (1.0 + torch.mean(torch.abs(attention - attention[r == r])))
        
        return 1.0 / (1.0 + torch.mean(torch.abs(attention - flipped)))


class DepthAnalyzer:
    """
    Depth perception and layering analysis using DenseNet-169 with ViT integration.
    """
    
    def __init__(self, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize depth analyzer.
        
        Args:
            model: Pre-trained depth estimation model
            device: Device to run the model on
        """
        self.model = model
        self.device = device
        
    def analyze(self, image):
        """
        Analyze depth and layering in an image.
        
        Args:
            image: Input image
            
        Returns:
            dict: Dictionary containing depth map and layering information
        """
        # Convert image to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image = image.float() / 255.0
        
        # Get depth prediction from model or use fallback
        if self.model is not None:
            with torch.no_grad():
                depth_map = self.model(image.to(self.device))
                depth_map = depth_map.cpu().numpy().squeeze()
        else:
            # Fallback depth estimation using basic image processing
            depth_map = self._fallback_depth_estimation(image)
        
        # Analyze depth layers
        layers = self._analyze_layers(depth_map)
        
        return {
            'depth_map': depth_map,
            'layers': layers,
            'focal_points': self._find_focal_points(depth_map),
            'depth_statistics': self._compute_depth_statistics(depth_map)
        }
    
    def _analyze_layers(self, depth_map):
        """
        Analyze depth layers in the image.
        """
        # Use histogram analysis to find distinct depth layers
        hist, bins = np.histogram(depth_map, bins=10)
        
        # Find peaks in histogram to identify distinct layers
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append({
                    'depth': (bins[i] + bins[i+1]) / 2,
                    'prominence': hist[i],
                    'range': (bins[i], bins[i+1])
                })
        
        # Sort layers by depth
        peaks.sort(key=lambda x: x['depth'])
        
        return peaks
    
    def _find_focal_points(self, depth_map):
        """
        Find potential focal points based on depth discontinuities.
        """
        # Compute depth gradients
        grad_y, grad_x = np.gradient(depth_map)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find local maxima in gradient magnitude
        local_max = ndimage.maximum_filter(gradient_magnitude, size=20)
        focal_points = []
        
        y, x = np.where((gradient_magnitude == local_max) & 
                       (gradient_magnitude > np.mean(gradient_magnitude)))
        
        for i in range(len(y)):
            focal_points.append({
                'position': (int(x[i]), int(y[i])),
                'depth': float(depth_map[y[i], x[i]]),
                'strength': float(gradient_magnitude[y[i], x[i]])
            })
        
        # Sort by strength
        focal_points.sort(key=lambda x: x['strength'], reverse=True)
        
        return focal_points[:5]  # Return top 5 focal points
    
    def _fallback_depth_estimation(self, image):
        """
        Fallback depth estimation using basic image processing techniques.
        
        Args:
            image: Input tensor or numpy array
            
        Returns:
            depth_map: Estimated depth map as numpy array
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:
                img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = image
        
        # Convert to grayscale for depth estimation
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Use blur as a simple depth cue (more blur = farther)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Use gradient magnitude as depth cue (edges = closer)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine brightness and gradient for depth estimation
        # Normalize to [0, 1]
        brightness_norm = gray.astype(np.float32) / 255.0
        gradient_norm = gradient_magnitude / np.max(gradient_magnitude)
        
        # Simple depth estimation: brighter and more detailed = closer
        depth_map = 0.7 * brightness_norm + 0.3 * gradient_norm
        
        return depth_map

    def _compute_depth_statistics(self, depth_map):
        """
        Compute statistical measures of depth distribution.
        """
        return {
            'mean_depth': float(np.mean(depth_map)),
            'std_depth': float(np.std(depth_map)),
            'min_depth': float(np.min(depth_map)),
            'max_depth': float(np.max(depth_map)),
            'depth_range': float(np.max(depth_map) - np.min(depth_map))
        }