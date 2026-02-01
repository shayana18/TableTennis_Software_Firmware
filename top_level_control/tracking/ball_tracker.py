"""
Enhanced Ball Tracker for Table Tennis

Robust ball detection using HSV + LAB color space fusion.
- HSV: Good for consistent lighting
- LAB: Lighting-independent color detection
- Fusion: 55% HSV + 45% LAB (configurable)

Note: This file is IDENTICAL for Arducam and Basler pipelines.
      Ball detection is camera-agnostic - it works on any BGR frame.
"""

import cv2
import numpy as np


class EnhancedBallTracker:
    """
    Ball detector using HSV + LAB fusion.
    
    Fusion weights: 55% HSV, 45% LAB (adjustable via hsv_weight)
    """

    def __init__(self):
        # HSV thresholds for orange ball
        self.hsv_lower = np.array([0, 78, 69], dtype=np.uint8)
        self.hsv_upper = np.array([50, 255, 255], dtype=np.uint8)
        
        # LAB thresholds for orange ball
        self.lab_lower = np.array([16, 125, 160], dtype=np.uint8)
        self.lab_upper = np.array([255, 248, 255], dtype=np.uint8)
        
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Detection parameters
        self.min_area = 100
        self.min_circularity = 0.6
        self.min_solidity = 0.7
        self.max_aspect_ratio = 1.6
        
        # Morphological kernels
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Fusion weight: 0.55 = 55% HSV, 45% LAB
        self.hsv_weight = 0.55

    def preprocess(self, frame):
        """Apply CLAHE to normalize lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def detect_hsv(self, frame):
        """Detect orange using HSV color space."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

    def detect_lab(self, frame):
        """Detect orange using LAB color space (lighting-independent)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self.lab_lower, self.lab_upper)
        return mask

    def fuse_masks(self, mask_hsv, mask_lab):
        """
        Combine HSV and LAB masks.
        
        Default: 55% HSV + 45% LAB
        """
        m_hsv = mask_hsv.astype(np.float32) / 255.0
        m_lab = mask_lab.astype(np.float32) / 255.0
        
        # Weighted fusion
        fused = m_hsv * self.hsv_weight + m_lab * (1 - self.hsv_weight)
        
        # Threshold: if >= 40% confidence, consider detected
        mask_fused = (fused > 0.4).astype(np.uint8) * 255
        
        return mask_fused

    def clean_mask(self, mask):
        """Apply morphological operations to clean up mask."""
        mask = cv2.erode(mask, self.kernel_small, iterations=2)
        mask = cv2.dilate(mask, self.kernel_medium, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def find_best_ball(self, mask):
        """Find the best circular contour in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Circularity check
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0
            if circularity < self.min_circularity:
                continue
            
            # Solidity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                continue
            
            # Aspect ratio check
            x_rect, y_rect, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            if aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Score: prefer larger, more circular contours
            score = area * circularity * solidity
            
            if score > best_score:
                best_score = score
                best_candidate = {
                    'center': (int(x), int(y)),
                    'radius': radius,
                    'contour': contour,
                    'circularity': circularity,
                    'solidity': solidity
                }
        
        return best_candidate

    def detect(self, frame, return_debug=False):
        """
        Detect the ball in a frame.
        
        Args:
            frame: BGR image
            return_debug: If True, return intermediate masks
        
        Returns:
            dict with 'found', 'center', 'radius', 'mask', 'confidence', 'debug'
        """
        result = {
            'found': False,
            'center': None,
            'radius': None,
            'contour': None,
            'confidence': 0.0,
            'mask': None
        }
        
        # Preprocess
        frame_normalized = self.preprocess(frame)
        
        # Detect in both color spaces
        mask_hsv = self.detect_hsv(frame_normalized)
        mask_lab = self.detect_lab(frame_normalized)
        
        # Fuse masks
        mask_fused = self.fuse_masks(mask_hsv, mask_lab)
        
        # Clean up
        mask_clean = self.clean_mask(mask_fused)
        result['mask'] = mask_clean
        
        # Find ball
        candidate = self.find_best_ball(mask_clean)
        
        if candidate:
            result['found'] = True
            result['center'] = candidate['center']
            result['radius'] = candidate['radius']
            result['contour'] = candidate['contour']
            result['confidence'] = candidate['circularity'] * candidate['solidity']
        
        if return_debug:
            result['debug'] = {
                'frame_normalized': frame_normalized,
                'mask_hsv': mask_hsv,
                'mask_lab': mask_lab,
                'mask_fused': mask_fused
            }
        
        return result

    def set_hsv_thresholds(self, lower, upper):
        """Update HSV thresholds."""
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def set_lab_thresholds(self, lower, upper):
        """Update LAB thresholds."""
        self.lab_lower = np.array(lower, dtype=np.uint8)
        self.lab_upper = np.array(upper, dtype=np.uint8)

    def set_fusion_weight(self, hsv_weight):
        """
        Set fusion weight.
        
        Args:
            hsv_weight: 0.0 to 1.0 (0.55 = 55% HSV, 45% LAB)
        """
        self.hsv_weight = max(0.0, min(1.0, hsv_weight))


# Backwards compatibility
BallTracker = EnhancedBallTracker