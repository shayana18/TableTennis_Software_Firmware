"""
HSV Color-based Ball Detection for Table Tennis

Simple and efficient ball detection using HSV color thresholding.
Only handles detection - no physics or prediction logic.
"""

import cv2
import numpy as np

# HSV thresholds for orange table tennis ball
# Narrower hue range (5-20) to avoid red/yellow, higher saturation minimum
DEFAULT_COLOR_LOWER = (5, 120, 120)
DEFAULT_COLOR_UPPER = (20, 255, 255)


class BallTracker:
    """HSV color-based ball detector with strict circularity checks."""

    def __init__(self, color_lower=DEFAULT_COLOR_LOWER, color_upper=DEFAULT_COLOR_UPPER,
                 min_radius=8, max_radius=80):
        """
        Initialize the ball tracker.

        Args:
            color_lower: Lower HSV threshold (H, S, V)
            color_upper: Upper HSV threshold (H, S, V)
            min_radius: Minimum ball radius to detect (pixels)
            max_radius: Maximum ball radius to detect (pixels)
        """
        self.color_lower = np.array(color_lower, dtype=np.uint8)
        self.color_upper = np.array(color_upper, dtype=np.uint8)
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Stricter circularity requirements
        self.min_circularity = 0.7  # Must be at least 70% circular
        self.min_solidity = 0.8     # Contour must fill 80% of convex hull
        self.max_aspect_ratio = 1.4 # Bounding box must be roughly square
        self.min_area = 200         # Minimum contour area in pixels

        # Morphological kernel for noise reduction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        """
        Detect the ball in a frame.

        Args:
            frame: BGR image from camera

        Returns:
            dict with keys:
                - 'found': bool, whether ball was detected
                - 'center': (x, y) tuple or None
                - 'radius': float or None
                - 'contour': contour points or None
                - 'mask': binary mask image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for orange color
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # Clean up mask with morphological operations
        mask = cv2.erode(mask, self.kernel, iterations=3)
        mask = cv2.dilate(mask, self.kernel, iterations=3)

        # Smooth edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = {
            'found': False,
            'center': None,
            'radius': None,
            'contour': None,
            'mask': mask
        }

        if not contours:
            return result

        # Find the best contour that passes all circularity checks
        best_contour = None
        best_score = 0
        best_center = None
        best_radius = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Check 1: Minimum area to reject noise
            if area < self.min_area:
                continue

            # Check 2: Radius constraints
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if not (self.min_radius <= radius <= self.max_radius):
                continue

            # Check 3: Circularity - how circular is the contour?
            circle_area = np.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0
            if circularity < self.min_circularity:
                continue

            # Check 4: Solidity - contour area vs convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                continue

            # Check 5: Aspect ratio - bounding box should be roughly square
            x_rect, y_rect, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # Score: prefer larger, more circular contours
            score = area * circularity * solidity
            if score > best_score:
                best_score = score
                best_radius = radius
                best_center = (int(x), int(y))
                best_contour = contour

        if best_center is not None:
            result['found'] = True
            result['center'] = best_center
            result['radius'] = best_radius
            result['contour'] = best_contour

        return result

    def set_hsv_thresholds(self, lower, upper):
        """Update HSV thresholds for different lighting conditions."""
        self.color_lower = np.array(lower, dtype=np.uint8)
        self.color_upper = np.array(upper, dtype=np.uint8)
