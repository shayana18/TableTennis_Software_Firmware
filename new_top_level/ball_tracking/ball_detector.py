"""
Shared Ball Detector - Single source of truth for ball detection logic.

Extracted from detection_experiments/stereo_detection.py (ground truth).
MOG2 background subtraction + contour-based multi-criteria scoring.

All modules that need ball detection should import BallDetector from here
instead of implementing their own detection pipeline.

Detection pipeline:
    1. MOG2 foreground mask (history=300, varThreshold=40)
    2. Morphological cleanup (MORPH_OPEN 3x3, MORPH_CLOSE 7x7)
    3. Contour extraction + filtering (area, circularity)
    4. Multi-criteria scoring (circularity + proximity + orange color)
"""

import cv2
import numpy as np


class BallDetector:
    """
    Ball detector using MOG2 background subtraction + contour scoring.

    This is the canonical detection implementation. All parameters and logic
    match detection_experiments/stereo_detection.py exactly.
    """

    def __init__(self, min_area=30, max_area=2000, min_circularity=0.30,
                 search_radius=150, roi=None):
        """
        Args:
            min_area: Minimum contour area in pixels (default 80)
            max_area: Maximum contour area in pixels (default 2000)
            min_circularity: Minimum circularity 0-1 (default 0.35)
            search_radius: Proximity search radius in pixels (default 150)
            roi: Region of interest as (x, y, w, h) or None for full frame
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.search_radius = search_radius
        self.roi = roi

        # Scoring weights (from stereo_detection.py)
        self.w_circularity = 0.4
        self.w_proximity = 0.3
        self.w_color = 0.3
        self.w_no_history = 0.15

        # MOG2 background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Tracking state
        self.last_pos = None

    def _get_roi(self, frame):
        """Crop frame to ROI, or return full frame if ROI is None."""
        if self.roi is None:
            return frame, 0, 0
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w], x, y

    def build_background(self, frame):
        """Feed frame to MOG2 with fast learning rate (0.05) for background building.

        Call this repeatedly during the background learning phase before
        starting detection. Returns the foreground mask.
        """
        roi_frame, _, _ = self._get_roi(frame)
        fg_mask = self.bg_sub.apply(roi_frame, learningRate=0.05)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return fg_mask

    def detect(self, frame):
        """
        Run MOG2 + contour detection on a frame.

        Returns:
            (best_candidate, all_candidates, rejected, fg_mask)

            best_candidate: dict or None, with keys:
                center: (cx, cy) float coordinates (ROI-relative if ROI set)
                area: contour area in pixels
                circularity: 0-1
                score: weighted detection score
                contour: cv2 contour array
                is_orange: bool
            all_candidates: list of candidate dicts that passed all filters
            rejected: list of rejected dicts with 'reason' field
            fg_mask: binary foreground mask after morphological cleanup
        """
        roi_frame, ox, oy = self._get_roi(frame)

        fg_mask = self.bg_sub.apply(roi_frame, learningRate=0.002)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = -1
        all_candidates = []
        rejected = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            if area < self.min_area:
                rejected.append({'center': (cx, cy), 'area': area,
                                 'reason': 'SMALL'})
                continue
            if area > self.max_area:
                rejected.append({'center': (cx, cy), 'area': area,
                                 'reason': 'BIG'})
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                rejected.append({'center': (cx, cy), 'area': area,
                                 'circularity': circularity, 'reason': 'SHAPE'})
                continue

            # --- Scoring ---
            score = circularity * self.w_circularity

            # Proximity to last known position
            if self.last_pos is not None:
                dist = np.hypot(cx - self.last_pos[0], cy - self.last_pos[1])
                if dist < self.search_radius:
                    score += (1.0 - dist / self.search_radius) * self.w_proximity
            else:
                score += self.w_no_history

            # Orange ball color boost
            mask_temp = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(roi_frame, mask=mask_temp)[:3]
            pixel = np.uint8([[list(mean_bgr)]])
            hsv_val = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])

            is_orange = (5 <= hue <= 25 and sat > 80 and val > 80)
            if is_orange:
                score += self.w_color

            candidate = {
                'center': (cx, cy),
                'area': area,
                'circularity': circularity,
                'score': score,
                'contour': cnt,
                'is_orange': is_orange
            }
            all_candidates.append(candidate)

            if score > best_score:
                best_score = score
                best_candidate = candidate

        # Update tracking state
        if best_candidate is not None:
            self.last_pos = best_candidate['center']

        return best_candidate, all_candidates, rejected, fg_mask

    def reset(self):
        """Reset MOG2 background model and tracking state."""
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self.last_pos = None
