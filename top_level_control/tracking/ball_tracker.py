"""
Enhanced Ball Tracker for Table Tennis

Motion-based ball detection using MOG2 background subtraction.
Ported from detection_experiments/step3_contour_filtering.py.
- MOG2 foreground mask isolates moving objects
- Contour filtering by area + circularity
- Scoring: circularity + proximity + orange color boost
"""

import cv2
import numpy as np


class EnhancedBallTracker:
    """
    Ball detector using MOG2 background subtraction + contour scoring.

    Replaces HSV+LAB fusion with a motion-based approach that is
    lighting-invariant and requires no manual threshold tuning.
    """

    def __init__(self):
        # MOG2 background subtractor
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)

        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Detection parameters (from step3)
        self.min_area = 150
        self.max_area = 1100
        self.min_circularity = 0.45

        # Proximity tracking
        self.last_pos = None
        self.search_radius = 150

        # Scoring weights (sum ~1.0 at max)
        self.w_circularity = 0.4
        self.w_proximity = 0.3
        self.w_color = 0.3
        self.w_no_history = 0.15

        # Warmup: MOG2 needs frames to build a background model
        self._frame_count = 0
        self._warmup_frames = 300
        self._is_warmed_up = False

    def _get_fg_mask(self, frame):
        """Apply MOG2 and morphological cleanup, return binary fg mask."""
        self._frame_count += 1

        if not self._is_warmed_up:
            if self._frame_count >= self._warmup_frames:
                self._is_warmed_up = True
            learning_rate = 0.05
        else:
            learning_rate = 0.002

        fg_mask = self.bg_sub.apply(frame, learningRate=learning_rate)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return fg_mask

    def find_best_ball(self, mask, frame):
        """
        Find the best ball candidate using contour scoring.

        Ported directly from step3_contour_filtering.py:
        1. findContours on fg_mask
        2. Filter: area 150-1100, circularity > 0.45
        3. Score = circularity*0.4 + proximity(up to 0.3) + orange boost(0.3)
        4. minEnclosingCircle for radius
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        best_candidate = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

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
            mask_temp = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_temp, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(frame, mask=mask_temp)[:3]
            pixel = np.uint8([[list(mean_bgr)]])
            hsv_val = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
            hue, sat, val = int(hsv_val[0]), int(hsv_val[1]), int(hsv_val[2])

            if 5 <= hue <= 25 and sat > 80 and val > 80:
                score += self.w_color

            # Enclosing circle for radius
            (ex, ey), radius = cv2.minEnclosingCircle(cnt)

            if score > best_score:
                best_score = score
                best_candidate = {
                    'center': (int(cx), int(cy)),
                    'radius': radius,
                    'contour': cnt,
                    'circularity': circularity,
                    'score': score
                }

        return best_candidate

    def detect(self, frame, return_debug=False):
        """
        Detect the ball in a frame.

        Args:
            frame: BGR image
            return_debug: If True, return fg_mask in debug dict

        Returns:
            dict with 'found', 'center', 'radius', 'contour', 'confidence', 'mask'
        """
        result = {
            'found': False,
            'center': None,
            'radius': None,
            'contour': None,
            'confidence': 0.0,
            'mask': None
        }

        fg_mask = self._get_fg_mask(frame)
        result['mask'] = fg_mask

        if not self._is_warmed_up:
            if return_debug:
                result['debug'] = {'fg_mask': fg_mask}
            return result

        candidate = self.find_best_ball(fg_mask, frame)

        if candidate:
            result['found'] = True
            result['center'] = candidate['center']
            result['radius'] = candidate['radius']
            result['contour'] = candidate['contour']
            result['confidence'] = min(1.0, candidate['score'])
            self.last_pos = candidate['center']

        if return_debug:
            result['debug'] = {'fg_mask': fg_mask}

        return result

    # --- Legacy no-op setters (prevent crashes in consumers) ---

    def set_hsv_thresholds(self, lower, upper):
        """No-op. HSV thresholds not used with MOG2 detection."""
        pass

    def set_lab_thresholds(self, lower, upper):
        """No-op. LAB thresholds not used with MOG2 detection."""
        pass

    def set_fusion_weight(self, hsv_weight):
        """No-op. Fusion weight not used with MOG2 detection."""
        pass

    # --- New MOG2-specific methods ---

    def reset_background(self):
        """Recreate MOG2 subtractor and reset tracking state."""
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False)
        self._frame_count = 0
        self._is_warmed_up = False
        self.last_pos = None

    def is_ready(self):
        """Return True once the background model warmup is complete."""
        return self._is_warmed_up

    def get_warmup_progress(self):
        """Return warmup progress as 0.0 to 1.0."""
        if self._is_warmed_up:
            return 1.0
        return min(1.0, self._frame_count / self._warmup_frames)


# Backwards compatibility
BallTracker = EnhancedBallTracker
