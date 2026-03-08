"""
Enhanced Ball Tracker for Table Tennis

Wraps BallDetector with automatic warmup and a standardized
result dict interface used by all tracking/test scripts.

Detection logic lives in ball_detector.py (single source of truth).
This class adds:
- Automatic warmup (300 frames at learningRate=0.05)
- Standardized result dict interface (found/center/radius/contour/confidence/mask)
- Legacy no-op setters for backward compatibility
"""

import cv2
import numpy as np
from .ball_detector import BallDetector


class EnhancedBallTracker:
    """
    Ball detector with automatic MOG2 warmup.

    Uses BallDetector for all detection logic.
    """

    def __init__(self):
        # All detection logic delegated to BallDetector
        self.detector = BallDetector()

        # Warmup: MOG2 needs frames to build a background model
        self._frame_count = 0
        self._warmup_frames = 300
        self._is_warmed_up = False

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

        # During warmup, feed frames to build background model
        if not self._is_warmed_up:
            self._frame_count += 1
            if self._frame_count >= self._warmup_frames:
                self._is_warmed_up = True
            fg_mask = self.detector.build_background(frame)
            result['mask'] = fg_mask
            if return_debug:
                result['debug'] = {'fg_mask': fg_mask}
            return result

        # Main detection via BallDetector
        best, all_candidates, rejected, fg_mask = self.detector.detect(frame)
        result['mask'] = fg_mask

        if best:
            result['found'] = True
            result['center'] = (int(best['center'][0]), int(best['center'][1]))
            # Compute radius via minEnclosingCircle for consumer compatibility
            (_, _), radius = cv2.minEnclosingCircle(best['contour'])
            result['radius'] = radius
            result['contour'] = best['contour']
            result['confidence'] = min(1.0, best['score'])

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

    # --- MOG2-specific methods ---

    def reset_background(self):
        """Recreate MOG2 subtractor and reset tracking state."""
        self.detector.reset()
        self._frame_count = 0
        self._is_warmed_up = False

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
