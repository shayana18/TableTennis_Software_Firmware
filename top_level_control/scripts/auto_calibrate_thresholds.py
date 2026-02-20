"""
Auto-Calibrate Ball Detection Thresholds
==========================================
Automatically finds optimal HSV + LAB thresholds for stereo ball detection.

WORKFLOW:
    1. Script starts → opens both cameras
    2. Hold ball steady → click on it in the LEFT camera
    3. Click on it in the RIGHT camera
    4. Auto-optimization runs (~3-5 seconds)
    5. Review live detection → press 's' to save
    6. Thresholds saved to config/ball_thresholds_stereo.json

CONTROLS:
    s       - Save thresholds
    r       - Retry (click again)
    q       - Quit

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG
"""

import cv2
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.ball_tracker import EnhancedBallTracker
from config.camera_config import (
    load_camera_settings, configure_camera,
    CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT
)


# ======================================================================
#  Stereo Camera Capture (self-contained, same as other scripts)
# ======================================================================

class ArducamStereoCapture:
    """Stereo camera capture for Arducam OV9782."""

    def __init__(self, cam_left_id=None, cam_right_id=None):
        self.cam_left_id = cam_left_id if cam_left_id is not None else CAMERA_LEFT_ID
        self.cam_right_id = cam_right_id if cam_right_id is not None else CAMERA_RIGHT_ID
        self.cap_left = None
        self.cap_right = None

    def start_cameras(self, width=None, height=None):
        if width is None:
            width = FRAME_WIDTH
        if height is None:
            height = FRAME_HEIGHT
        self.cap_left = cv2.VideoCapture(self.cam_left_id)
        self.cap_right = cv2.VideoCapture(self.cam_right_id)

        if not self.cap_left.isOpened():
            raise RuntimeError(f"Failed to open left camera (ID: {self.cam_left_id})")
        if not self.cap_right.isOpened():
            raise RuntimeError(f"Failed to open right camera (ID: {self.cam_right_id})")

        s_l = configure_camera(self.cap_left, width, height)
        s_r = configure_camera(self.cap_right, width, height)
        print(f"  LEFT:  {s_l['width']}x{s_l['height']} @ {s_l['fps']:.0f}fps ({s_l['fourcc']})")
        print(f"  RIGHT: {s_r['width']}x{s_r['height']} @ {s_r['fps']:.0f}fps ({s_r['fourcc']})")

    def read(self):
        ret_l, frame_l = self.cap_left.read()
        ret_r, frame_r = self.cap_right.read()
        return ret_l, frame_l, ret_r, frame_r

    def stop_cameras(self):
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()


# ======================================================================
#  Color Sampler
# ======================================================================

class ColorSampler:
    """Samples HSV and LAB color statistics from a pixel patch around a click."""

    PATCH_RADIUS = 12  # 25x25 patch

    @staticmethod
    def sample(frame, click_x, click_y, tracker):
        """
        Sample color stats from a patch around the click point.

        Args:
            frame: Raw BGR frame from camera
            click_x, click_y: Pixel coords of user click
            tracker: EnhancedBallTracker (used for CLAHE preprocessing)

        Returns:
            dict with hsv_min/max, lab_min/max, hsv_median, lab_median
            or None if click is out of bounds.
        """
        h, w = frame.shape[:2]
        r = ColorSampler.PATCH_RADIUS

        y1 = max(0, click_y - r)
        y2 = min(h, click_y + r + 1)
        x1 = max(0, click_x - r)
        x2 = min(w, click_x + r + 1)

        if (y2 - y1) < 5 or (x2 - x1) < 5:
            return None

        # Apply CLAHE — must match what the tracker does internally
        frame_norm = tracker.preprocess(frame)
        patch = frame_norm[y1:y2, x1:x2]

        # Convert to HSV and LAB
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

        # Flatten to (N, 3)
        hsv_flat = patch_hsv.reshape(-1, 3).astype(np.float32)
        lab_flat = patch_lab.reshape(-1, 3).astype(np.float32)

        # Per-channel percentiles (5th/95th to reject outlier edge pixels)
        hsv_min = [int(np.percentile(hsv_flat[:, i], 5)) for i in range(3)]
        hsv_max = [int(np.percentile(hsv_flat[:, i], 95)) for i in range(3)]
        lab_min = [int(np.percentile(lab_flat[:, i], 5)) for i in range(3)]
        lab_max = [int(np.percentile(lab_flat[:, i], 95)) for i in range(3)]

        hsv_median = [int(np.median(hsv_flat[:, i])) for i in range(3)]
        lab_median = [int(np.median(lab_flat[:, i])) for i in range(3)]

        return {
            'hsv_min': hsv_min, 'hsv_max': hsv_max, 'hsv_median': hsv_median,
            'lab_min': lab_min, 'lab_max': lab_max, 'lab_median': lab_median,
            'patch_bgr': patch.copy()
        }


# ======================================================================
#  Threshold Optimizer
# ======================================================================

class ThresholdOptimizer:
    """
    Finds the tightest HSV+LAB thresholds that reliably detect the ball.

    Strategy: start with the exact sampled color range (zero margin),
    then expand only the minimum needed per channel.
    """

    MAX_MARGIN = 25         # Never expand beyond this
    MAX_NOISE = 50          # Target: fewer than 50 stray pixels

    # Upper bounds per channel: HSV H is 0-179, rest 0-255
    HSV_UB = [179, 255, 255]
    LAB_UB = [255, 255, 255]

    def __init__(self, tracker):
        self.tracker = tracker

    def _apply(self, hsv_lo, hsv_hi, lab_lo, lab_hi):
        """Apply thresholds to tracker."""
        self.tracker.set_hsv_thresholds(hsv_lo, hsv_hi)
        self.tracker.set_lab_thresholds(lab_lo, lab_hi)

    def _evaluate(self, frame):
        """Detect on one frame → (found, noise_pixels, num_contours)."""
        result = self.tracker.detect(frame)
        if not result['found']:
            return False, 0, 0
        total_white = cv2.countNonZero(result['mask'])
        ball_area = cv2.contourArea(result['contour'])
        noise = max(0, total_white - int(ball_area))
        # Count stray blobs
        contours, _ = cv2.findContours(
            result['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return True, noise, len(contours)

    def _test(self, frame_generator, n=3):
        """Test detection across n frames. Returns (found, avg_noise, avg_blobs)."""
        found_count = 0
        total_noise = 0
        total_blobs = 0
        for _ in range(n):
            frame = frame_generator()
            if frame is None:
                continue
            found, noise, blobs = self._evaluate(frame)
            if found:
                found_count += 1
                total_noise += noise
                total_blobs += blobs
        if found_count < 2:
            return False, 0, 0
        return True, total_noise // found_count, total_blobs // found_count

    def optimize(self, sample_stats, frame_generator, progress_callback=None):
        """
        Find tightest thresholds via expand-from-tight strategy.

        Phase 1: Find minimum uniform margin where ball is detected
        Phase 2: Per-channel squeeze — tighten each of 12 bounds independently
        Phase 3: Stability check

        Returns dict with hsv_lower/upper, lab_lower/upper, success, final_noise.
        """
        # Phase 1: Expand from 0 until ball is detected
        best_margin = self.MAX_MARGIN
        for margin in range(0, self.MAX_MARGIN + 1):
            hsv_lo, hsv_hi = self._make_bounds(
                sample_stats['hsv_min'], sample_stats['hsv_max'],
                margin, self.HSV_UB)
            lab_lo, lab_hi = self._make_bounds(
                sample_stats['lab_min'], sample_stats['lab_max'],
                margin, self.LAB_UB)

            self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
            found, noise, blobs = self._test(frame_generator)

            if progress_callback:
                progress_callback(0, margin, noise, found)

            if found:
                best_margin = margin
                break

        # Add a small safety buffer (+2) so we don't lose detection during squeeze
        safe_margin = min(best_margin + 2, self.MAX_MARGIN)
        hsv_lo, hsv_hi = self._make_bounds(
            sample_stats['hsv_min'], sample_stats['hsv_max'],
            safe_margin, self.HSV_UB)
        lab_lo, lab_hi = self._make_bounds(
            sample_stats['lab_min'], sample_stats['lab_max'],
            safe_margin, self.LAB_UB)

        # Phase 2: Per-channel squeeze — tighten each bound independently
        hsv_lo, hsv_hi, lab_lo, lab_hi = self._squeeze_per_channel(
            hsv_lo, hsv_hi, lab_lo, lab_hi, frame_generator, progress_callback)

        # Apply final
        self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)

        # Phase 3: Stability check (5 consecutive frames)
        stable = True
        worst_noise = 0
        for _ in range(5):
            frame = frame_generator()
            if frame is None:
                stable = False
                break
            found, noise, _ = self._evaluate(frame)
            if not found:
                stable = False
                break
            worst_noise = max(worst_noise, noise)

        return {
            'hsv_lower': hsv_lo, 'hsv_upper': hsv_hi,
            'lab_lower': lab_lo, 'lab_upper': lab_hi,
            'success': stable, 'final_noise': worst_noise
        }

    def _make_bounds(self, ch_min, ch_max, margin, upper_bounds):
        """Create lower/upper arrays from sampled range + margin."""
        lo = [max(0, ch_min[i] - margin) for i in range(3)]
        hi = [min(upper_bounds[i], ch_max[i] + margin) for i in range(3)]
        return lo, hi

    def _squeeze_per_channel(self, hsv_lo, hsv_hi, lab_lo, lab_hi,
                              frame_generator, progress_callback):
        """
        Independently tighten each of the 12 bounds (6 lower + 6 upper).
        For each bound, binary search the tightest value that still detects.
        """
        hsv_lo = list(hsv_lo)
        hsv_hi = list(hsv_hi)
        lab_lo = list(lab_lo)
        lab_hi = list(lab_hi)

        # --- Squeeze HSV lower bounds UP (tighter = higher lower bound) ---
        for ch in range(3):
            original = hsv_lo[ch]
            low, high = original, self.HSV_UB[ch]
            # Don't push lower above upper
            high = min(high, hsv_hi[ch] - 1)
            best = original
            while low <= high:
                mid = (low + high) // 2
                hsv_lo[ch] = mid
                self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
                found, noise, _ = self._test(frame_generator, n=2)
                if found:
                    best = mid
                    low = mid + 1  # Try even tighter
                else:
                    high = mid - 1  # Too tight, back off
            hsv_lo[ch] = best

        # --- Squeeze HSV upper bounds DOWN ---
        for ch in range(3):
            original = hsv_hi[ch]
            low, high = hsv_lo[ch] + 1, original
            best = original
            while low <= high:
                mid = (low + high) // 2
                hsv_hi[ch] = mid
                self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
                found, noise, _ = self._test(frame_generator, n=2)
                if found:
                    best = mid
                    high = mid - 1  # Try even tighter
                else:
                    low = mid + 1  # Too tight
            hsv_hi[ch] = best

        # --- Squeeze LAB lower bounds UP ---
        for ch in range(3):
            original = lab_lo[ch]
            low, high = original, min(255, lab_hi[ch] - 1)
            best = original
            while low <= high:
                mid = (low + high) // 2
                lab_lo[ch] = mid
                self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
                found, noise, _ = self._test(frame_generator, n=2)
                if found:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            lab_lo[ch] = best

        # --- Squeeze LAB upper bounds DOWN ---
        for ch in range(3):
            original = lab_hi[ch]
            low, high = lab_lo[ch] + 1, original
            best = original
            while low <= high:
                mid = (low + high) // 2
                lab_hi[ch] = mid
                self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
                found, noise, _ = self._test(frame_generator, n=2)
                if found:
                    best = mid
                    high = mid - 1
                else:
                    low = mid + 1
            lab_hi[ch] = best

        if progress_callback:
            self._apply(hsv_lo, hsv_hi, lab_lo, lab_hi)
            frame = frame_generator()
            if frame:
                _, noise, _ = self._evaluate(frame)
                progress_callback(1, 0, noise, True)

        return hsv_lo, hsv_hi, lab_lo, lab_hi


# ======================================================================
#  Auto Calibrator (main orchestrator)
# ======================================================================

class AutoCalibrator:
    """
    Full auto-calibration pipeline for stereo ball detection thresholds.

    Phases:
        0 - Click ball in LEFT camera
        1 - Click ball in RIGHT camera
        2 - Auto-optimizing
        3 - Verify and save
    """

    PHASE_CLICK_LEFT = 0
    PHASE_CLICK_RIGHT = 1
    PHASE_OPTIMIZING = 2
    PHASE_VERIFY = 3

    DISPLAY_WIDTH = 640

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.thresholds_path = os.path.join(
            self.script_dir, '..', 'config', 'ball_thresholds_stereo.json')

        cam_settings = load_camera_settings()
        self.cam_left_id = cam_settings['camera0']
        self.cam_right_id = cam_settings['camera1']
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']
        self.display_height = int(self.DISPLAY_WIDTH * self.frame_height / self.frame_width)

        self.capture = None
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()

        # State
        self.phase = self.PHASE_CLICK_LEFT
        self.mouse_pos = (0, 0)
        self.click_left = None
        self.click_right = None
        self.sample_left = None
        self.sample_right = None

        self.frame_left = None
        self.frame_right = None

        self.thresholds = {'left': None, 'right': None}
        self.opt_status = ""  # Status text during optimization

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback for the combined display window."""
        dw = self.DISPLAY_WIDTH
        dh = self.display_height
        scale_x = self.frame_width / dw
        scale_y = self.frame_height / dh

        self.mouse_pos = (x, y)

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Which camera half was clicked?
        if x < dw:
            cam = 'left'
            local_x = x
        else:
            cam = 'right'
            local_x = x - dw
        local_y = y

        real_x = int(local_x * scale_x)
        real_y = int(local_y * scale_y)

        if self.phase == self.PHASE_CLICK_LEFT and cam == 'left':
            if self.frame_left is not None:
                sample = ColorSampler.sample(
                    self.frame_left, real_x, real_y, self.tracker_left)
                if sample is not None:
                    self.sample_left = sample
                    self.click_left = (real_x, real_y)
                    self.phase = self.PHASE_CLICK_RIGHT
                    print(f"  [LEFT] Sampled at ({real_x}, {real_y})")
                    print(f"    HSV median: {sample['hsv_median']}")
                    print(f"    LAB median: {sample['lab_median']}")

        elif self.phase == self.PHASE_CLICK_RIGHT and cam == 'right':
            if self.frame_right is not None:
                sample = ColorSampler.sample(
                    self.frame_right, real_x, real_y, self.tracker_right)
                if sample is not None:
                    self.sample_right = sample
                    self.click_right = (real_x, real_y)
                    self.phase = self.PHASE_OPTIMIZING
                    print(f"  [RIGHT] Sampled at ({real_x}, {real_y})")
                    print(f"    HSV median: {sample['hsv_median']}")
                    print(f"    LAB median: {sample['lab_median']}")

    def _grab_frame(self, camera):
        """Read fresh frames from both cameras, return requested one."""
        ret_l, frame_l, ret_r, frame_r = self.capture.read()
        if ret_l:
            self.frame_left = frame_l
        if ret_r:
            self.frame_right = frame_r
        cv2.waitKey(1)
        if camera == 'left':
            return self.frame_left
        return self.frame_right

    def _optimize_both(self):
        """Run auto-optimization for both cameras."""
        # Left camera
        print("\n--- Optimizing LEFT camera ---")
        self.opt_status = "Optimizing LEFT..."
        optimizer = ThresholdOptimizer(self.tracker_left)
        result_l = optimizer.optimize(
            self.sample_left,
            frame_generator=lambda: self._grab_frame('left'),
            progress_callback=lambda it, m, n, f:
                self._update_progress(f"LEFT: margin={m} noise={n} {'OK' if f else 'LOST'}")
        )
        self.thresholds['left'] = {
            'hsv_lower': result_l['hsv_lower'],
            'hsv_upper': result_l['hsv_upper'],
            'lab_lower': result_l['lab_lower'],
            'lab_upper': result_l['lab_upper']
        }
        status_l = "OK" if result_l['success'] else "UNSTABLE"
        print(f"  Result: {status_l}, noise={result_l['final_noise']}")
        print(f"  HSV: {result_l['hsv_lower']} → {result_l['hsv_upper']}")
        print(f"  LAB: {result_l['lab_lower']} → {result_l['lab_upper']}")

        # Right camera
        print("\n--- Optimizing RIGHT camera ---")
        self.opt_status = "Optimizing RIGHT..."
        optimizer = ThresholdOptimizer(self.tracker_right)
        result_r = optimizer.optimize(
            self.sample_right,
            frame_generator=lambda: self._grab_frame('right'),
            progress_callback=lambda it, m, n, f:
                self._update_progress(f"RIGHT: margin={m} noise={n} {'OK' if f else 'LOST'}")
        )
        self.thresholds['right'] = {
            'hsv_lower': result_r['hsv_lower'],
            'hsv_upper': result_r['hsv_upper'],
            'lab_lower': result_r['lab_lower'],
            'lab_upper': result_r['lab_upper']
        }
        status_r = "OK" if result_r['success'] else "UNSTABLE"
        print(f"  Result: {status_r}, noise={result_r['final_noise']}")
        print(f"  HSV: {result_r['hsv_lower']} → {result_r['hsv_upper']}")
        print(f"  LAB: {result_r['lab_lower']} → {result_r['lab_upper']}")

        self.phase = self.PHASE_VERIFY
        print("\n--- Optimization complete! Review detection. ---")
        print("  Press 's' to save, 'r' to retry, 'q' to quit.")

    def _update_progress(self, text):
        """Update optimization progress on screen."""
        self.opt_status = text
        # Keep GUI alive during optimization
        if self.frame_left is not None and self.frame_right is not None:
            self._display_frames()

    def _display_frames(self):
        """Resize and show current frames with phase overlay."""
        dw = self.DISPLAY_WIDTH
        dh = self.display_height
        scale_x = dw / self.frame_width
        scale_y = dh / self.frame_height

        left_vis = cv2.resize(self.frame_left, (dw, dh))
        right_vis = cv2.resize(self.frame_right, (dw, dh))

        if self.phase == self.PHASE_CLICK_LEFT:
            # Dim right camera
            right_vis = (right_vis * 0.4).astype(np.uint8)
            # Crosshair on left
            mx, my = self.mouse_pos
            if 0 <= mx < dw:
                cv2.line(left_vis, (mx, 0), (mx, dh), (0, 255, 0), 1)
                cv2.line(left_vis, (0, my), (dw, my), (0, 255, 0), 1)
            # Instruction
            cv2.putText(left_vis, "Hold ball steady. Click on ball.",
                        (10, dh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(left_vis, "LEFT CAMERA", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif self.phase == self.PHASE_CLICK_RIGHT:
            # Show click point on left
            if self.click_left:
                cx = int(self.click_left[0] * scale_x)
                cy = int(self.click_left[1] * scale_y)
                cv2.circle(left_vis, (cx, cy), 15, (0, 255, 0), 2)
                cv2.circle(left_vis, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(left_vis, "LEFT: sampled", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Crosshair on right
            mx, my = self.mouse_pos
            rx = mx - dw  # local x in right frame
            if 0 <= rx < dw:
                cv2.line(right_vis, (rx, 0), (rx, dh), (0, 255, 0), 1)
                cv2.line(right_vis, (0, my), (dw, my), (0, 255, 0), 1)
            cv2.putText(right_vis, "Now click on ball here.",
                        (10, dh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(right_vis, "RIGHT CAMERA", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif self.phase == self.PHASE_OPTIMIZING:
            cv2.putText(left_vis, self.opt_status, (10, dh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(right_vis, "Please wait...", (10, dh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        elif self.phase == self.PHASE_VERIFY:
            # Run detection and show results
            det_l = self.tracker_left.detect(self.frame_left)
            det_r = self.tracker_right.detect(self.frame_right)

            for det, vis, label in [(det_l, left_vis, "LEFT"),
                                     (det_r, right_vis, "RIGHT")]:
                if det['found']:
                    cx = int(det['center'][0] * scale_x)
                    cy = int(det['center'][1] * scale_y)
                    r = int(det['radius'] * scale_x)
                    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)

                    # Noise stat
                    total_px = cv2.countNonZero(det['mask'])
                    ball_px = int(cv2.contourArea(det['contour']))
                    noise = total_px - ball_px
                    cv2.putText(vis, f"{label}: OK (noise={noise}px)",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, f"{label}: NOT DETECTED",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(left_vis, "s=save  r=retry  q=quit",
                        (10, dh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = cv2.hconcat([left_vis, right_vis])
        cv2.imshow('Auto Calibrate', combined)

    def _save_thresholds(self):
        """Save optimized thresholds to stereo JSON file."""
        os.makedirs(os.path.dirname(self.thresholds_path), exist_ok=True)
        with open(self.thresholds_path, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        print(f"\n[SAVED] Thresholds saved to {self.thresholds_path}")

    def _reset(self):
        """Reset to click phase."""
        self.phase = self.PHASE_CLICK_LEFT
        self.click_left = None
        self.click_right = None
        self.sample_left = None
        self.sample_right = None
        self.thresholds = {'left': None, 'right': None}
        # Reset trackers to defaults
        self.tracker_left = EnhancedBallTracker()
        self.tracker_right = EnhancedBallTracker()
        print("\n[RETRY] Click on ball again.")

    def run(self):
        print("\n" + "=" * 60)
        print("AUTO-CALIBRATE BALL DETECTION THRESHOLDS")
        print("=" * 60)
        print(f"\nCameras: Left=ID{self.cam_left_id}, Right=ID{self.cam_right_id}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"\nOutput: {self.thresholds_path}")
        print(f"\nHold the ball steady in view of both cameras.")
        print(f"Click on the ball in the LEFT camera to begin.")
        print("=" * 60)

        self.capture = ArducamStereoCapture(self.cam_left_id, self.cam_right_id)
        try:
            self.capture.start_cameras()
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return

        cv2.namedWindow('Auto Calibrate')
        cv2.setMouseCallback('Auto Calibrate', self._on_mouse)

        # Warm-up: discard first 10 frames
        for _ in range(10):
            self.capture.read()

        try:
            while True:
                # Read fresh frames
                ret_l, frame_l, ret_r, frame_r = self.capture.read()
                if not ret_l or not ret_r:
                    continue

                self.frame_left = frame_l
                self.frame_right = frame_r

                # Run optimization (blocking but keeps GUI alive internally)
                if self.phase == self.PHASE_OPTIMIZING:
                    self._optimize_both()
                    continue

                # Display
                self._display_frames()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and self.phase == self.PHASE_VERIFY:
                    self._save_thresholds()
                elif key == ord('r'):
                    self._reset()

        except KeyboardInterrupt:
            pass
        finally:
            self.capture.stop_cameras()
            cv2.destroyAllWindows()

        print("\nDone!")


def main():
    AutoCalibrator().run()


if __name__ == '__main__':
    main()
