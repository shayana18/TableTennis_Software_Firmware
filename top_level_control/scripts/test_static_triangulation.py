"""
Static Ball Triangulation Verification

Verifies stereo triangulation accuracy using a STATIONARY orange ball.
Unlike the normal pipeline (MOG2 background subtraction, needs motion),
this script uses color-based or click-based detection for static objects.

TWO DETECTION MODES:
  HSV mode  (default) — automatic orange color thresholding, hands-free
  Click mode          — manual click near ball, local circle detection

USAGE:
    cd top_level_control
    python scripts/test_static_triangulation.py

CONTROLS:
    q     - Quit
    m     - Toggle between HSV / Click mode
    s     - Save current 3D measurement
    r     - Reset/clear saved measurements
    d     - Toggle HSV mask debug view (HSV mode only)
    t     - Toggle HSV trackbar tuning window (HSV mode only)
    c     - Clear current click points (Click mode only)
    SPACE - Freeze/unfreeze frame
    p     - Print measurement summary

VERIFICATION:
    1. Place orange ball at known distances (30cm, 50cm, 100cm)
    2. HSV mode: confirm auto-detection (green circle), read Z value
    3. Click mode (m to switch): click ball in left view, then right view
    4. Press 's' to save, repeat, check consistency (low std)
    5. Compare reported Z vs tape-measured distance
"""

import cv2
import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings


class StaticTriangulationTester:
    """Stereo triangulation tester for stationary ball verification."""

    # HSV defaults for orange ping pong ball
    HSV_LOW = np.array([5, 80, 80])
    HSV_HIGH = np.array([25, 255, 255])

    # Contour filters (same as BallDetector)
    MIN_AREA = 80
    MAX_AREA = 2000
    MIN_CIRCULARITY = 0.35

    # Click mode ROI size (half-width)
    CLICK_ROI_HALF = 75

    # HoughCircles parameters for small ROI
    HOUGH_DP = 1.2
    HOUGH_MIN_DIST = 50
    HOUGH_PARAM1 = 50
    HOUGH_PARAM2 = 25
    HOUGH_MIN_RADIUS = 5
    HOUGH_MAX_RADIUS = 60

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.calibration_dir = os.path.join(
            self.script_dir, '..', 'camera_calibration', 'camera_parameters')

        cam_settings = load_camera_settings()
        self.frame_width = cam_settings['frame_width']
        self.frame_height = cam_settings['frame_height']

        self.triangulator = None
        self.measurements = []

        # Mode: 'hsv' or 'click'
        self.mode = 'hsv'

        # HSV tuning state
        self.hsv_low = self.HSV_LOW.copy()
        self.hsv_high = self.HSV_HIGH.copy()
        self.show_debug = False
        self.show_trackbars = False

        # Click mode state
        self.click_left = None   # (x, y) raw click
        self.click_right = None
        self.detect_left = None  # refined detection dict
        self.detect_right = None

        # Frame freeze
        self.frozen = False
        self.frozen_left = None
        self.frozen_right = None

    def check_calibration(self):
        """Verify all 4 calibration .dat files exist."""
        required = [
            'camera0_intrinsics.dat', 'camera1_intrinsics.dat',
            'camera0_rot_trans.dat', 'camera1_rot_trans.dat'
        ]
        missing = [f for f in required
                   if not os.path.exists(os.path.join(self.calibration_dir, f))]
        if missing:
            print("\nERROR: Missing calibration files!")
            for f in missing:
                print(f"  - {f}")
            print("\nRun calibration first.")
            return False
        return True

    # ================================================================
    # HSV DETECTION
    # ================================================================

    def detect_ball_hsv(self, frame):
        """
        Detect orange ball via HSV color thresholding.

        Returns:
            (det_dict_or_None, binary_mask)
            det_dict has keys: 'center', 'area', 'circularity'
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # Morphological cleanup (same kernels as BallDetector)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_circ = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.MIN_CIRCULARITY:
                continue

            if circularity > best_circ:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                best = {
                    'center': (cx, cy),
                    'area': area,
                    'circularity': circularity,
                }
                best_circ = circularity

        return best, mask

    # ================================================================
    # CLICK-TO-DETECT
    # ================================================================

    def detect_ball_click(self, frame, click_pt):
        """
        Detect ball near a click point using local circle detection.

        1. Crop 150x150 ROI around click
        2. HoughCircles for precise circle
        3. Fallback: adaptive threshold + contour + minEnclosingCircle

        Returns:
            det_dict_or_None with keys: 'center', 'area', 'radius'
        """
        cx, cy = int(click_pt[0]), int(click_pt[1])
        h, w = frame.shape[:2]
        half = self.CLICK_ROI_HALF

        # Clamp ROI to frame bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try HoughCircles first
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=self.HOUGH_DP,
            minDist=self.HOUGH_MIN_DIST,
            param1=self.HOUGH_PARAM1,
            param2=self.HOUGH_PARAM2,
            minRadius=self.HOUGH_MIN_RADIUS,
            maxRadius=self.HOUGH_MAX_RADIUS
        )

        if circles is not None:
            # Pick circle closest to ROI center
            roi_cx = cx - x1
            roi_cy = cy - y1
            best_dist = float('inf')
            best_circle = None
            for c in circles[0]:
                d = np.hypot(c[0] - roi_cx, c[1] - roi_cy)
                if d < best_dist:
                    best_dist = d
                    best_circle = c
            if best_circle is not None:
                lx = best_circle[0] + x1
                ly = best_circle[1] + y1
                radius = best_circle[2]
                return {
                    'center': (float(lx), float(ly)),
                    'area': np.pi * radius * radius,
                    'radius': float(radius),
                }

        # Fallback: adaptive threshold + contour
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Pick largest contour near click
        roi_cx = cx - x1
        roi_cy = cy - y1
        best = None
        best_dist = float('inf')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            (ccx, ccy), radius = cv2.minEnclosingCircle(cnt)
            d = np.hypot(ccx - roi_cx, ccy - roi_cy)
            if d < best_dist:
                best_dist = d
                best = (ccx + x1, ccy + y1, radius, area)

        if best is not None:
            return {
                'center': (float(best[0]), float(best[1])),
                'area': float(best[3]),
                'radius': float(best[2]),
            }

        return None

    # ================================================================
    # MOUSE CALLBACK
    # ================================================================

    def on_mouse_click(self, event, x, y, flags, param):
        """Handle mouse clicks in click mode."""
        if self.mode != 'click' or event != cv2.EVENT_LBUTTONDOWN:
            return

        # Determine if click is on left or right panel
        if x < self.frame_width:
            # Left panel
            self.click_left = (x, y)
            # Detect from current frame
            frame = self.frozen_left if self.frozen else None
            if frame is None:
                # Will be processed in main loop
                self.detect_left = None
            else:
                self.detect_left = self.detect_ball_click(frame, (x, y))
                if self.detect_left:
                    print(f"  Left: detected at ({self.detect_left['center'][0]:.1f}, "
                          f"{self.detect_left['center'][1]:.1f})")
                else:
                    print("  Left: no circle found near click, using raw click")
                    self.detect_left = {'center': (float(x), float(y)), 'area': 0, 'radius': 0}
        else:
            # Right panel — adjust x to right frame coordinates
            rx = x - self.frame_width
            self.click_right = (rx, y)
            frame = self.frozen_right if self.frozen else None
            if frame is None:
                self.detect_right = None
            else:
                self.detect_right = self.detect_ball_click(frame, (rx, y))
                if self.detect_right:
                    print(f"  Right: detected at ({self.detect_right['center'][0]:.1f}, "
                          f"{self.detect_right['center'][1]:.1f})")
                else:
                    print("  Right: no circle found near click, using raw click")
                    self.detect_right = {'center': (float(rx), float(y)), 'area': 0, 'radius': 0}

    # ================================================================
    # TRACKBAR CALLBACKS
    # ================================================================

    def _create_trackbars(self):
        """Create HSV tuning trackbar window."""
        cv2.namedWindow('HSV Tuning', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('H Low', 'HSV Tuning', self.hsv_low[0], 179, lambda v: None)
        cv2.createTrackbar('S Low', 'HSV Tuning', self.hsv_low[1], 255, lambda v: None)
        cv2.createTrackbar('V Low', 'HSV Tuning', self.hsv_low[2], 255, lambda v: None)
        cv2.createTrackbar('H High', 'HSV Tuning', self.hsv_high[0], 179, lambda v: None)
        cv2.createTrackbar('S High', 'HSV Tuning', self.hsv_high[1], 255, lambda v: None)
        cv2.createTrackbar('V High', 'HSV Tuning', self.hsv_high[2], 255, lambda v: None)

    def _read_trackbars(self):
        """Read current trackbar values into hsv_low/hsv_high."""
        self.hsv_low[0] = cv2.getTrackbarPos('H Low', 'HSV Tuning')
        self.hsv_low[1] = cv2.getTrackbarPos('S Low', 'HSV Tuning')
        self.hsv_low[2] = cv2.getTrackbarPos('V Low', 'HSV Tuning')
        self.hsv_high[0] = cv2.getTrackbarPos('H High', 'HSV Tuning')
        self.hsv_high[1] = cv2.getTrackbarPos('S High', 'HSV Tuning')
        self.hsv_high[2] = cv2.getTrackbarPos('V High', 'HSV Tuning')

    # ================================================================
    # TRIANGULATION HELPERS
    # ================================================================

    def _triangulate_points(self, pt_left, pt_right):
        """
        Undistort, triangulate, validate.

        Returns:
            dict with 'found_3d', 'position_3d', 'disparity', 'reproj_err', 'reject_reason'
        """
        tri = self.triangulator

        ul = tri._rectify_point(pt_left, tri.cmtx0, tri.dist0, tri.R_rect0, tri.P_rect0)
        ur = tri._rectify_point(pt_right, tri.cmtx1, tri.dist1, tri.R_rect1, tri.P_rect1)

        disparity = ul[0] - ur[0]
        point_3d = tri.triangulate(ul, ur)

        is_valid, reject_reason = tri._validate_triangulation(ul, ur, point_3d)

        result = {
            'found_3d': False,
            'position_3d': None,
            'disparity': disparity,
            'reproj_err': None,
            'reject_reason': reject_reason,
        }

        if is_valid:
            result['found_3d'] = True
            result['position_3d'] = tuple(point_3d)
            err_l, err_r = tri._reprojection_error(point_3d, ul, ur)
            result['reproj_err'] = max(err_l, err_r)
            result['reject_reason'] = None

        return result

    # ================================================================
    # DISPLAY HELPERS
    # ================================================================

    def _draw_detection(self, vis, det, color=(0, 255, 0)):
        """Draw a circle + crosshair at detection point."""
        if det is None:
            return
        cx, cy = int(det['center'][0]), int(det['center'][1])
        r = det.get('radius', None)
        if r is None:
            r = max(8, int(np.sqrt(det['area'] / np.pi))) if det['area'] > 0 else 8
        else:
            r = int(r)
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    def _draw_click_crosshair(self, vis, pt, color=(0, 0, 255)):
        """Draw a crosshair at click point."""
        if pt is None:
            return
        x, y = int(pt[0]), int(pt[1])
        size = 15
        cv2.line(vis, (x - size, y), (x + size, y), color, 1)
        cv2.line(vis, (x, y - size), (x, y + size), color, 1)

    def _print_summary(self):
        """Print measurement summary statistics."""
        if not self.measurements:
            print("\n  No measurements saved.")
            return

        xs = [m[0] for m in self.measurements]
        ys = [m[1] for m in self.measurements]
        zs = [m[2] for m in self.measurements]
        ds = [m[3] for m in self.measurements]
        rs = [m[4] for m in self.measurements]

        print(f"\n{'='*60}")
        print(f"  MEASUREMENT SUMMARY  ({len(self.measurements)} samples)")
        print(f"{'='*60}")
        print(f"  {'':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'X':>8} {np.mean(xs):>10.2f} {np.std(xs):>10.2f} {np.min(xs):>10.2f} {np.max(xs):>10.2f}")
        print(f"  {'Y':>8} {np.mean(ys):>10.2f} {np.std(ys):>10.2f} {np.min(ys):>10.2f} {np.max(ys):>10.2f}")
        print(f"  {'Z':>8} {np.mean(zs):>10.2f} {np.std(zs):>10.2f} {np.min(zs):>10.2f} {np.max(zs):>10.2f}")
        print(f"  {'Disp(px)':>8} {np.mean(ds):>10.2f} {np.std(ds):>10.2f} {np.min(ds):>10.2f} {np.max(ds):>10.2f}")
        print(f"  {'Reproj':>8} {np.mean(rs):>10.2f} {np.std(rs):>10.2f} {np.min(rs):>10.2f} {np.max(rs):>10.2f}")
        print(f"{'='*60}\n")

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def run(self):
        """Main loop."""
        if not self.check_calibration():
            return

        print("\n[StaticTriangulation] Initializing...")
        self.triangulator = StereoTriangulator(
            calibration_dir=self.calibration_dir)

        print("[StaticTriangulation] Starting cameras...")
        self.triangulator.start_cameras()

        window_name = 'Static Triangulation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse_click)

        print(f"\n  Mode: HSV AUTO")
        print(f"  Controls: q=quit  m=toggle mode  s=save  r=reset  d=debug  t=trackbars  p=summary")
        print(f"  Click mode: click left panel then right panel, c=clear\n")

        tri_result = None
        frame_count = 0

        try:
            while True:
                # --- Capture ---
                if not self.frozen:
                    grabbed_l = self.triangulator.cap_left.grab()
                    grabbed_r = self.triangulator.cap_right.grab()
                    if not grabbed_l or not grabbed_r:
                        print("  Frame grab failed, retrying...")
                        continue

                    ret_l, frame_left = self.triangulator.cap_left.retrieve()
                    ret_r, frame_right = self.triangulator.cap_right.retrieve()
                    if not ret_l or not ret_r:
                        continue

                    self.frozen_left = frame_left.copy()
                    self.frozen_right = frame_right.copy()
                    frame_count += 1
                else:
                    frame_left = self.frozen_left
                    frame_right = self.frozen_right

                left_vis = frame_left.copy()
                right_vis = frame_right.copy()

                det_left = None
                det_right = None
                mask_left = None
                mask_right = None
                tri_result = None

                # --- Detection ---
                if self.mode == 'hsv':
                    if self.show_trackbars:
                        self._read_trackbars()

                    det_left, mask_left = self.detect_ball_hsv(frame_left)
                    det_right, mask_right = self.detect_ball_hsv(frame_right)

                    self._draw_detection(left_vis, det_left)
                    self._draw_detection(right_vis, det_right)

                    if det_left and det_right:
                        tri_result = self._triangulate_points(
                            det_left['center'], det_right['center'])

                elif self.mode == 'click':
                    # Process pending clicks against current frames
                    if self.click_left and self.detect_left is None:
                        self.detect_left = self.detect_ball_click(frame_left, self.click_left)
                        if self.detect_left:
                            print(f"  Left: detected at ({self.detect_left['center'][0]:.1f}, "
                                  f"{self.detect_left['center'][1]:.1f})")
                        else:
                            print("  Left: no circle found near click, using raw click")
                            self.detect_left = {
                                'center': (float(self.click_left[0]), float(self.click_left[1])),
                                'area': 0, 'radius': 0
                            }

                    if self.click_right and self.detect_right is None:
                        self.detect_right = self.detect_ball_click(frame_right, self.click_right)
                        if self.detect_right:
                            print(f"  Right: detected at ({self.detect_right['center'][0]:.1f}, "
                                  f"{self.detect_right['center'][1]:.1f})")
                        else:
                            print("  Right: no circle found near click, using raw click")
                            self.detect_right = {
                                'center': (float(self.click_right[0]), float(self.click_right[1])),
                                'area': 0, 'radius': 0
                            }

                    # Draw click crosshairs
                    self._draw_click_crosshair(left_vis, self.click_left)
                    self._draw_click_crosshair(right_vis, self.click_right)

                    # Draw refined detections
                    self._draw_detection(left_vis, self.detect_left)
                    self._draw_detection(right_vis, self.detect_right)

                    # Triangulate if both detected
                    if self.detect_left and self.detect_right:
                        tri_result = self._triangulate_points(
                            self.detect_left['center'], self.detect_right['center'])

                    # Click mode prompts
                    if self.click_left is None:
                        cv2.putText(left_vis, "Click LEFT ball", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    elif self.click_right is None:
                        cv2.putText(right_vis, "Click RIGHT ball", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # --- Overlay 3D result ---
                if tri_result and tri_result['found_3d']:
                    X, Y, Z = tri_result['position_3d']
                    cv2.putText(left_vis, f"3D: X={X:.1f} Y={Y:.1f} Z={Z:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(left_vis,
                                f"Disp: {tri_result['disparity']:.1f}px  "
                                f"Reproj: {tri_result['reproj_err']:.1f}px",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    # Terminal output (throttle to avoid spam)
                    if frame_count % 15 == 0 or self.mode == 'click':
                        print(f"  3D: X={X:.1f} Y={Y:.1f} Z={Z:.1f}  "
                              f"disp={tri_result['disparity']:.1f}px  "
                              f"reproj={tri_result['reproj_err']:.1f}px")
                elif tri_result and tri_result['reject_reason']:
                    cv2.putText(left_vis, f"REJECTED: {tri_result['reject_reason']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- Mode indicator ---
                mode_text = "[HSV AUTO]" if self.mode == 'hsv' else "[CLICK MODE]"
                if self.frozen:
                    mode_text += " FROZEN"
                mode_y = 30 if not (tri_result and tri_result['found_3d']) else 90
                cv2.putText(left_vis, mode_text, (10, mode_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                # --- Compose side-by-side view ---
                combined = np.hstack([left_vis, right_vis])
                cv2.imshow(window_name, combined)

                # --- Debug mask view (HSV mode) ---
                if self.show_debug and self.mode == 'hsv' and mask_left is not None:
                    debug = np.hstack([mask_left, mask_right])
                    cv2.imshow('HSV Masks', debug)
                elif not self.show_debug:
                    try:
                        cv2.destroyWindow('HSV Masks')
                    except cv2.error:
                        pass

                # --- Key handling ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('m'):
                    self.mode = 'click' if self.mode == 'hsv' else 'hsv'
                    self.click_left = None
                    self.click_right = None
                    self.detect_left = None
                    self.detect_right = None
                    print(f"\n  Mode: {'HSV AUTO' if self.mode == 'hsv' else 'CLICK'}")

                elif key == ord('s'):
                    if tri_result and tri_result['found_3d']:
                        X, Y, Z = tri_result['position_3d']
                        d = tri_result['disparity']
                        r = tri_result['reproj_err']
                        self.measurements.append((X, Y, Z, d, r))
                        print(f"  SAVED #{len(self.measurements)}: "
                              f"X={X:.1f} Y={Y:.1f} Z={Z:.1f}  "
                              f"disp={d:.1f}px  reproj={r:.1f}px")
                    else:
                        print("  Nothing to save (no valid 3D point)")

                elif key == ord('r'):
                    self.measurements.clear()
                    print("  Measurements cleared")

                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"  Debug view: {'ON' if self.show_debug else 'OFF'}")

                elif key == ord('t'):
                    self.show_trackbars = not self.show_trackbars
                    if self.show_trackbars:
                        self._create_trackbars()
                        print("  Trackbars: ON")
                    else:
                        try:
                            cv2.destroyWindow('HSV Tuning')
                        except cv2.error:
                            pass
                        print("  Trackbars: OFF")

                elif key == ord('c'):
                    self.click_left = None
                    self.click_right = None
                    self.detect_left = None
                    self.detect_right = None
                    print("  Click points cleared")

                elif key == ord(' '):
                    self.frozen = not self.frozen
                    print(f"  {'FROZEN' if self.frozen else 'LIVE'}")

                elif key == ord('p'):
                    self._print_summary()

        except KeyboardInterrupt:
            print("\n  Interrupted.")

        finally:
            self._print_summary()
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()
            print("[StaticTriangulation] Done.")


if __name__ == '__main__':
    tester = StaticTriangulationTester()
    tester.run()
