"""
Test Trajectory Prediction — Hawk-Eye 3D Validation

GEOMETRY:
    Camera on SIDE of table, looking ALONG the table length.

         Camera Z (depth) = along table length (ball travel axis)
         ───────────────────────────────────→

    ┌───────────────────────────────────────┐
    │           1 (Robot endline)            │
    │                                       │  Camera X
    │ ─ ─ ─ ─ ─ ─ NET ─ ─ ─ ─ ─ ─ ─ ─ ─  │  (across
    │                                       │   width)
    │           R (Player)                  │
    └───────────────────────────────────────┘
    📷 Camera here, looking along Z

LAYOUT:
  ┌──────────────────┬──────────────────┐
  │  LEFT CAMERA      │  RIGHT CAMERA    │
  │  + trajectory arc │  + ROBOT CMD     │
  ├─────────────────────────────────────┤
  │       3D TRAJECTORY VIEW (rotatable) │
  └─────────────────────────────────────┘

CALIBRATION (press 'c' twice):
  1. Place ball at ROBOT ENDLINE (center of width) → press 'c'
     → captures camera Z (robot endline), X (center width), Y (table surface)
  2. Done! (single point is enough for basic calibration)

CONTROLS:
    q - Quit, c - Calibrate, r - Reset, b - Reset background
    ← → ↑ ↓ - Rotate 3D, 1/2/3 - View presets, p - Stats, SPACE - Skip warmup
"""

import cv2
import sys
import os
import time
import math
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor
from config.camera_config import load_camera_settings


# ================================================================
# 3D TRAJECTORY VIEW
# ================================================================

class TrajectoryView3D:
    """
    Rotatable 3D view. Camera: X=width, Y=down, Z=length.
    View flips Y so up appears up on screen.
    """

    BG          = (20, 20, 28)
    GRID        = (45, 45, 55)
    ACTUAL      = (0, 230, 0)
    ACTUAL_LINE = (0, 160, 0)
    PRED        = (240, 180, 40)
    PRED_GLOW   = (180, 120, 25)
    INTERCEPT   = (0, 0, 255)
    APEX_CLR    = (0, 220, 255)
    SNAP        = (130, 80, 35)
    AXIS_X      = (80, 80, 255)
    AXIS_Y      = (80, 220, 80)
    AXIS_Z      = (255, 140, 80)
    TEXT        = (180, 180, 190)
    TEXT_DIM    = (100, 100, 110)

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.azimuth = math.radians(-35)
        self.elevation = math.radians(25)
        self.center = np.array([0.0, 0.0, 0.0])
        self.scale = 2.5

    def _R(self):
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        Ry = np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
        Rx = np.array([[1,0,0],[0,ce,-se],[0,se,ce]])
        return Rx @ Ry

    def project(self, x, yc, z):
        y = -yc
        pt = np.array([x-self.center[0], y+self.center[1], z-self.center[2]])
        rv = self._R() @ pt
        return (int(self.w/2 + rv[0]*self.scale), int(self.h/2 - rv[1]*self.scale))

    def project_batch(self, pts):
        if not pts: return []
        R = self._R()
        cx, cy, cz = self.center
        hw, hh, s = self.w/2, self.h/2, self.scale
        out = []
        for p in pts:
            v = np.array([p[0]-cx, -p[1]+cy, p[2]-cz])
            rv = R @ v
            out.append((int(hw + rv[0]*s), int(hh - rv[1]*s)))
        return out

    def auto_fit(self, actual, predicted=None, snapshot=None, pad=1.3):
        all_p = []
        for s in [actual, predicted, snapshot]:
            if s:
                for p in s: all_p.append((p[0], p[1], p[2]))
        if len(all_p) < 2: return
        arr = np.array(all_p)
        self.center = arr.mean(axis=0)
        self.center[1] = arr[:,1].mean()  # keep as cam Y
        rng = arr.max(0) - arr.min(0)
        mr = max(rng[0], rng[1], rng[2], 10.0)
        self.scale = np.clip(min(self.w, self.h)*0.6 / (mr*pad), 0.5, 15.0)

    def _ok(self, px, py, m=5):
        return -m <= px < self.w+m and -m <= py < self.h+m

    def _ln(self, img, p1, p2, c, t=1):
        if self._ok(*p1, 200) or self._ok(*p2, 200):
            cv2.line(img, p1, p2, c, t, cv2.LINE_AA)

    def render(self, actual=None, predicted=None, snapshot=None,
               intercept=None, apex=None, actual_apex=None,
               robot_z_cam=None, errors=None, throw_count=0,
               in_workspace=None):
        img = np.full((self.h, self.w, 3), self.BG, dtype=np.uint8)

        if not actual and not predicted:
            cv2.putText(img, "3D Trajectory — toss ball to begin",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT, 1)
            self._draw_info(img)
            return img

        self.auto_fit(actual or [], predicted, snapshot)
        self._draw_grid(img, actual)

        # Robot Z plane
        if robot_z_cam is not None and actual:
            self._draw_z_plane(img, robot_z_cam, actual)

        self._draw_axes(img)

        # Snapshot (dim)
        if snapshot and len(snapshot) > 1:
            p2 = self.project_batch(snapshot)
            for i in range(len(p2)-1):
                self._ln(img, p2[i], p2[i+1], self.SNAP, 1)

        # Predicted (bright)
        if predicted and len(predicted) > 1:
            p2 = self.project_batch(predicted)
            for i in range(len(p2)-1):
                if self._ok(*p2[i], 50):
                    self._ln(img, p2[i], p2[i+1], self.PRED_GLOW, 4)
            for i in range(len(p2)-1):
                if self._ok(*p2[i], 50):
                    self._ln(img, p2[i], p2[i+1], self.PRED, 2)
            # Time ticks
            step = max(1, len(predicted)//8)
            for i in range(0, len(predicted), step):
                if i < len(p2) and self._ok(*p2[i]):
                    cv2.circle(img, p2[i], 3, self.PRED, -1)
                    if i % max(1, len(predicted)//4) == 0 and i > 0:
                        cv2.putText(img, f"{predicted[i][3]*1000:.0f}ms",
                                    (p2[i][0]+6, p2[i][1]-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.PRED, 1)

        # Actual (green, fading)
        if actual:
            p2 = self.project_batch(actual)
            n = len(p2)
            for i in range(n-1):
                if self._ok(*p2[i]) or self._ok(*p2[i+1]):
                    cv2.line(img, p2[i], p2[i+1], self.ACTUAL_LINE, 1, cv2.LINE_AA)
            for i in range(n):
                if not self._ok(*p2[i]): continue
                b = 0.4 + 0.6*(i/max(n-1,1))
                c = tuple(int(v*b) for v in self.ACTUAL)
                cv2.circle(img, p2[i], 6 if i==n-1 else 3, c, -1)
            if n > 0 and self._ok(*p2[-1]):
                cv2.circle(img, p2[-1], 8, self.ACTUAL, 2)

        # Apex markers
        if actual_apex:
            ap = self.project(*actual_apex)
            if self._ok(*ap):
                cv2.circle(img, ap, 9, self.APEX_CLR, 2)
                cv2.putText(img, "apex", (ap[0]+12, ap[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.APEX_CLR, 1)
        if apex:
            ap = self.project(*apex)
            if self._ok(*ap):
                cv2.drawMarker(img, ap, (0,255,255), cv2.MARKER_DIAMOND, 14, 2)

        # Intercept (red X)
        if intercept:
            ip = self.project(*intercept)
            if self._ok(*ip):
                s = 14
                cv2.line(img, (ip[0]-s,ip[1]-s), (ip[0]+s,ip[1]+s), self.INTERCEPT, 3)
                cv2.line(img, (ip[0]-s,ip[1]+s), (ip[0]+s,ip[1]-s), self.INTERCEPT, 3)
                cv2.circle(img, ip, s+6, self.INTERCEPT, 2)
                lbl = "intercept"
                if in_workspace is not None:
                    lbl += " OK" if in_workspace else " OUT"
                    lc = (0,255,0) if in_workspace else (0,0,255)
                else:
                    lc = self.INTERCEPT
                cv2.putText(img, lbl, (ip[0]+s+8, ip[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1)

        self._draw_info(img)
        self._draw_legend(img)
        if errors and errors.get('valid'):
            self._draw_errors(img, errors, throw_count)
        return img

    def _draw_grid(self, img, actual):
        if not actual or len(actual) < 2: return
        arr = np.array([(p[0],p[1],p[2]) for p in actual])
        yg = np.max(arr[:,1]) + 5
        xmn, xmx = np.min(arr[:,0])-20, np.max(arr[:,0])+20
        zmn, zmx = max(0, np.min(arr[:,2])-20), np.max(arr[:,2])+20
        sp = 20
        for x in range(int(xmn/sp)*sp, int(xmx)+sp, sp):
            self._ln(img, self.project(x,yg,zmn), self.project(x,yg,zmx), self.GRID)
        for z in range(int(zmn/sp)*sp, int(zmx)+sp, sp):
            self._ln(img, self.project(xmn,yg,z), self.project(xmx,yg,z), self.GRID)

    def _draw_z_plane(self, img, tz, actual):
        arr = np.array([(p[0],p[1],p[2]) for p in actual])
        ymn, ymx = np.min(arr[:,1])-15, np.max(arr[:,1])+15
        xmn, xmx = np.min(arr[:,0])-15, np.max(arr[:,0])+15
        c = [self.project(xmn,ymn,tz), self.project(xmx,ymn,tz),
             self.project(xmx,ymx,tz), self.project(xmn,ymx,tz)]
        for i in range(4):
            self._ln(img, c[i], c[(i+1)%4], (50,50,120), 1)
        lp = self.project((xmn+xmx)/2, ymn-5, tz)
        if self._ok(*lp):
            cv2.putText(img, f"Robot Z={tz:.0f}cm", (lp[0]-30, lp[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,80,160), 1)

    def _draw_axes(self, img):
        ox, oy, ln = 60, self.h-40, 35
        R = self._R()
        for ax, lbl, clr in [
                (np.array([1,0,0]), "X(width)", self.AXIS_X),
                (np.array([0,1,0]), "Y(up)",    self.AXIS_Y),
                (np.array([0,0,1]), "Z(length)", self.AXIS_Z)]:
            d = R @ ax
            e = (int(ox+d[0]*ln), int(oy-d[1]*ln))
            cv2.arrowedLine(img, (ox,oy), e, clr, 2, tipLength=0.25)
            cv2.putText(img, lbl, (e[0]+4, e[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, clr, 1)

    def _draw_info(self, img):
        az, el = math.degrees(self.azimuth), math.degrees(self.elevation)
        cv2.putText(img, f"az={az:.0f} el={el:.0f}  Arrows:rotate 1/2/3:preset",
                    (self.w-280, self.h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.TEXT_DIM, 1)

    def _draw_legend(self, img):
        x0, y = self.w-120, 18
        cv2.circle(img, (x0,y), 4, self.ACTUAL, -1)
        cv2.putText(img, "actual", (x0+10,y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.ACTUAL, 1)
        y += 16
        cv2.line(img, (x0-5,y), (x0+5,y), self.PRED, 2)
        cv2.putText(img, "predicted", (x0+10,y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.PRED, 1)
        y += 16
        cv2.line(img, (x0-3,y-3), (x0+3,y+3), self.INTERCEPT, 2)
        cv2.line(img, (x0-3,y+3), (x0+3,y-3), self.INTERCEPT, 2)
        cv2.putText(img, "intercept", (x0+10,y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.INTERCEPT, 1)

    def _draw_errors(self, img, e, tc):
        y = 20
        cv2.putText(img, f"Throw #{tc}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 1)
        for txt in [f"Mean: {e['mean_err']:.1f} cm",
                     f"Max:  {e['max_err']:.1f} cm",
                     f"Pts: {e['n_points']}/{e['n_actual']}",
                     f"Dur: {e['throw_duration']*1000:.0f}ms"]:
            y += 17
            cv2.putText(img, txt, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT, 1)
        if e.get('final_intercept_err') is not None:
            y += 17
            cv2.putText(img, f"Final: {e['final_intercept_err']:.1f} cm",
                        (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,200,255), 1)

    def rotate(self, d_az=0, d_el=0):
        self.azimuth += math.radians(d_az)
        self.elevation = np.clip(self.elevation + math.radians(d_el),
                                  math.radians(-85), math.radians(85))

    def set_preset(self, n):
        if n == 1: self.azimuth, self.elevation = math.radians(-35), math.radians(25)
        elif n == 2: self.azimuth, self.elevation = math.radians(0), math.radians(0)
        elif n == 3: self.azimuth, self.elevation = math.radians(0), math.radians(85)


# ================================================================
# THROW RECORDER
# ================================================================

class ThrowRecorder:
    LOST_THRESHOLD = 20

    def __init__(self):
        self.active = False
        self.actual_positions = []
        self.snapshot_traj = []
        self.snapshot_pred = None
        self.snapshot_idx = -1
        self.final_pred = None
        self._lost = 0
        self.errors = {}
        self.throw_count = 0

    def start(self):
        self.active = True
        self.actual_positions = []
        self.snapshot_traj = []
        self.snapshot_pred = None
        self.snapshot_idx = -1
        self.final_pred = None
        self._lost = 0
        self.errors = {}
        self.throw_count += 1

    def add_actual(self, x, y, z, t=None):
        if t is None: t = time.perf_counter()
        self.actual_positions.append((float(x), float(y), float(z), float(t)))
        self._lost = 0

    def tick_lost(self):
        self._lost += 1

    def should_end(self):
        return self._lost >= self.LOST_THRESHOLD and len(self.actual_positions) > 0

    def snapshot_prediction(self, predictor, target_z):
        if self.snapshot_pred is not None: return
        traj = predictor.predict_trajectory(duration=1.0, dt=0.005)
        pred = predictor.predict(target_z=target_z)
        if not traj or not pred['valid']: return
        self.snapshot_traj = traj
        self.snapshot_pred = pred
        self.snapshot_idx = len(self.actual_positions)

    def update_final(self, predictor, target_z):
        pred = predictor.predict(target_z=target_z)
        if pred['valid']: self.final_pred = pred

    def end(self):
        self.active = False
        self._lost = 0
        self.compute_errors()

    def compute_errors(self):
        if not self.snapshot_traj or len(self.actual_positions) < 2:
            self.errors = {'valid': False}; return
        if self.snapshot_idx < 0 or self.snapshot_idx >= len(self.actual_positions):
            self.errors = {'valid': False}; return
        t_snap = self.actual_positions[self.snapshot_idx][3]
        future = [(x,y,z,t) for x,y,z,t in self.actual_positions if t >= t_snap]
        if len(future) < 2: self.errors = {'valid': False}; return
        pred_abs = [(x,y,z, t_snap+dt) for x,y,z,dt in self.snapshot_traj]
        errs = []
        for ax,ay,az,at in future:
            bd, bp = float('inf'), None
            for px,py,pz,pt in pred_abs:
                td = abs(at-pt)
                if td < bd: bd, bp = td, (px,py,pz)
            if bp and bd < 0.05:
                errs.append(math.sqrt((ax-bp[0])**2+(ay-bp[1])**2+(az-bp[2])**2))
        if not errs: self.errors = {'valid': False}; return
        fe = None
        if self.final_pred and self.final_pred['valid']:
            l = self.actual_positions[-1]
            fe = math.sqrt((self.final_pred['intercept_x']-l[0])**2 +
                           (self.final_pred['intercept_y']-l[1])**2 +
                           (self.final_pred['intercept_z']-l[2])**2)
        self.errors = {
            'valid': True, 'mean_err': float(np.mean(errs)),
            'max_err': float(np.max(errs)), 'median_err': float(np.median(errs)),
            'n_points': len(errs), 'n_actual': len(future),
            'throw_duration': self.actual_positions[-1][3]-self.actual_positions[0][3],
            'final_intercept_err': fe}

    def get_actual_apex(self):
        if len(self.actual_positions) < 3: return None
        p = min(self.actual_positions, key=lambda p: p[1])
        return (p[0], p[1], p[2])


# ================================================================
# MAIN TESTER
# ================================================================

class TrajectoryTester:
    CAM_W, CAM_H = 480, 300
    V3D_W, V3D_H = 960, 360
    SNAP_AFTER = 6

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        self.calib_dir = os.path.join(
            self.base_dir, 'camera_calibration', 'camera_parameters')

        cam = load_camera_settings()
        self.fw = cam['frame_width']
        self.fh = cam['frame_height']
        self.cam_l = cam['camera0']
        self.cam_r = cam['camera1']

        self.tri = None
        self.pred = None
        self.throw = ThrowRecorder()
        self.v3d = TrajectoryView3D(self.V3D_W, self.V3D_H)
        self.frozen = None
        self.history = []
        self.calibrated = False

    def check_calib(self):
        req = ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
               'camera0_rot_trans.dat', 'camera1_rot_trans.dat']
        miss = [f for f in req if not os.path.exists(os.path.join(self.calib_dir, f))]
        if miss: print("Missing:", miss); return False
        return True

    def warmup(self):
        print("\n  Remove ball. Learning background (SPACE=skip)...")
        t0, dur = time.time(), 2.0
        while time.time()-t0 < dur:
            if not self.tri.cap_left.grab(): continue
            if not self.tri.cap_right.grab(): continue
            _, fl = self.tri.cap_left.retrieve()
            _, fr = self.tri.cap_right.retrieve()
            if fl is None or fr is None: continue
            self.tri.build_background(fl, fr)
            p = min((time.time()-t0)/dur, 1.0)
            dl = cv2.resize(fl, (self.CAM_W, self.CAM_H))
            bw = int(p*(self.CAM_W-40))
            cv2.rectangle(dl, (20,self.CAM_H-30), (20+bw,self.CAM_H-15), (0,255,255), -1)
            cv2.putText(dl, f"BG: {p*100:.0f}%", (20,self.CAM_H-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            cv2.imshow('Trajectory Validation', dl)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '): break
            elif k == ord('q'): return False
        print("  Ready!\n")
        return True

    def _click_point(self, window_name, frame, label):
        """
        Show a frame and let user click a point. Returns (x, y) pixel coords.

        Draws crosshair at cursor, click confirms. ESC cancels.
        """
        click_result = {'point': None, 'done': False}
        cursor = [0, 0]

        def on_mouse(event, x, y, flags, param):
            cursor[0], cursor[1] = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                click_result['point'] = (x, y)
                click_result['done'] = True

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

        while not click_result['done']:
            display = frame.copy()
            cx, cy = cursor
            # Crosshair at cursor
            cv2.line(display, (cx-20, cy), (cx+20, cy), (0,255,255), 1)
            cv2.line(display, (cx, cy-20), (cx, cy+20), (0,255,255), 1)
            cv2.circle(display, (cx, cy), 5, (0,255,255), 1)
            cv2.putText(display, f"({cx}, {cy})", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            # Instructions
            cv2.putText(display, f"Click: {label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(display, "ESC to cancel", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None

        # Draw confirmed click
        pt = click_result['point']
        confirm = frame.copy()
        cv2.drawMarker(confirm, pt, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(confirm, f"Confirmed: ({pt[0]}, {pt[1]})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow(window_name, confirm)
        cv2.waitKey(500)
        cv2.destroyWindow(window_name)
        return pt

    def do_calibrate(self):
        """
        Click-to-calibrate: user clicks the same physical point in both views.

        Recommended: click the center of the ball placed at the robot endline,
        on the table surface, at the center of the table width.

        Flow:
          1. Grab fresh frames from both cameras
          2. Show LEFT camera → user clicks the reference point
          3. Show RIGHT camera → user clicks the same physical point
          4. Triangulate the two pixel coords → 3D position
          5. Set as calibration reference
        """
        print("\n" + "="*55)
        print("  CALIBRATION MODE")
        print("="*55)
        print("  Place a ball (or marker) at the ROBOT ENDLINE,")
        print("  centered on the table width, ON the table surface.")
        print("  Then click its center in both camera views.")
        print("="*55)

        # Grab fresh frames
        for _ in range(5):  # flush stale frames
            self.tri.cap_left.grab()
            self.tri.cap_right.grab()

        ret_l, frame_l = self.tri.cap_left.read()
        ret_r, frame_r = self.tri.cap_right.read()

        if not ret_l or not ret_r or frame_l is None or frame_r is None:
            print("  ERROR: Could not grab frames!")
            return

        # Step 1: Click on LEFT camera
        print("\n  → Click the reference point in the LEFT camera")
        pt_left = self._click_point("LEFT Camera — click reference point",
                                     frame_l, "Robot endline center (LEFT)")
        if pt_left is None:
            print("  Cancelled."); return

        # Step 2: Click on RIGHT camera
        print(f"  → Left click: ({pt_left[0]}, {pt_left[1]})")
        print("  → Now click the SAME point in the RIGHT camera")
        pt_right = self._click_point("RIGHT Camera — click SAME point",
                                      frame_r, "Same point (RIGHT)")
        if pt_right is None:
            print("  Cancelled."); return

        print(f"  → Right click: ({pt_right[0]}, {pt_right[1]})")

        # Step 3: Triangulate
        point_3d = self.tri.triangulate(pt_left, pt_right)
        x, y, z = float(point_3d[0]), float(point_3d[1]), float(point_3d[2])

        # Sanity check
        err_l, err_r = self.tri._reprojection_error(point_3d, pt_left, pt_right)
        reproj_err = (err_l + err_r) / 2.0

        print(f"\n  Triangulated 3D position:")
        print(f"    Camera X = {x:.2f} cm  (across table width)")
        print(f"    Camera Y = {y:.2f} cm  (vertical, down=+)")
        print(f"    Camera Z = {z:.2f} cm  (along table length = depth)")
        print(f"    Reprojection error: {reproj_err:.2f} px")

        if reproj_err > 10.0:
            print(f"\n  WARNING: High reprojection error ({reproj_err:.1f}px)!")
            print(f"  Clicks might not correspond to the same physical point.")
            print(f"  Calibration saved anyway — redo with 'c' if needed.\n")

        # Step 4: Set calibration
        #   Z = robot endline (along table length = camera depth)
        #   X = table center width (lateral)
        #   Y = table surface (vertical)
        self.pred.set_robot_endline(z)
        self.pred.set_table_calibration(x, y)
        self.calibrated = True

        print(f"\n  ✓ CALIBRATED!")
        print(f"    Robot endline:  Z = {z:.2f} cm")
        print(f"    Table center:   X = {x:.2f} cm")
        print(f"    Table surface:  Y = {y:.2f} cm")
        print(f"\n  Now toss the ball — prediction is active!\n")

    # --- Overlays ---

    def draw_traj_cam(self, frame, traj, color=(220,160,50), t=2):
        h, w = frame.shape[:2]
        pts = []
        for x,y,z,dt in traj:
            uv = self.tri.project_to_image((x,y,z), camera='left')
            if uv is None: continue
            u, v = int(uv[0]), int(uv[1])
            if 0<=u<w and 0<=v<h: pts.append((u,v))
        for i in range(len(pts)-1):
            cv2.line(frame, pts[i], pts[i+1], color, t, cv2.LINE_AA)
        for i in range(0, len(pts), max(1, len(pts)//15)):
            if i < len(pts): cv2.circle(frame, pts[i], 3, color, -1)

    def draw_intercept_cam(self, frame, pr, color=(0,0,255)):
        if not pr['valid']: return
        pt = (pr['intercept_x'], pr['intercept_y'], pr['intercept_z'])
        uv = self.tri.project_to_image(pt, camera='left')
        if uv is None: return
        h, w = frame.shape[:2]
        u, v = int(uv[0]), int(uv[1])
        if not (0<=u<w and 0<=v<h): return
        s = 12
        cv2.line(frame, (u-s,v-s), (u+s,v+s), color, 3)
        cv2.line(frame, (u-s,v+s), (u+s,v-s), color, 3)
        cv2.circle(frame, (u,v), s+5, color, 2)

    def draw_cmd(self, frame, cmd):
        if not cmd['valid']:
            cv2.putText(frame, "ROBOT: waiting", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
            if not self.calibrated:
                cv2.putText(frame, "Press 'c' to calibrate", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)
            return
        c = cmd['confidence']
        clr = (0,255,0) if c>0.6 else (0,255,255) if c>0.3 else (0,100,255)
        st = (cmd['strategy'] or '?').upper()
        ws = cmd.get('in_workspace')
        ws_c = (0,200,0) if ws else (0,0,200)
        y = 20
        cv2.putText(frame, f"ROBOT [{st}]", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        y += 18
        cv2.putText(frame, "Cam (cm):", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140,140,140), 1)
        y += 15
        cv2.putText(frame, f" X:{cmd['cam_x']:6.1f} Y:{cmd['cam_y']:6.1f} Z:{cmd['cam_z']:6.1f}",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        y += 18
        cv2.putText(frame, "Robot (mm):", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140,140,140), 1)
        y += 15
        cv2.putText(frame, f" X:{cmd['robot_x']:5.0f} Y:{cmd['robot_y']:5.0f} Z:{cmd['robot_z']:5.0f}",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        y += 20
        cv2.putText(frame, f"T: {cmd['t']*1000:.0f}ms", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,200,255), 1)
        cv2.putText(frame, f"[{'IN' if ws else 'OUT'}]", (110,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, ws_c, 2)
        y += 18
        cv2.putText(frame, f"Conf:{c:.0%} Pts:{cmd['buffer_points']}",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1)

    # ================================================================
    # RUN
    # ================================================================

    def run(self):
        print("\n" + "="*60)
        print("  TRAJECTORY — 3D HAWK-EYE DEMO")
        print("="*60)
        print(f"  Cams: L={self.cam_l} R={self.cam_r}")
        print(f"  1) Warmup  2) Place ball at robot endline → 'c'")
        print(f"  3) Toss ball → watch prediction!")
        print(f"  q c r b ←→↑↓ 1/2/3 p")
        print("="*60)

        if not self.check_calib(): return
        try:
            self.tri = StereoTriangulator(
                calibration_dir=self.calib_dir,
                cam_left_id=self.cam_l, cam_right_id=self.cam_r)
        except Exception as e:
            print(f"ERROR: {e}"); return
        try:
            self.tri.start_cameras(self.fw, self.fh)
        except RuntimeError as e:
            print(f"ERROR: {e}"); return

        if not self.warmup():
            self.tri.stop_cameras(); cv2.destroyAllWindows(); return

        self.pred = TrajectoryPredictor(
            buffer_size=15, min_points=4, velocity_method='regression',
            gravity=981.0, y_down=True, enable_drag=True)

        print("--- Place ball at ROBOT ENDLINE and press 'c' ---\n")
        fps_n, fps_t, fps = 0, time.perf_counter(), 0

        try:
            while True:
                result = self.tri.update()
                if result['left_frame'] is None: continue
                fps_n += 1
                if fps_n % 30 == 0:
                    fps = 30.0/(time.perf_counter()-fps_t)
                    fps_t = time.perf_counter()

                # --- Throw tracking ---
                cmd = {'valid': False}
                live_tr, live_pr = [], {'valid': False}

                if self.calibrated and result['found_3d']:
                    x, y, z = result['position_3d']
                    if not self.throw.active:
                        self.throw.start()
                        self.pred.reset()
                        print(f"\n  [THROW #{self.throw.throw_count}]")

                    accepted = self.pred.add_position(x, y, z)
                    if accepted:
                        self.throw.add_actual(x, y, z)
                    if self.pred.is_ready() and len(self.throw.actual_positions) >= self.SNAP_AFTER:
                        self.throw.snapshot_prediction(self.pred, self.pred.robot_z_cam)
                    if self.pred.is_ready():
                        self.throw.update_final(self.pred, self.pred.robot_z_cam)

                elif self.calibrated:
                    if self.throw.active:
                        self.throw.tick_lost()
                        if self.throw.should_end():
                            self.throw.end()
                            self.frozen = self.throw
                            self.history.append(self.throw)
                            cnt = self.throw.throw_count
                            self.throw = ThrowRecorder()
                            self.throw.throw_count = cnt
                            e = self.frozen.errors
                            s = self.pred.get_stats()
                            print(f"\n  [END] {len(self.frozen.actual_positions)}pts "
                                  f"(rej {s['rejected']})")
                            if e.get('valid'):
                                fi = e.get('final_intercept_err')
                                print(f"    err: mean={e['mean_err']:.1f} max={e['max_err']:.1f}"
                                      + (f" final={fi:.1f}" if fi else ""))

                if self.calibrated:
                    cmd = self.pred.get_robot_command()
                    if self.pred.is_ready() and self.throw.active:
                        live_tr = self.pred.predict_trajectory(duration=0.8, dt=0.005)
                        live_pr = self.pred.predict()

                # Pick throw to display
                if self.throw.active and self.throw.actual_positions:
                    st, s_tr, s_pr = self.throw, live_tr, live_pr
                elif self.frozen:
                    st = self.frozen
                    s_tr = self.frozen.snapshot_traj
                    s_pr = self.frozen.snapshot_pred or {'valid': False}
                else:
                    st, s_tr, s_pr = self.throw, [], {'valid': False}

                # ---- CAMERA PANELS ----
                lv, rv = self.tri.draw_results(result)
                if live_tr and self.throw.active:
                    self.draw_traj_cam(lv, live_tr)
                    self.draw_intercept_cam(lv, live_pr)
                elif self.frozen and self.frozen.snapshot_traj:
                    self.draw_traj_cam(lv, self.frozen.snapshot_traj, (150,100,40))

                if result['found_3d']:
                    x,y,z = result['position_3d']
                    cv2.putText(lv, f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
                                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

                wu = self.tri.warmup_status()
                if not wu['ready']:
                    cv2.putText(lv, f"Warmup {wu['progress']*100:.0f}%",
                                (10,lv.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                ps = self.pred.get_stats()
                cv2.putText(lv, f"FPS:{fps:.0f} Buf:{ps['buffer_size']} Rej:{ps['rejected']}",
                            (10,lv.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180,180,180), 1)
                vel = self.pred.get_velocity()
                if vel['valid'] and self.throw.active:
                    cv2.putText(lv, f"Spd:{vel['speed']:.0f}cm/s",
                                (10,lv.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,0), 1)
                if not self.calibrated:
                    cv2.putText(lv, "NOT CALIBRATED — press 'c'",
                                (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

                lv = cv2.resize(lv, (self.CAM_W, self.CAM_H))
                rv = cv2.resize(rv, (self.CAM_W, self.CAM_H))
                self.draw_cmd(rv, cmd)
                cv2.putText(rv, "q c r b arrows 1/2/3 p",
                            (10,self.CAM_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120,120,120), 1)
                cam_row = np.hstack([lv, rv])

                # ---- 3D VIEW ----
                ipt = apt = aa = None
                iws = None
                if s_pr and s_pr.get('valid'):
                    ipt = (s_pr['intercept_x'], s_pr['intercept_y'], s_pr['intercept_z'])
                    if s_pr.get('strategy') == 'apex': apt = ipt
                if cmd.get('valid'): iws = cmd.get('in_workspace')
                if st and len(st.actual_positions) >= 5: aa = st.get_actual_apex()

                se, stc = None, 0
                if self.frozen and not self.throw.active and self.frozen.errors.get('valid'):
                    se = self.frozen.errors; stc = self.frozen.throw_count

                v3 = self.v3d.render(
                    actual=st.actual_positions if st else [],
                    predicted=s_tr if s_tr else [],
                    snapshot=st.snapshot_traj if st and not self.throw.active else [],
                    intercept=ipt, apex=apt, actual_apex=aa,
                    robot_z_cam=self.pred.robot_z_cam,
                    errors=se, throw_count=stc, in_workspace=iws)

                if cam_row.shape[1] != v3.shape[1]:
                    v3 = cv2.resize(v3, (cam_row.shape[1], v3.shape[0]))
                cv2.imshow('Trajectory Validation', np.vstack([cam_row, v3]))

                if cmd.get('valid') and self.throw.active:
                    ws = "IN" if cmd.get('in_workspace') else "OUT"
                    print(f"\r  CMD: cam({cmd['cam_x']:5.1f},{cmd['cam_y']:5.1f},{cmd['cam_z']:5.1f}) "
                          f"robot({cmd['robot_x']:5.0f},{cmd['robot_y']:5.0f},{cmd['robot_z']:5.0f})mm "
                          f"T={cmd['t']*1000:4.0f}ms [{ws}]   ", end='')

                # ---- KEYS ----
                key = cv2.waitKeyEx(1) & 0xFFFFFF
                if key == ord('q'): break
                elif key == ord('c'): self.do_calibrate()
                elif key == ord('r'):
                    self.pred.reset()
                    if self.throw.active: self.throw.end()
                    c = self.throw.throw_count
                    self.throw = ThrowRecorder(); self.throw.throw_count = c
                    print("\n  [RESET]")
                elif key == ord('b'):
                    self.tri.reset_background(); self.pred.reset()
                    if self.throw.active: self.throw.end()
                    self.throw = ThrowRecorder()
                    print("\n  [BG RESET]"); self.warmup()
                elif key == ord('p'):
                    print(f"\n  {self.pred.get_stats()}")
                    print(f"  Calibrated: {self.calibrated}")
                    for t in self.history[-5:]:
                        e = t.errors
                        if e.get('valid'):
                            print(f"    #{t.throw_count}: mean={e['mean_err']:.1f} max={e['max_err']:.1f}")
                    print()
                elif key in (65361, 2424832): self.v3d.rotate(d_az=-10)
                elif key in (65363, 2555904): self.v3d.rotate(d_az=10)
                elif key in (65362, 2490368): self.v3d.rotate(d_el=8)
                elif key in (65364, 2621440): self.v3d.rotate(d_el=-8)
                elif key == ord('1'): self.v3d.set_preset(1)
                elif key == ord('2'): self.v3d.set_preset(2)
                elif key == ord('3'): self.v3d.set_preset(3)

        except KeyboardInterrupt: pass
        finally:
            self.tri.stop_cameras(); cv2.destroyAllWindows()

        if self.history:
            print(f"\n{'='*55}")
            print("  SUMMARY")
            print(f"{'='*55}")
            vld = [t for t in self.history if t.errors.get('valid')]
            print(f"  Throws:{len(self.history)} Valid:{len(vld)}")
            if vld:
                m = [t.errors['mean_err'] for t in vld]
                print(f"  Avg: {np.mean(m):.1f} cm (std {np.std(m):.1f})")
                for t in vld:
                    e = t.errors; fi = e.get('final_intercept_err')
                    print(f"    #{t.throw_count}: mean={e['mean_err']:5.1f} max={e['max_err']:5.1f}"
                          + (f" final={fi:.1f}" if fi else ""))
            print(f"{'='*55}")
        print("\nDone!")

def main():
    TrajectoryTester().run()

if __name__ == '__main__':
    main()