"""
Test Trajectory Prediction — Camera-Space 3D Validation

Tracks the ball in stereo, triangulates to camera XYZ (cm),
and predicts the trajectory arc using physics (gravity + drag).

GEOMETRY:
    Camera on SIDE of table, looking ACROSS the table width.
    Camera frame: X=along table length, Y=vertical (down=+), Z=depth (across table width)

LAYOUT:
  ┌──────────────────┬──────────────────┐
  │  LEFT CAMERA      │  RIGHT CAMERA    │
  │  + trajectory arc │  + status info   │
  ├─────────────────────────────────────┤
  │       3D TRAJECTORY VIEW (rotatable) │
  └─────────────────────────────────────┘

CONTROLS:
    q - Quit, r - Reset throw, b - Reset background
    ← → ↑ ↓ - Rotate 3D, 1/2/3 - View presets, p - Stats
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
from trajectory.trajectory_predictor import TrajectoryPredictor  # deprecated, camera-frame analysis
from trajectory.workspace import (
    ROBOT_HOME, CM_TO_MM,
)
# Workspace constants - use new names (firmware ellipse with safety margins)
from trajectory.workspace import ELLIPSE_A as ELLIPSE_RADIUS_X
from trajectory.workspace import ELLIPSE_B as ELLIPSE_RADIUS_Y
from trajectory.workspace import Z_MAX as LIMIT_POS_Z
from trajectory.workspace import Z_MIN as LIMIT_NEG_Z
from config.camera_config import load_camera_settings


# ================================================================
# 3D TRAJECTORY VIEW
# ================================================================

class TrajectoryView3D:
    """
    Rotatable 3D view. Camera: X=length, Y=down, Z=width.
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
               robot_x_cam=None, errors=None, throw_count=0,
               in_workspace=None, apex_err=None,
               robot_cmd=None, predictor=None):
        img = np.full((self.h, self.w, 3), self.BG, dtype=np.uint8)

        if not actual and not predicted:
            cv2.putText(img, "3D Trajectory — toss ball to begin",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT, 1)
            self._draw_info(img)
            return img

        self.auto_fit(actual or [], predicted, snapshot)
        self._draw_grid(img, actual)

        # Robot X plane (interception plane along table length)
        if robot_x_cam is not None and actual:
            self._draw_x_plane(img, robot_x_cam, actual)

        # Workspace wireframe (ellipse limits → camera coords)
        if predictor is not None and predictor._R_cam_to_robot is not None:
            self._draw_workspace_ellipse(img, predictor)

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
            aa3 = actual_apex[:3]
            ap = self.project(*aa3)
            if self._ok(*ap):
                cv2.circle(img, ap, 9, self.APEX_CLR, 2)
                cv2.putText(img, f"apex ({aa3[0]:.1f},{aa3[1]:.1f},{aa3[2]:.1f})",
                            (ap[0]+12, ap[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.APEX_CLR, 1)
        if apex:
            ap = self.project(*apex)
            if self._ok(*ap):
                cv2.drawMarker(img, ap, (0,255,255), cv2.MARKER_DIAMOND, 14, 2)
                cv2.putText(img, f"pred ({apex[0]:.1f},{apex[1]:.1f},{apex[2]:.1f})",
                            (ap[0]+12, ap[1]+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,255,255), 1)

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
            self._draw_errors(img, errors, throw_count, apex_err=apex_err)
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

    def _draw_x_plane(self, img, tx, actual):
        arr = np.array([(p[0],p[1],p[2]) for p in actual])
        ymn, ymx = np.min(arr[:,1])-15, np.max(arr[:,1])+15
        zmn, zmx = np.min(arr[:,2])-15, np.max(arr[:,2])+15
        c = [self.project(tx,ymn,zmn), self.project(tx,ymn,zmx),
             self.project(tx,ymx,zmx), self.project(tx,ymx,zmn)]
        for i in range(4):
            self._ln(img, c[i], c[(i+1)%4], (50,50,120), 1)
        lp = self.project(tx, ymn-5, (zmn+zmx)/2)
        if self._ok(*lp):
            cv2.putText(img, f"Robot X={tx:.0f}cm", (lp[0]-30, lp[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80,80,160), 1)

    def _draw_workspace_ellipse(self, img, predictor):
        """Draw robot workspace as an elliptical cylinder in camera coords."""
        ws_clr = (60, 120, 60)
        segments = 48
        bottom_ring = []
        top_ring = []

        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            rx = ELLIPSE_RADIUS_X * math.cos(th)
            ry = ELLIPSE_RADIUS_Y * math.sin(th)

            bcx, bcy, bcz = predictor.robot_to_cam(rx, ry, LIMIT_NEG_Z)
            tcx, tcy, tcz = predictor.robot_to_cam(rx, ry, LIMIT_POS_Z)
            bottom_ring.append(self.project(bcx, bcy, bcz))
            top_ring.append(self.project(tcx, tcy, tcz))

        for i in range(segments):
            j = (i + 1) % segments
            self._ln(img, bottom_ring[i], bottom_ring[j], ws_clr, 1)
            self._ln(img, top_ring[i], top_ring[j], ws_clr, 1)

        for i in range(0, segments, max(1, segments // 8)):
            self._ln(img, bottom_ring[i], top_ring[i], ws_clr, 1)

        label_pt = top_ring[0]
        if self._ok(*label_pt, 50):
            cv2.putText(img, "WS", (label_pt[0] + 5, label_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, ws_clr, 1)

        # Robot home marker
        hx, hy, hz = predictor.robot_to_cam(ROBOT_HOME[0], ROBOT_HOME[1], ROBOT_HOME[2])
        hp = self.project(hx, hy, hz)
        if self._ok(*hp):
            cv2.drawMarker(img, hp, (0, 200, 200), cv2.MARKER_STAR, 10, 1)
            cv2.putText(img, "HOME", (hp[0] + 8, hp[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 200), 1)

    def _draw_axes(self, img):
        ox, oy, ln = 60, self.h-40, 35
        R = self._R()
        for ax, lbl, clr in [
                (np.array([1,0,0]), "X(length)", self.AXIS_X),
                (np.array([0,1,0]), "Y(up)",     self.AXIS_Y),
                (np.array([0,0,1]), "Z(width)",  self.AXIS_Z)]:
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

    def _draw_errors(self, img, e, tc, apex_err=None):
        y = 20
        cv2.putText(img, f"Throw #{tc}", (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 1)
        lines = [f"Mean: {e['mean_err']:.1f} cm",
                 f"Max:  {e['max_err']:.1f} cm",
                 f"Pts: {e['n_points']}/{e['n_actual']}",
                 f"Dur: {e['throw_duration']*1000:.0f}ms"]
        if apex_err is not None:
            lines.append(f"Apex err: {apex_err:.1f} cm")
        for txt in lines:
            y += 17
            cv2.putText(img, txt, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT, 1)

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
        self.snapshot_idx = -1
        self._lost = 0
        self.errors = {}
        self.throw_count = 0
        self.predicted_apex = None
        self.predicted_apex_frame = -1
        self.predicted_apex_time = None
        self.last_robot_cmd = None

    def start(self):
        self.active = True
        self.actual_positions = []
        self.snapshot_traj = []
        self.snapshot_idx = -1
        self._lost = 0
        self.errors = {}
        self.throw_count += 1
        self.predicted_apex = None
        self.predicted_apex_frame = -1
        self.predicted_apex_time = None
        self.last_robot_cmd = None

    def add_actual(self, x, y, z, t=None):
        if t is None: t = time.perf_counter()
        self.actual_positions.append((float(x), float(y), float(z), float(t)))
        self._lost = 0

    def tick_lost(self):
        self._lost += 1

    def should_end(self):
        return self._lost >= self.LOST_THRESHOLD and len(self.actual_positions) > 0

    def snapshot_prediction(self, predictor):
        if self.snapshot_traj: return
        traj = predictor.predict_trajectory(duration=1.0, dt=0.005)
        if not traj: return
        self.snapshot_traj = traj
        self.snapshot_idx = len(self.actual_positions)

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
        self.errors = {
            'valid': True, 'mean_err': float(np.mean(errs)),
            'max_err': float(np.max(errs)), 'median_err': float(np.median(errs)),
            'n_points': len(errs), 'n_actual': len(future),
            'throw_duration': self.actual_positions[-1][3]-self.actual_positions[0][3]}

    def record_apex_prediction(self, predictor):
        """Record the first valid apex prediction (snapshot approach)."""
        if self.predicted_apex is not None:
            return
        traj = predictor.predict_trajectory(duration=1.0, dt=0.005)
        if not traj or len(traj) < 3:
            return
        # Find apex = minimum Y in trajectory (most negative = highest point)
        apex_pt = min(traj, key=lambda p: p[1])
        self.predicted_apex = (apex_pt[0], apex_pt[1], apex_pt[2])
        self.predicted_apex_frame = len(self.actual_positions)
        # Time from current position to apex
        self.predicted_apex_time = apex_pt[3]

    def get_actual_apex(self):
        """Return (x, y, z, t, frame_idx) for highest point, or None."""
        if len(self.actual_positions) < 3:
            return None
        best_idx = 0
        best_y = self.actual_positions[0][1]
        for i, p in enumerate(self.actual_positions):
            if p[1] < best_y:
                best_y = p[1]
                best_idx = i
        p = self.actual_positions[best_idx]
        return (p[0], p[1], p[2], p[3], best_idx)


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
        self._diag_frame = 0
        self._throw_t0 = None
        self._live_apex = None

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

    def draw_status(self, frame, result, robot_cmd=None):
        """Draw camera-space status info on the right panel."""
        y = 20
        ps = self.pred.get_stats()
        vel = self.pred.get_velocity()

        # Current 3D position
        if result['found_3d']:
            x, y3, z = result['position_3d']
            cv2.putText(frame, "Ball (cm):", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            y += 18
            cv2.putText(frame, f" X:{x:7.1f}  Y:{y3:7.1f}  Z:{z:7.1f}",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
        else:
            cv2.putText(frame, "No 3D detection", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

        # Velocity
        y += 22
        if vel['valid']:
            cv2.putText(frame, f"Vel: vx={vel['vx']:+.0f} vy={vel['vy']:+.0f} vz={vel['vz']:+.0f}",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
            y += 16
            cv2.putText(frame, f"Speed: {vel['speed']:.0f} cm/s",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
        else:
            cv2.putText(frame, "Vel: --", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)

        # Robot target position
        y += 22
        if robot_cmd and robot_cmd['valid']:
            rx, ry, rz = robot_cmd['robot_x'], robot_cmd['robot_y'], robot_cmd['robot_z']
            cv2.putText(frame, "Robot (mm):", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,200), 1)
            y += 16
            cv2.putText(frame, f" X:{rx:+6.0f}  Y:{ry:+6.0f}  Z:{rz:+7.0f}",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
            y += 16
            ws_txt = "IN WORKSPACE" if robot_cmd['in_workspace'] else "OUT OF RANGE"
            ws_clr = (0, 255, 0) if robot_cmd['in_workspace'] else (0, 0, 255)
            cv2.putText(frame, ws_txt, (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ws_clr, 1)
            # Time to intercept and strategy
            y += 16
            strat = robot_cmd.get('strategy', '?')
            t_int = robot_cmd['t']
            cv2.putText(frame, f"t={t_int*1000:.0f}ms  [{strat}]",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        else:
            cv2.putText(frame, "Robot: --", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)

        # Buffer / rejection
        y += 22
        total = ps['accepted'] + ps['rejected']
        rej_pct = ps['rejected'] / max(total, 1) * 100
        clr = (0,200,0) if rej_pct < 30 else (0,200,255) if rej_pct < 60 else (0,0,255)
        cv2.putText(frame, f"Buf:{ps['buffer_size']}/{self.pred.buffer_size} "
                    f"Acc:{ps['accepted']} Rej:{ps['rejected']}({rej_pct:.0f}%)",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, clr, 1)

        # Prediction readiness
        y += 18
        if ps['is_ready']:
            cv2.putText(frame, "PREDICTION READY", (10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        else:
            reasons = []
            if ps['buffer_size'] < self.pred.min_points:
                reasons.append(f"need {self.pred.min_points}pts")
            if ps['time_span'] < self.pred.MIN_TIME_SPAN:
                reasons.append(f"need {self.pred.MIN_TIME_SPAN*1000:.0f}ms span")
            if not self.pred._velocity_valid:
                reasons.append("no valid vel")
            cv2.putText(frame, f"NOT READY: {', '.join(reasons)}",
                        (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,100,255), 1)

        # Thresholds reference
        y += 22
        cv2.putText(frame, f"Limits: jump<{self.pred.MAX_POSITION_JUMP:.0f}cm "
                    f"spd<{self.pred.MAX_BALL_SPEED:.0f}cm/s",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120,120,120), 1)

    # ================================================================
    # POST-THROW SUMMARY
    # ================================================================

    def _print_throw_summary(self, throw_rec):
        """Print structured post-throw analysis."""
        n_pts = len(throw_rec.actual_positions)
        if n_pts < 2:
            print(f"\n  [END] Throw #{throw_rec.throw_count}: too few points ({n_pts})")
            return

        dur_s = throw_rec.actual_positions[-1][3] - throw_rec.actual_positions[0][3]
        dur_ms = dur_s * 1000

        vel = self.pred.get_velocity()
        spd = vel['speed'] if vel['valid'] else 0

        bar = "\u2550" * 64
        print(f"\n  {bar}")
        print(f"  THROW #{throw_rec.throw_count} SUMMARY  |  "
              f"{n_pts} pts  |  {dur_ms:.0f}ms  |  {spd:.0f} cm/s")
        print(f"  {bar}")

        # Actual apex
        aa = throw_rec.get_actual_apex()
        if aa:
            ax, ay, az, at, ai = aa
            t_rel = at - throw_rec.actual_positions[0][3]
            print(f"\n  Actual Apex:     X={ax:7.1f}   Y={ay:7.1f}   Z={az:7.1f}"
                  f"   (frame {ai+1}, t={t_rel:.3f}s)")
        else:
            print(f"\n  Actual Apex:     --")

        # Predicted apex
        pa = throw_rec.predicted_apex
        if pa:
            print(f"  Predicted Apex:  X={pa[0]:7.1f}   Y={pa[1]:7.1f}   Z={pa[2]:7.1f}"
                  f"   (at frame {throw_rec.predicted_apex_frame}, "
                  f"t_to={throw_rec.predicted_apex_time*1000:.0f}ms)")
        else:
            print(f"  Predicted Apex:  --")

        # Apex error
        if aa and pa:
            dx = pa[0] - aa[0]
            dy = pa[1] - aa[1]
            dz = pa[2] - aa[2]
            d3 = math.sqrt(dx*dx + dy*dy + dz*dz)
            print(f"  Apex Error:      dX={dx:7.1f}   dY={dy:7.1f}   dZ={dz:7.1f}"
                  f"   3D={d3:5.1f} cm")

        # Prediction accuracy (snapshot)
        e = throw_rec.errors
        if e.get('valid'):
            print(f"\n  Prediction Accuracy (from snapshot onward):")
            print(f"    Mean: {e['mean_err']:.1f} cm   "
                  f"Max: {e['max_err']:.1f} cm   "
                  f"Pts: {e['n_points']}/{e['n_actual']}")

        # Robot intercept
        rc = throw_rec.last_robot_cmd
        if rc and rc['valid']:
            ws = "IN WORKSPACE" if rc['in_workspace'] else "OUT OF RANGE"
            reach = "REACHABLE" if rc.get('reachable') else "TOO FAR/SLOW"
            print(f"\n  Robot Intercept: X={rc['robot_x']:+.0f}  Y={rc['robot_y']:+.0f}  "
                  f"Z={rc['robot_z']:+.0f} mm")
            print(f"                   t={rc['t']*1000:.0f}ms  [{rc['strategy']}]  "
                  f"{ws}  {reach}")
        else:
            print(f"\n  Robot Intercept: --")

        # Final velocity
        if vel['valid']:
            print(f"\n  Final Velocity:  Vx={vel['vx']:+.0f}  Vy={vel['vy']:+.0f}  "
                  f"Vz={vel['vz']:+.0f}  Spd={vel['speed']:.0f} cm/s")
        print(f"  {bar}\n")

    def _print_p_summary(self):
        """Print clean summary table of all recorded throws."""
        if not self.history:
            print("\n  No throws recorded yet.\n")
            return
        bar = "\u2550" * 57
        print(f"\n  {bar}")
        print(f"  SUMMARY  ({len(self.history)} throws)")
        print(f"  {bar}")
        print(f"  {'#':>3}   {'Pts':>3}   {'Dur(ms)':>7}   {'Spd':>4}   "
              f"  {'Apex Err(cm)':>12}   {'Pred Err':>10}")
        for t in self.history:
            n = len(t.actual_positions)
            dur = 0
            if n >= 2:
                dur = (t.actual_positions[-1][3] - t.actual_positions[0][3]) * 1000

            e = t.errors

            # Apex error
            apex_str = "--"
            aa = t.get_actual_apex()
            pa = t.predicted_apex
            if aa and pa:
                dx = pa[0]-aa[0]; dy = pa[1]-aa[1]; dz = pa[2]-aa[2]
                apex_str = f"{math.sqrt(dx*dx + dy*dy + dz*dz):.1f}"

            # Prediction error
            pred_str = "--"
            if e.get('valid'):
                pred_str = f"{e['mean_err']:.1f}/{e['max_err']:.1f}"

            print(f"  {t.throw_count:3d}   {n:3d}   {dur:7.0f}   {'':>4}   "
                  f"  {apex_str:>12}   {pred_str:>10}")
        print(f"  {bar}\n")

    # ================================================================
    # RUN
    # ================================================================

    def run(self):
        print("\n" + "="*60)
        print("  TRAJECTORY — CAMERA-SPACE 3D PREDICTION")
        print("="*60)
        print(f"  Cams: L={self.cam_l} R={self.cam_r}")
        print(f"  Toss ball after warmup — tracking starts immediately")
        print(f"  q r b ←→↑↓ 1/2/3 p")
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

        print("--- Ready! Toss ball to begin tracking ---\n")
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
                live_tr = []
                self._live_apex = None
                has_det = result['left_detection'] is not None or result['right_detection'] is not None

                if result['found_3d']:
                    x, y, z = result['position_3d']
                    if not self.throw.active:
                        self.throw.start()
                        self.pred.reset()
                        self._diag_frame = 0
                        self._throw_t0 = None
                        print(f"\n  [THROW #{self.throw.throw_count}]")
                        print(f"  {'Frame':>6}  {'Time(s)':>8}  {'X(cm)':>8}  {'Y(cm)':>8}  "
                              f"{'Z(cm)':>8}  |  {'Vx':>5}  {'Vy':>5}  {'Vz':>5}  {'Spd':>4}  "
                              f"|  {'RobX':>6} {'RobY':>6} {'RobZ':>7} {'WS':>3}  "
                              f"|  Prediction")
                        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  "
                              f"{'─'*8}  |  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*4}  "
                              f"|  {'─'*6} {'─'*6} {'─'*7} {'─'*3}  "
                              f"|  {'─'*20}")

                    self._diag_frame += 1
                    now = time.perf_counter()

                    accepted = self.pred.add_position(x, y, z)

                    if accepted:
                        # Set t0 on first accepted point
                        if self._throw_t0 is None:
                            self._throw_t0 = now
                        t_rel = now - self._throw_t0

                        vel = self.pred.get_velocity()
                        if vel['valid']:
                            v_str = (f"{vel['vx']:+5.0f}  {vel['vy']:+5.0f}  "
                                     f"{vel['vz']:+5.0f}  {vel['speed']:4.0f}")
                        else:
                            v_str = f"{'--':>5}  {'--':>5}  {'--':>5}  {'--':>4}"

                        # Robot command + prediction column
                        buf_n = len(self.pred.position_buffer)
                        rob_str = f"{'--':>6} {'--':>6} {'--':>7} {'--':>3}"
                        if self.pred.is_ready():
                            cmd = self.pred.get_robot_command()
                            if cmd['valid']:
                                ws = "OK" if cmd['in_workspace'] else "OUT"
                                rob_str = (f"{cmd['robot_x']:+6.0f} {cmd['robot_y']:+6.0f} "
                                           f"{cmd['robot_z']:+7.0f} {ws:>3}")
                            # Try to get apex from trajectory
                            apex_traj = self.pred.predict_trajectory(duration=1.0, dt=0.005)
                            if apex_traj and len(apex_traj) >= 3:
                                ap = min(apex_traj, key=lambda p: p[1])
                                self._live_apex = (ap[0], ap[1], ap[2])
                                pred_str = (f"t={cmd['t']*1000:.0f}ms [{cmd.get('strategy','?')}]"
                                            if cmd['valid'] else f"READY [buf={buf_n}]")
                            else:
                                pred_str = f"READY [buf={buf_n}]"
                        else:
                            pred_str = f"[buf={buf_n}]"

                        print(f"  {self._diag_frame:6d}  {t_rel:8.4f}  {x:8.1f}  {y:8.1f}  "
                              f"{z:8.1f}  |  {v_str}  |  {rob_str}  |  {pred_str}")

                        self.throw.add_actual(x, y, z)
                        # Snapshot early prediction for accuracy comparison
                        if (self.pred.is_ready() and
                                len(self.throw.actual_positions) >= self.SNAP_AFTER):
                            self.throw.snapshot_prediction(self.pred)
                            self.throw.record_apex_prediction(self.pred)
                    else:
                        # Rejected point
                        t_rel = (now - self._throw_t0) if self._throw_t0 else 0.0
                        print(f"  {self._diag_frame:6d}  {t_rel:8.4f}  {x:8.1f}  {y:8.1f}  "
                              f"{z:8.1f}  |  {'--':>5}  {'--':>5}  {'--':>5}  {'--':>4}  "
                              f"|  {'--':>6} {'--':>6} {'--':>7} {'--':>3}  "
                              f"|  REJ: {self.pred._last_reject_reason}")

                elif has_det and self.throw.active:
                    # Ball seen in one/both cameras but stereo failed
                    reason = result.get('reject_reason', 'no_match')
                    self._diag_frame += 1
                    t_rel = (time.perf_counter() - self._throw_t0) if self._throw_t0 else 0.0
                    print(f"  {self._diag_frame:6d}  {t_rel:8.4f}  {'---':>8}  {'---':>8}  "
                          f"{'---':>8}  |  {'':>5}  {'':>5}  {'':>5}  {'':>4}  "
                          f"|  {'':>6} {'':>6} {'':>7} {'':>3}  "
                          f"|  STEREO FAIL: {reason}")

                if not result['found_3d']:
                    if self.throw.active:
                        self.throw.tick_lost()
                        if self.throw.should_end():
                            self.throw.end()
                            self.frozen = self.throw
                            self.history.append(self.throw)
                            cnt = self.throw.throw_count
                            self._print_throw_summary(self.frozen)
                            self.throw = ThrowRecorder()
                            self.throw.throw_count = cnt

                # Live trajectory prediction + apex + robot command
                robot_cmd = None
                if self.pred.is_ready() and self.throw.active:
                    live_tr = self.pred.predict_trajectory(duration=0.8, dt=0.005)
                    robot_cmd = self.pred.get_robot_command()
                    if robot_cmd and robot_cmd['valid']:
                        self.throw.last_robot_cmd = robot_cmd
                    if self._live_apex is None and live_tr and len(live_tr) >= 3:
                        ap = min(live_tr, key=lambda p: p[1])
                        self._live_apex = (ap[0], ap[1], ap[2])

                # Pick throw to display
                if self.throw.active and self.throw.actual_positions:
                    st, s_tr = self.throw, live_tr
                elif self.frozen:
                    st = self.frozen
                    s_tr = self.frozen.snapshot_traj
                else:
                    st, s_tr = self.throw, []

                # ---- CAMERA PANELS ----
                lv, rv = self.tri.draw_results(result)
                if live_tr and self.throw.active:
                    self.draw_traj_cam(lv, live_tr)
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

                lv = cv2.resize(lv, (self.CAM_W, self.CAM_H))
                rv = cv2.resize(rv, (self.CAM_W, self.CAM_H))
                self.draw_status(rv, result, robot_cmd=robot_cmd)
                cv2.putText(rv, "q r b arrows 1/2/3 p",
                            (10,self.CAM_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120,120,120), 1)
                cam_row = np.hstack([lv, rv])

                # ---- 3D VIEW ----
                apt = aa = None
                apex_err_3d = None
                if st and len(st.actual_positions) >= 5:
                    aa_full = st.get_actual_apex()
                    if aa_full:
                        aa = aa_full[:3]

                # Predicted apex: live during throw, stored after
                if self.throw.active and self._live_apex:
                    apt = self._live_apex
                elif st and st.predicted_apex:
                    apt = st.predicted_apex

                # Compute apex error for overlay
                if aa and apt:
                    dx = apt[0]-aa[0]; dy = apt[1]-aa[1]; dz = apt[2]-aa[2]
                    apex_err_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

                se, stc = None, 0
                if self.frozen and not self.throw.active and self.frozen.errors.get('valid'):
                    se = self.frozen.errors; stc = self.frozen.throw_count

                # Intercept point for 3D view
                intercept_pt = None
                intercept_ws = None
                if robot_cmd and robot_cmd['valid']:
                    intercept_pt = (robot_cmd['cam_x'], robot_cmd['cam_y'], robot_cmd['cam_z'])
                    intercept_ws = robot_cmd['in_workspace']

                v3 = self.v3d.render(
                    actual=st.actual_positions if st else [],
                    predicted=s_tr if s_tr else [],
                    snapshot=st.snapshot_traj if st and not self.throw.active else [],
                    intercept=intercept_pt,
                    apex=apt, actual_apex=aa,
                    robot_x_cam=self.pred.robot_x_cam if self.pred else None,
                    errors=se, throw_count=stc,
                    in_workspace=intercept_ws,
                    apex_err=apex_err_3d,
                    robot_cmd=robot_cmd,
                    predictor=self.pred)

                if cam_row.shape[1] != v3.shape[1]:
                    v3 = cv2.resize(v3, (cam_row.shape[1], v3.shape[0]))
                cv2.imshow('Trajectory Validation', np.vstack([cam_row, v3]))

                # ---- KEYS ----
                key = cv2.waitKeyEx(1) & 0xFFFFFF
                if key == ord('q'): break
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
                    self._print_p_summary()
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
            self._print_p_summary()
        print("\nDone!")

def main():
    TrajectoryTester().run()

if __name__ == '__main__':
    main()
