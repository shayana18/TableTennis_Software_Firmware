"""
Test Velocity Validation — Post-hoc Analysis of Real-time Velocity Estimates

Records stereo-triangulated throws, then compares the real-time velocity
estimates (from TrajectoryPredictor's regression) against ground-truth
curve fits computed post-hoc over ALL collected points.

GROUND TRUTH (physics):
    X(t) = linear   → slope = true Vx
    Y(t) = quadratic → Vy0 + gravity  (measured_g should be ~981 cm/s²)
    Z(t) = linear   → slope = true Vz

LAYOUT:
  ┌──────────────────┬──────────────────┐
  │  LEFT CAMERA      │  RIGHT CAMERA    │
  ├──────────────────────────────────────┤
  │  ANALYSIS CHART  (post-throw only)   │
  └──────────────────────────────────────┘

CONTROLS:
    q - Quit, r - Reset throw, b - Reset background
    SPACE - Skip warmup, p - Print all-throws summary
"""

import cv2
import sys
import os
import time
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracking.stereo_triangulator import StereoTriangulator
from trajectory.trajectory_predictor import TrajectoryPredictor
from trajectory.physics_model import PhysicsModel
from config.camera_config import load_camera_settings


# ================================================================
# THROW ANALYZER — pure math, no UI
# ================================================================

class ThrowAnalyzer:
    """
    Post-hoc analysis of a recorded throw.

    Given a list of (x, y, z, t) positions, fits physics curves
    and compares against real-time velocity estimates.
    """

    GRAVITY_EXPECTED = 981.0  # cm/s²
    CAMERA_PITCH_DEG = 20.0   # camera pitched 20° down
    # Gravity decomposed into camera frame
    G_Y_EXPECTED = 981.0 * math.cos(math.radians(20.0))   # ~921.5 cm/s²
    G_Z_EXPECTED = -981.0 * math.sin(math.radians(20.0))  # ~-335.5 cm/s²

    def __init__(self, positions, rt_velocity=None, rt_corrected_velocity=None,
                 predictor_was_ready=False, rt_snap_time=None):
        """
        Args:
            positions: list of (x, y, z, t) tuples
            rt_velocity: dict from TrajectoryPredictor.get_velocity() at snapshot time
            rt_corrected_velocity: (vx, vy, vz) from _get_corrected_velocity()
            predictor_was_ready: whether predictor had reached is_ready()
            rt_snap_time: relative time of RT snapshot (seconds from throw start)
        """
        self.positions = positions
        self.rt_velocity = rt_velocity
        self.rt_corrected = rt_corrected_velocity
        self.predictor_was_ready = predictor_was_ready
        self.rt_snap_time = rt_snap_time

    def analyze(self):
        """Run full post-hoc analysis. Returns results dict."""
        n = len(self.positions)
        if n < 4:
            return {'valid': False, 'reason': 'insufficient data', 'n_points': n}

        arr = np.array(self.positions)
        t = arr[:, 3] - arr[0, 3]  # relative time
        duration = t[-1]

        if duration < 0.05:
            return {'valid': False, 'reason': 'throw too short',
                    'n_points': n, 'duration': duration}

        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

        # Bounce segmentation — per-arc analysis
        arcs, bounce_indices = self._segment_arcs(t, x, y, z)
        arc_results = [self._analyze_arc(arc) for arc in arcs]

        # Find best arc (highest Y R², longest if tied)
        best_arc = None
        if arc_results:
            best_arc = max(arc_results,
                           key=lambda a: (a['fit_y']['r_squared'], a['n_points']))

        # Post-hoc curve fits (whole trajectory — kept for backward compat)
        fit_x = self._fit_linear(t, x)
        fit_y = self._fit_quadratic(t, y)
        fit_z = self._fit_quadratic(t, z)  # Z is quadratic due to camera pitch

        # Post-hoc velocities (initial)
        posthoc_vx = fit_x['slope']
        posthoc_vy0 = fit_y['b']  # dy/dt at t=0
        posthoc_vz = fit_z['b']   # dz/dt at t=0 (quadratic initial velocity)
        posthoc_speed = math.sqrt(posthoc_vx**2 + posthoc_vy0**2 + posthoc_vz**2)

        # Gravity measurement (decomposed into camera frame)
        measured_g_y = fit_y['measured_g']
        measured_g_z = fit_z['measured_g']
        # Composite gravity magnitude
        measured_g = math.sqrt(measured_g_y**2 + measured_g_z**2)
        g_error_pct = (measured_g - self.GRAVITY_EXPECTED) / self.GRAVITY_EXPECTED * 100
        g_y_error_pct = (measured_g_y - self.G_Y_EXPECTED) / self.G_Y_EXPECTED * 100
        g_z_error_pct = ((measured_g_z - self.G_Z_EXPECTED) / abs(self.G_Z_EXPECTED) * 100
                         if abs(self.G_Z_EXPECTED) > 1 else 0.0)

        # Finite-difference velocities
        fd = self._compute_finite_differences(t, x, y, z)

        # Per-axis fit residuals (RMS)
        res_x = x - np.polyval(fit_x['coeffs'], t)
        res_y = y - np.polyval(fit_y['coeffs'], t)
        res_z = z - np.polyval(fit_z['coeffs'], t)
        fit_residuals = {
            'rms_x': float(np.sqrt(np.mean(res_x**2))),
            'rms_y': float(np.sqrt(np.mean(res_y**2))),
            'rms_z': float(np.sqrt(np.mean(res_z**2))),
        }

        # Forward prediction errors (whole trajectory — kept for reference)
        pred_errors = self._compute_prediction_errors(
            t, x, y, z, posthoc_vx, posthoc_vy0, posthoc_vz,
            gravity=self.GRAVITY_EXPECTED,
            camera_pitch_deg=self.CAMERA_PITCH_DEG)

        # Forward prediction errors (using measured gravity from fit)
        pred_errors_measured = self._compute_prediction_errors(
            t, x, y, z, posthoc_vx, posthoc_vy0, posthoc_vz,
            gravity=abs(measured_g),
            camera_pitch_deg=self.CAMERA_PITCH_DEG)

        # Per-arc forward prediction (MEANINGFUL metric — stays within arc)
        per_arc_pred_errors = []
        for arc, ar in zip(arcs, arc_results):
            arc_vx = ar['fit_x']['slope']
            arc_vy0 = ar['fit_y']['b']
            arc_vz = ar['fit_z']['b']  # quadratic initial velocity
            pe_arc = self._compute_prediction_errors(
                arc['t'], arc['x'], arc['y'], arc['z'],
                arc_vx, arc_vy0, arc_vz,
                gravity=self.GRAVITY_EXPECTED,
                camera_pitch_deg=self.CAMERA_PITCH_DEG)
            per_arc_pred_errors.append(pe_arc)

        # Real-time vs post-hoc comparison — use PER-ARC fit at snapshot time
        rt_comparison = None
        if self.predictor_was_ready and self.rt_corrected is not None:
            rt_vx, rt_vy, rt_vz = self.rt_corrected
            rt_speed = math.sqrt(rt_vx**2 + rt_vy**2 + rt_vz**2)

            t_snap = self.rt_snap_time if self.rt_snap_time is not None else 0.0

            # Find which arc the snapshot falls in
            snap_arc = self._find_arc_for_time(t_snap, arcs, arc_results)

            if snap_arc is not None:
                # Use per-arc fit as reference
                t_in_arc = t_snap - snap_arc['t_abs'][0]
                ph_vx_at_snap = snap_arc['fit_x']['slope']
                ph_vy_at_snap = snap_arc['fit_y']['b'] + snap_arc['fit_y']['measured_g'] * t_in_arc
                ph_vz_at_snap = snap_arc['fit_z']['b'] + snap_arc['fit_z']['measured_g'] * t_in_arc
            else:
                # Fallback to whole-trajectory fit
                ph_vx_at_snap = posthoc_vx
                ph_vy_at_snap = posthoc_vy0 + measured_g_y * t_snap
                ph_vz_at_snap = posthoc_vz + measured_g_z * t_snap

            rt_comparison = {
                'rt_vx': rt_vx, 'rt_vy': rt_vy, 'rt_vz': rt_vz,
                'rt_speed': rt_speed,
                'snap_time': t_snap,
                'ph_vx_at_snap': ph_vx_at_snap,
                'ph_vy_at_snap': ph_vy_at_snap,
                'ph_vz_at_snap': ph_vz_at_snap,
                'err_vx': abs(rt_vx - ph_vx_at_snap),
                'err_vy': abs(rt_vy - ph_vy_at_snap),
                'err_vz': abs(rt_vz - ph_vz_at_snap),
                'pct_vx': self._safe_pct(rt_vx, ph_vx_at_snap),
                'pct_vy': self._safe_pct(rt_vy, ph_vy_at_snap),
                'pct_vz': self._safe_pct(rt_vz, ph_vz_at_snap),
            }

        return {
            'valid': True,
            'n_points': n,
            'duration': duration,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'fit_z': fit_z,
            'posthoc_vx': posthoc_vx,
            'posthoc_vy0': posthoc_vy0,
            'posthoc_vz': posthoc_vz,
            'posthoc_speed': posthoc_speed,
            'measured_g': measured_g,
            'measured_g_y': measured_g_y,
            'measured_g_z': measured_g_z,
            'g_error_pct': g_error_pct,
            'g_y_error_pct': g_y_error_pct,
            'g_z_error_pct': g_z_error_pct,
            'finite_diff': fd,
            'fit_residuals': fit_residuals,
            'pred_errors': pred_errors,
            'pred_errors_measured': pred_errors_measured,
            'per_arc_pred_errors': per_arc_pred_errors,
            'rt_comparison': rt_comparison,
            't': t,
            'x': x, 'y': y, 'z': z,
            'bounce_indices': bounce_indices,
            'n_bounces': len(bounce_indices),
            'arcs': arcs,
            'arc_results': arc_results,
            'best_arc': best_arc,
        }

    def _fit_linear(self, t, values):
        """Fit values = slope*t + intercept. Returns dict with R²."""
        coeffs = np.polyfit(t, values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        fitted = np.polyval(coeffs, t)
        ss_res = np.sum((values - fitted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {'slope': float(slope), 'intercept': float(intercept),
                'r_squared': float(r_squared), 'coeffs': coeffs}

    def _fit_quadratic(self, t, values):
        """Fit values = a*t² + b*t + c. measured_g = 2*a (Y down = +g)."""
        coeffs = np.polyfit(t, values, 2)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        fitted = np.polyval(coeffs, t)
        ss_res = np.sum((values - fitted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        measured_g = 2.0 * a  # Y(t) = ½g·t² + Vy0·t + Y0 → a = ½g
        return {'a': float(a), 'b': float(b), 'c': float(c),
                'r_squared': float(r_squared), 'measured_g': float(measured_g),
                'coeffs': coeffs}

    def _compute_finite_differences(self, t, x, y, z):
        """Central differences for velocity visualization."""
        n = len(t)
        if n < 3:
            return {'t': np.array([]), 'vx': np.array([]),
                    'vy': np.array([]), 'vz': np.array([])}
        fd_t = t[1:-1]
        dt = t[2:] - t[:-2]
        # Avoid division by zero
        dt = np.where(dt > 1e-9, dt, 1e-9)
        fd_vx = (x[2:] - x[:-2]) / dt
        fd_vy = (y[2:] - y[:-2]) / dt
        fd_vz = (z[2:] - z[:-2]) / dt
        return {'t': fd_t, 'vx': fd_vx, 'vy': fd_vy, 'vz': fd_vz}

    def _compute_prediction_errors(self, t, x, y, z, vx0, vy0, vz0,
                                    gravity=None, camera_pitch_deg=0.0):
        """Forward prediction from first point using physics model."""
        if len(t) < 2:
            return {'t': np.array([]), 'errors': np.array([]),
                    'mean': 0.0, 'max': 0.0, 'n_points': 0}

        g = gravity if gravity is not None else self.GRAVITY_EXPECTED
        model = PhysicsModel(gravity=g, y_down=True,
                             enable_drag=False,
                             camera_pitch_deg=camera_pitch_deg)
        pos0 = (float(x[0]), float(y[0]), float(z[0]))
        vel0 = (vx0, vy0, vz0)

        errors = []
        error_t = []
        for i in range(1, len(t)):
            dt = float(t[i])
            pred = model.predict_position(pos0, vel0, dt)
            err = math.sqrt((x[i] - pred[0])**2 +
                            (y[i] - pred[1])**2 +
                            (z[i] - pred[2])**2)
            errors.append(err)
            error_t.append(t[i])

        errors = np.array(errors)
        error_t = np.array(error_t)
        return {
            't': error_t, 'errors': errors,
            'mean': float(np.mean(errors)) if len(errors) > 0 else 0.0,
            'max': float(np.max(errors)) if len(errors) > 0 else 0.0,
            'n_points': len(errors),
        }

    # ================================================================
    # BOUNCE SEGMENTATION — split trajectory into per-arc segments
    # ================================================================

    MIN_BOUNCE_FALL = 10.0   # cm, same as real-time detector
    BOUNCE_RISE_FRAMES = 2
    GAP_THRESHOLD = 0.06     # seconds — gaps larger than this split arcs

    def _detect_bounces(self, t, y):
        """
        Find bounce indices in Y data.

        Returns list of indices where bounces occur.
        Uses same logic as real-time detector: Y must have fallen
        MIN_BOUNCE_FALL, then rise for BOUNCE_RISE_FRAMES consecutive frames.
        """
        bounces = []
        y_max_since_start = y[0]  # max Y = lowest point (Y+ = down)
        y_start = y[0]
        rising_count = 0

        for i in range(1, len(y)):
            y_max_since_start = max(y_max_since_start, y[i])
            dy = y[i] - y[i - 1]

            if dy < 0:  # rising (Y decreasing)
                rising_count += 1
            else:
                rising_count = 0

            fall_amount = y_max_since_start - y_start

            if (fall_amount >= self.MIN_BOUNCE_FALL and
                    rising_count >= self.BOUNCE_RISE_FRAMES):
                bounces.append(i - 1)  # bounce point is where reversal started
                # Reset for next arc
                y_start = y[i]
                y_max_since_start = y[i]
                rising_count = 0

        return bounces

    def _detect_gaps(self, t):
        """Find indices where time gaps exceed GAP_THRESHOLD."""
        gaps = []
        for i in range(1, len(t)):
            if t[i] - t[i - 1] > self.GAP_THRESHOLD:
                gaps.append(i)
        return gaps

    def _segment_arcs(self, t, x, y, z):
        """
        Segment trajectory into arcs split by bounces AND time gaps.

        Time gaps (stereo dead zone) mean the ball crossed between cameras
        and may have bounced invisibly. Each continuous segment gets its
        own arc with independent physics fits.

        Returns list of dicts with per-arc data arrays and indices.
        """
        bounce_indices = self._detect_bounces(t, y)
        gap_indices = self._detect_gaps(t)

        # Merge all split points (bounces and gaps), deduplicate and sort
        all_splits = sorted(set(bounce_indices + gap_indices))

        # Build arc boundaries: [0, split0, split1, ..., end]
        boundaries = [0] + all_splits + [len(t)]
        arcs = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start < 4:  # need at least 4 points for quadratic fit
                continue
            arcs.append({
                'arc_num': len(arcs),
                'start_idx': start,
                'end_idx': end,
                't': t[start:end] - t[start],  # relative time within arc
                't_abs': t[start:end],           # absolute time
                'x': x[start:end],
                'y': y[start:end],
                'z': z[start:end],
                'n_points': end - start,
                'duration': t[end - 1] - t[start],
            })
        return arcs, bounce_indices

    def _analyze_arc(self, arc):
        """Fit physics curves to a single arc. Returns per-arc results."""
        t, y = arc['t'], arc['y']
        fit_y = self._fit_quadratic(t, y)
        fit_x = self._fit_linear(t, arc['x'])
        fit_z = self._fit_quadratic(t, arc['z'])  # Z is quadratic (camera pitch)

        measured_g_y = fit_y['measured_g']
        measured_g_z = fit_z['measured_g']
        measured_g = math.sqrt(measured_g_y**2 + measured_g_z**2)
        g_error_pct = ((measured_g - self.GRAVITY_EXPECTED) /
                       self.GRAVITY_EXPECTED * 100)
        g_y_error_pct = ((measured_g_y - self.G_Y_EXPECTED) /
                         self.G_Y_EXPECTED * 100)
        g_z_error_pct = ((measured_g_z - self.G_Z_EXPECTED) /
                         abs(self.G_Z_EXPECTED) * 100
                         if abs(self.G_Z_EXPECTED) > 1 else 0.0)

        return {
            'arc_num': arc['arc_num'],
            'n_points': arc['n_points'],
            'duration': arc['duration'],
            'measured_g': measured_g,
            'measured_g_y': measured_g_y,
            'measured_g_z': measured_g_z,
            'g_error_pct': g_error_pct,
            'g_y_error_pct': g_y_error_pct,
            'g_z_error_pct': g_z_error_pct,
            'fit_y': fit_y,
            'fit_x': fit_x,
            'fit_z': fit_z,
            'start_idx': arc['start_idx'],
            'end_idx': arc['end_idx'],
        }

    @staticmethod
    def _safe_pct(actual, reference):
        """Percentage error, avoiding division by near-zero."""
        if abs(reference) < 5.0:
            return None  # absolute error only
        return abs(actual - reference) / abs(reference) * 100.0

    @staticmethod
    def _find_arc_for_time(t_abs, arcs, arc_results):
        """Find which arc a given absolute time falls in. Returns arc_result with t_abs attached, or None."""
        for arc, ar in zip(arcs, arc_results):
            if arc['t_abs'][0] <= t_abs <= arc['t_abs'][-1]:
                # Return arc_result with t_abs for reference
                result = dict(ar)
                result['t_abs'] = arc['t_abs']
                return result
        return None


# ================================================================
# VELOCITY CHART — matplotlib → BGR numpy array
# ================================================================

class VelocityChart:
    """Renders 2x3 analysis chart as a BGR numpy array for OpenCV display."""

    CHART_W = 960
    CHART_H = 640

    def render(self, analysis, throw_number):
        """
        Render analysis results into a BGR numpy array.

        Args:
            analysis: dict from ThrowAnalyzer.analyze()
            throw_number: int

        Returns:
            np.ndarray (H, W, 3) BGR
        """
        if not analysis.get('valid'):
            img = np.full((self.CHART_H, self.CHART_W, 3), 20, dtype=np.uint8)
            reason = analysis.get('reason', 'unknown')
            cv2.putText(img, f"Throw #{throw_number}: {reason}",
                        (30, self.CHART_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img

        dpi = 100
        fig = Figure(figsize=(self.CHART_W / dpi, self.CHART_H / dpi), dpi=dpi)
        fig.set_facecolor('#141418')
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3)

        t = analysis['t']
        x, y, z = analysis['x'], analysis['y'], analysis['z']
        fit_x = analysis['fit_x']
        fit_y = analysis['fit_y']
        fit_z = analysis['fit_z']
        fd = analysis['finite_diff']
        rt = analysis.get('rt_comparison')

        style = {'facecolor': '#141418'}
        for row in axes:
            for ax in row:
                ax.set_facecolor('#1a1a22')
                ax.tick_params(colors='#888', labelsize=7)
                ax.spines['bottom'].set_color('#444')
                ax.spines['top'].set_color('#444')
                ax.spines['left'].set_color('#444')
                ax.spines['right'].set_color('#444')

        t_fit = np.linspace(t[0], t[-1], 200)

        # ---- Row 1: Position fits ----
        self._plot_position(axes[0, 0], t, x, t_fit, fit_x, 'X(t)',
                            is_quadratic=False)
        self._plot_position(axes[0, 1], t, y, t_fit, fit_y, 'Y(t)',
                            is_quadratic=True, analysis=analysis)
        self._plot_position(axes[0, 2], t, z, t_fit, fit_z, 'Z(t)',
                            is_quadratic=True)

        # ---- Row 2: Velocity ----
        rt_snap_t = rt['snap_time'] if rt else None
        self._plot_velocity(axes[1, 0], fd, 'vx', fit_x['slope'],
                            rt['rt_vx'] if rt else None, 'Vx(t)',
                            rt_time=rt_snap_t)
        self._plot_velocity_y(axes[1, 1], fd, fit_y, analysis, rt)
        self._plot_velocity_z(axes[1, 2], fd, fit_z, analysis, rt)

        fig.suptitle(f'Throw #{throw_number}  |  {analysis["n_points"]} pts  |  '
                     f'{analysis["duration"]*1000:.0f}ms  |  '
                     f'{analysis["posthoc_speed"]:.0f} cm/s',
                     color='#ccc', fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        # RGBA → BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        if img_bgr.shape[:2] != (self.CHART_H, self.CHART_W):
            img_bgr = cv2.resize(img_bgr, (self.CHART_W, self.CHART_H))
        return img_bgr

    def _plot_position(self, ax, t, vals, t_fit, fit, title,
                       is_quadratic=False, analysis=None):
        ax.scatter(t * 1000, vals, c='#33cc55', s=12, alpha=0.8, zorder=3)

        fitted = np.polyval(fit['coeffs'], t_fit)
        ax.plot(t_fit * 1000, fitted, color='#dda830', linewidth=1.5, zorder=2)

        r2 = fit['r_squared']
        r2_color = '#33cc55' if r2 > 0.9 else '#ddaa00' if r2 > 0.8 else '#dd3333'
        label = f"R²={r2:.3f}"

        if is_quadratic and analysis:
            mg = analysis['measured_g']
            ge = analysis['g_error_pct']
            g_color = '#33cc55' if abs(ge) < 5 else '#ddaa00' if abs(ge) < 15 else '#dd3333'
            label += f"\ng={mg:.0f} ({ge:+.1f}%)"
            ax.text(0.97, 0.05, label, transform=ax.transAxes,
                    fontsize=7, color=g_color, ha='right', va='bottom',
                    family='monospace')
        else:
            ax.text(0.97, 0.05, label, transform=ax.transAxes,
                    fontsize=7, color=r2_color, ha='right', va='bottom',
                    family='monospace')

        ax.set_title(title, color='#aaa', fontsize=9)
        ax.set_xlabel('ms', color='#666', fontsize=7)
        ax.set_ylabel('cm', color='#666', fontsize=7)

    def _plot_velocity(self, ax, fd, axis_key, posthoc_val, rt_val, title,
                       rt_time=None):
        if len(fd['t']) > 0:
            ax.scatter(fd['t'] * 1000, fd[axis_key], c='#5588ee',
                       s=10, alpha=0.7, zorder=3, label='FD')

        t_range = fd['t']
        if len(t_range) > 0:
            ax.axhline(y=posthoc_val, color='#dda830', linestyle='--',
                       linewidth=1.5, label=f'posthoc={posthoc_val:.1f}')
        if rt_val is not None and rt_time is not None:
            ax.scatter([rt_time * 1000], [rt_val], c='#ee4444', s=60,
                       marker='D', zorder=5, label=f'RT={rt_val:.1f}')
        elif rt_val is not None:
            ax.axhline(y=rt_val, color='#ee4444', linestyle=':',
                       linewidth=1.5, label=f'RT={rt_val:.1f}')

        ax.set_title(title, color='#aaa', fontsize=9)
        ax.set_xlabel('ms', color='#666', fontsize=7)
        ax.set_ylabel('cm/s', color='#666', fontsize=7)
        ax.legend(fontsize=6, loc='best', facecolor='#222', edgecolor='#444',
                  labelcolor='#ccc')

    def _plot_velocity_y(self, ax, fd, fit_y, analysis, rt):
        if len(fd['t']) > 0:
            ax.scatter(fd['t'] * 1000, fd['vy'], c='#5588ee',
                       s=10, alpha=0.7, zorder=3, label='FD')

        # Expected Vy(t) = Vy0 + g*t (from quadratic fit: Vy0=b, g=2a)
        t_range = fd['t']
        if len(t_range) > 0:
            t_line = np.linspace(t_range[0], t_range[-1], 100)
            vy_line = fit_y['b'] + fit_y['measured_g'] * t_line
            ax.plot(t_line * 1000, vy_line, color='#dda830', linewidth=1.5,
                    label=f'posthoc Vy(t)')

        if rt is not None:
            rt_vy = rt['rt_vy']
            t_snap = rt.get('snap_time', 0.0)
            ax.scatter([t_snap * 1000], [rt_vy], c='#ee4444', s=60,
                       marker='D', zorder=5, label=f'RT={rt_vy:.1f}')

        mg = analysis.get('measured_g_y', analysis.get('measured_g', 0))
        ge = analysis.get('g_y_error_pct', analysis.get('g_error_pct', 0))
        ax.text(0.97, 0.05, f"g_y={mg:.0f} ({ge:+.1f}%)",
                transform=ax.transAxes, fontsize=7,
                color='#33cc55' if abs(ge) < 5 else '#dd3333',
                ha='right', va='bottom', family='monospace')

        ax.set_title('Vy(t)', color='#aaa', fontsize=9)
        ax.set_xlabel('ms', color='#666', fontsize=7)
        ax.set_ylabel('cm/s', color='#666', fontsize=7)
        ax.legend(fontsize=6, loc='best', facecolor='#222', edgecolor='#444',
                  labelcolor='#ccc')

    def _plot_velocity_z(self, ax, fd, fit_z, analysis, rt):
        """Plot Vz with time-varying reference (gravity in Z from camera pitch)."""
        if len(fd['t']) > 0:
            ax.scatter(fd['t'] * 1000, fd['vz'], c='#5588ee',
                       s=10, alpha=0.7, zorder=3, label='FD')

        # Expected Vz(t) = Vz0 + g_z*t (from quadratic fit)
        t_range = fd['t']
        if len(t_range) > 0:
            t_line = np.linspace(t_range[0], t_range[-1], 100)
            vz_line = fit_z['b'] + fit_z['measured_g'] * t_line
            ax.plot(t_line * 1000, vz_line, color='#dda830', linewidth=1.5,
                    label=f'posthoc Vz(t)')

        if rt is not None:
            rt_vz = rt['rt_vz']
            t_snap = rt.get('snap_time', 0.0)
            ax.scatter([t_snap * 1000], [rt_vz], c='#ee4444', s=60,
                       marker='D', zorder=5, label=f'RT={rt_vz:.1f}')

        mg_z = analysis.get('measured_g_z', 0)
        ge_z = analysis.get('g_z_error_pct', 0)
        ax.text(0.97, 0.05, f"g_z={mg_z:.0f} ({ge_z:+.1f}%)",
                transform=ax.transAxes, fontsize=7,
                color='#33cc55' if abs(ge_z) < 20 else '#dd3333',
                ha='right', va='bottom', family='monospace')

        ax.set_title('Vz(t)', color='#aaa', fontsize=9)
        ax.set_xlabel('ms', color='#666', fontsize=7)
        ax.set_ylabel('cm/s', color='#666', fontsize=7)
        ax.legend(fontsize=6, loc='best', facecolor='#222', edgecolor='#444',
                  labelcolor='#ccc')


# ================================================================
# VELOCITY VALIDATOR — main loop
# ================================================================

class VelocityValidator:
    """
    Main validator: cameras → stereo tracking → velocity analysis.

    Records throws, captures real-time velocity at frame 6,
    then runs ThrowAnalyzer post-hoc and displays chart.
    """

    CAM_W, CAM_H = 480, 300
    CHART_W, CHART_H = 960, 640
    SNAP_AFTER = 6          # capture RT velocity at this frame
    LOST_THRESHOLD = 20     # frames lost before ending throw

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
        self.chart_renderer = VelocityChart()

        # Throw state
        self._throw_active = False
        self._throw_positions = []
        self._lost_count = 0
        self._throw_count = 0
        self._diag_frame = 0
        self._throw_t0 = None

        # RT velocity snapshot
        self._rt_velocity = None
        self._rt_corrected = None
        self._rt_snapped = False
        self._predictor_was_ready = False
        self._rt_snap_time = None  # relative to throw t0

        # Display
        self._chart_img = None
        self._history = []  # list of analysis dicts

        # FPS
        self._fps_n = 0
        self._fps_t = time.perf_counter()
        self._fps = 0.0

        # Velocity convergence tracking (per-throw)
        self._velocity_snapshots = []
        self._prev_accept_t = None

    # ================================================================
    # SETUP
    # ================================================================

    def check_calibration(self):
        req = ['camera0_intrinsics.dat', 'camera1_intrinsics.dat',
               'camera0_rot_trans.dat', 'camera1_rot_trans.dat']
        missing = [f for f in req
                   if not os.path.exists(os.path.join(self.calib_dir, f))]
        if missing:
            print(f"  Missing calibration files: {missing}")
            return False
        return True

    def warmup(self):
        print("\n  Remove ball from view. Learning background (SPACE=skip)...")
        t0, dur = time.time(), 2.0
        while time.time() - t0 < dur:
            if not self.tri.cap_left.grab():
                continue
            if not self.tri.cap_right.grab():
                continue
            _, fl = self.tri.cap_left.retrieve()
            _, fr = self.tri.cap_right.retrieve()
            if fl is None or fr is None:
                continue
            self.tri.build_background(fl, fr)

            p = min((time.time() - t0) / dur, 1.0)
            dl = cv2.resize(fl, (self.CAM_W, self.CAM_H))
            bw = int(p * (self.CAM_W - 40))
            cv2.rectangle(dl, (20, self.CAM_H - 30),
                          (20 + bw, self.CAM_H - 15), (0, 255, 255), -1)
            cv2.putText(dl, f"BG: {p*100:.0f}%", (20, self.CAM_H - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow('Velocity Validation', dl)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                break
            elif k == ord('q'):
                return False
        print("  Ready!\n")
        return True

    # ================================================================
    # THROW MANAGEMENT
    # ================================================================

    def _start_throw(self):
        self._throw_active = True
        self._throw_positions = []
        self._lost_count = 0
        self._throw_count += 1
        self._diag_frame = 0
        self._rt_velocity = None
        self._rt_corrected = None
        self._rt_snapped = False
        self._predictor_was_ready = False
        self._rt_snap_time = None
        self.pred.reset()

        self._throw_t0 = None  # set on first accepted point
        self._velocity_snapshots = []
        self._prev_accept_t = None

        print(f"\n  [THROW #{self._throw_count}]")
        print(f"  {'Frame':>6}  {'Time(s)':>8}  {'dt_ms':>5}  {'X(cm)':>8}  {'Y(cm)':>8}  "
              f"{'Z(cm)':>8}  |  {'Disp':>5}  {'Repr':>5}  |  Status")
        print(f"  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*8}  {'─'*8}  "
              f"{'─'*8}  |  {'─'*5}  {'─'*5}  |  {'─'*30}")

    def _end_throw(self):
        self._throw_active = False
        self._lost_count = 0

        positions = self._throw_positions
        n = len(positions)

        if n < 4:
            print(f"\n  [END] insufficient data ({n} points)")
            return

        duration = positions[-1][3] - positions[0][3]
        if duration < 0.05:
            print(f"\n  [END] throw too short ({duration*1000:.0f}ms)")
            return

        analyzer = ThrowAnalyzer(
            positions=positions,
            rt_velocity=self._rt_velocity,
            rt_corrected_velocity=self._rt_corrected,
            predictor_was_ready=self._predictor_was_ready,
            rt_snap_time=self._rt_snap_time)

        analysis = analyzer.analyze()
        analysis['throw_number'] = self._throw_count
        analysis['velocity_snapshots'] = list(self._velocity_snapshots)
        self._history.append(analysis)

        self._print_analysis(analysis)
        self._chart_img = self.chart_renderer.render(analysis, self._throw_count)

    def _reset_throw(self):
        self._throw_active = False
        self._throw_positions = []
        self._lost_count = 0
        self._throw_t0 = None
        self._rt_velocity = None
        self._rt_corrected = None
        self._rt_snapped = False
        self._predictor_was_ready = False
        self._rt_snap_time = None
        self._velocity_snapshots = []
        self._prev_accept_t = None
        self.pred.reset()
        print("\n  [RESET]")

    # ================================================================
    # TERMINAL OUTPUT
    # ================================================================

    def _print_analysis(self, analysis):
        if not analysis.get('valid'):
            reason = analysis.get('reason', 'unknown')
            print(f"\n  [ANALYSIS] {reason}")
            return

        n = analysis['n_points']
        dur_ms = analysis['duration'] * 1000
        spd = analysis['posthoc_speed']

        print()
        print(f"  {'═'*56}")
        print(f"  THROW #{analysis['throw_number']} ANALYSIS  |  "
              f"{n} pts  |  {dur_ms:.0f}ms  |  {spd:.0f} cm/s")
        print(f"  {'═'*56}")

        fx = analysis['fit_x']
        fy = analysis['fit_y']
        fz = analysis['fit_z']

        print(f"  Fits:")
        r2x_flag = " !" if fx['r_squared'] < 0.8 else ""
        r2z_flag = " !" if fz['r_squared'] < 0.8 else ""
        r2y_flag = " !" if fy['r_squared'] < 0.8 else ""
        print(f"    X(t) = {fx['slope']:+.1f}t + {fx['intercept']:.1f}"
              f"{'':>20}R²={fx['r_squared']:.3f}{r2x_flag}")
        print(f"    Y(t) = {fy['a']:.1f}t² {fy['b']:+.1f}t {fy['c']:+.1f}"
              f"{'':>10}R²={fy['r_squared']:.3f}{r2y_flag}"
              f"   g_y={fy['measured_g']:.0f} ({analysis['g_y_error_pct']:+.1f}%)")
        print(f"    Z(t) = {fz['a']:.1f}t² {fz['b']:+.1f}t {fz['c']:+.1f}"
              f"{'':>10}R²={fz['r_squared']:.3f}{r2z_flag}"
              f"   g_z={fz['measured_g']:.0f} ({analysis['g_z_error_pct']:+.1f}%)")
        print(f"    |g| = sqrt(g_y²+g_z²) = {analysis['measured_g']:.0f} "
              f"({analysis['g_error_pct']:+.1f}% vs 981)")

        # Flag anomalous gravity
        mg = analysis['measured_g']
        if mg < 700 or mg > 1300:
            print(f"  ** ANOMALOUS GRAVITY (whole traj): {mg:.0f} cm/s² **")

        # Per-arc analysis (arcs split by bounces AND time gaps)
        arc_results = analysis.get('arc_results', [])
        arcs = analysis.get('arcs', [])
        best_arc = analysis.get('best_arc')
        n_arcs = len(arc_results)
        if n_arcs > 1:
            print(f"\n  ARCS: {n_arcs} (split by bounces + stereo gaps)")
            print(f"  Per-arc gravity (expected: g_y={ThrowAnalyzer.G_Y_EXPECTED:.0f}  "
                  f"g_z={ThrowAnalyzer.G_Z_EXPECTED:.0f}):")
            for ar in arc_results:
                marker = " <<<" if best_arc and ar['arc_num'] == best_arc['arc_num'] else ""
                g_color = "OK" if abs(ar['g_error_pct']) < 10 else "!"
                print(f"    Arc {ar['arc_num']}: {ar['n_points']:2d} pts  "
                      f"{ar['duration']*1000:.0f}ms  "
                      f"|g|={ar['measured_g']:.0f} ({ar['g_error_pct']:+.1f}%)  "
                      f"g_y={ar['measured_g_y']:.0f}  g_z={ar['measured_g_z']:.0f}  "
                      f"R²y={ar['fit_y']['r_squared']:.3f}  {g_color}{marker}")
            if best_arc:
                print(f"  Best arc: |g|={best_arc['measured_g']:.0f} "
                      f"g_y={best_arc['measured_g_y']:.0f} "
                      f"g_z={best_arc['measured_g_z']:.0f}")
        elif n_arcs == 1:
            ar = arc_results[0]
            g_color = "OK" if abs(ar['g_error_pct']) < 10 else "!"
            print(f"  Single arc: |g|={ar['measured_g']:.0f} ({ar['g_error_pct']:+.1f}%)  "
                  f"g_y={ar['measured_g_y']:.0f}  g_z={ar['measured_g_z']:.0f}  "
                  f"R²y={ar['fit_y']['r_squared']:.3f}  {g_color}")

        # Fit residuals
        fr = analysis.get('fit_residuals')
        if fr:
            print(f"  Fit residuals (RMS cm):  "
                  f"X={fr['rms_x']:.2f}  Y={fr['rms_y']:.2f}  Z={fr['rms_z']:.2f}")

        print()
        print(f"  Post-hoc (t=0):  Vx={analysis['posthoc_vx']:+.1f}  "
              f"Vy0={analysis['posthoc_vy0']:+.1f}  "
              f"Vz={analysis['posthoc_vz']:+.1f}  "
              f"Spd={spd:.0f} cm/s")

        rt = analysis.get('rt_comparison')
        if rt:
            t_snap = rt['snap_time']
            print(f"  Post-hoc (t={t_snap:.3f}s):  "
                  f"Vx={rt['ph_vx_at_snap']:+.1f}  "
                  f"Vy={rt['ph_vy_at_snap']:+.1f}  "
                  f"Vz={rt['ph_vz_at_snap']:+.1f}")
            print(f"  Real-time(t={t_snap:.3f}s):  "
                  f"Vx={rt['rt_vx']:+.1f}  "
                  f"Vy={rt['rt_vy']:+.1f}  "
                  f"Vz={rt['rt_vz']:+.1f}  "
                  f"Spd={rt['rt_speed']:.0f} cm/s")

            parts = []
            for axis in ['vx', 'vy', 'vz']:
                err = rt[f'err_{axis}']
                pct = rt[f'pct_{axis}']
                label = axis.upper()[0] + axis[1]
                if pct is not None:
                    parts.append(f"{label}={err:.1f}({pct:.1f}%)")
                else:
                    parts.append(f"{label}={err:.1f}(abs)")
            print(f"  Error @t={t_snap:.3f}s: {'  '.join(parts)}")
        else:
            print(f"  Real-time: (predictor not ready — no comparison)")

        # Per-arc forward prediction (meaningful — stays within each arc)
        pa_pe = analysis.get('per_arc_pred_errors', [])
        if pa_pe:
            all_arc_means = [p['mean'] for p in pa_pe if p['n_points'] > 0]
            all_arc_maxes = [p['max'] for p in pa_pe if p['n_points'] > 0]
            if all_arc_means:
                overall_mean = np.mean(all_arc_means)
                overall_max = max(all_arc_maxes)
                print(f"  Per-arc fwd pred (g=981): mean={overall_mean:.1f}cm  "
                      f"max={overall_max:.1f}cm  ({len(all_arc_means)} arcs)")
                for i, p in enumerate(pa_pe):
                    if p['n_points'] > 0:
                        print(f"    Arc {i}: mean={p['mean']:.1f}cm  "
                              f"max={p['max']:.1f}cm  ({p['n_points']} pts)")

        # --- Velocity convergence table (per-arc reference) ---
        snapshots = analysis.get('velocity_snapshots', [])
        if snapshots and analysis.get('valid'):
            print()
            print(f"  Velocity convergence (RT vs per-arc fit):")
            print(f"  {'#pt':>3}  {'t(s)':>6}  {'arc':>3}  |  "
                  f"{'Vx_RT':>7}  {'Vy_corr':>7}  {'Vz_RT':>7}  |  "
                  f"{'ref_Vx':>7}  {'ref_Vy':>7}  {'ref_Vz':>7}  |  "
                  f"{'eVx':>5}  {'eVy':>5}  {'eVz':>5}  {'e3D':>5}")
            print(f"  {'─'*3}  {'─'*6}  {'─'*3}  |  "
                  f"{'─'*7}  {'─'*7}  {'─'*7}  |  "
                  f"{'─'*7}  {'─'*7}  {'─'*7}  |  "
                  f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")

            for snap in snapshots:
                t_s = snap['t']
                # Find which arc this snapshot belongs to
                snap_arc = ThrowAnalyzer._find_arc_for_time(t_s, arcs, arc_results)
                if snap_arc is not None:
                    t_in_arc = t_s - snap_arc['t_abs'][0]
                    ref_vx = snap_arc['fit_x']['slope']
                    ref_vy = snap_arc['fit_y']['b'] + snap_arc['fit_y']['measured_g'] * t_in_arc
                    ref_vz = snap_arc['fit_z']['b'] + snap_arc['fit_z']['measured_g'] * t_in_arc
                    arc_label = str(snap_arc['arc_num'])
                else:
                    # No matching arc (snapshot in gap)
                    ref_vx = analysis['posthoc_vx']
                    ref_vy = analysis['posthoc_vy0'] + analysis['measured_g_y'] * t_s
                    ref_vz = analysis['posthoc_vz'] + analysis['measured_g_z'] * t_s
                    arc_label = "?"

                evx = abs(snap['vx'] - ref_vx)
                evy = abs(snap['vy_corr'] - ref_vy)
                evz = abs(snap['vz'] - ref_vz)
                e3d = math.sqrt(evx**2 + evy**2 + evz**2)
                print(f"  {snap['n_pts']:3d}  {t_s:6.3f}  {arc_label:>3}  |  "
                      f"{snap['vx']:+7.1f}  {snap['vy_corr']:+7.1f}  "
                      f"{snap['vz']:+7.1f}  |  "
                      f"{ref_vx:+7.1f}  {ref_vy:+7.1f}  {ref_vz:+7.1f}  |  "
                      f"{evx:5.1f}  {evy:5.1f}  {evz:5.1f}  {e3d:5.1f}")

        # --- Raw data dump ---
        print()
        print(f"  Raw data (copy-paste for analysis):")
        print(f"  # t_s, x_cm, y_cm, z_cm")
        arr = np.array(analysis.get('t', []))
        x_arr = analysis.get('x', np.array([]))
        y_arr = analysis.get('y', np.array([]))
        z_arr = analysis.get('z', np.array([]))
        for i in range(len(arr)):
            print(f"  {float(arr[i]):.4f}, {float(x_arr[i]):.2f}, "
                  f"{float(y_arr[i]):.2f}, {float(z_arr[i]):.2f}")

        print(f"  {'═'*56}")

    def _print_summary(self):
        valid = [a for a in self._history if a.get('valid')]
        print(f"\n  {'═'*56}")
        print(f"  ALL-THROWS SUMMARY  ({len(valid)} valid / {len(self._history)} total)")
        print(f"  {'═'*56}")

        if not valid:
            print("  No valid throws recorded.")
            print(f"  {'═'*56}")
            return

        for a in valid:
            tn = a['throw_number']
            n = a['n_points']
            dur = a['duration'] * 1000
            n_arcs = len(a.get('arc_results', []))
            best = a.get('best_arc')
            rt = a.get('rt_comparison')

            rt_str = ""
            if rt:
                rt_str = (f"  err@t={rt['snap_time']:.2f}s: "
                          f"Vx={rt['err_vx']:.1f} "
                          f"Vy={rt['err_vy']:.1f} Vz={rt['err_vz']:.1f}")
            else:
                rt_str = "  (no RT)"

            # Per-arc forward prediction
            pa_pe = a.get('per_arc_pred_errors', [])
            arc_means = [p['mean'] for p in pa_pe if p['n_points'] > 0]
            pred_str = ""
            if arc_means:
                pred_str = f"  fwd: {np.mean(arc_means):.1f}/{max(arc_means):.1f}cm"

            arc_str = f"  arcs={n_arcs}" if n_arcs > 1 else ""
            best_g_str = ""
            if best:
                best_g_str = (f"  |g|={best['measured_g']:.0f}"
                              f"(y={best['measured_g_y']:.0f},z={best['measured_g_z']:.0f})")

            print(f"  #{tn:2d}: {n:2d}pts {dur:4.0f}ms"
                  f"{arc_str}{best_g_str}{rt_str}{pred_str}")

        # Aggregate gravity stats (best arc per throw)
        best_g_vals = [a['best_arc']['measured_g'] for a in valid
                       if a.get('best_arc')]
        best_gy_vals = [a['best_arc']['measured_g_y'] for a in valid
                        if a.get('best_arc')]
        best_gz_vals = [a['best_arc']['measured_g_z'] for a in valid
                        if a.get('best_arc')]
        if best_g_vals:
            print(f"\n  Gravity (best arc):")
            print(f"    |g|: mean={np.mean(best_g_vals):.0f}  "
                  f"std={np.std(best_g_vals):.0f}  expected=981")
            print(f"    g_y: mean={np.mean(best_gy_vals):.0f}  "
                  f"std={np.std(best_gy_vals):.0f}  "
                  f"expected={ThrowAnalyzer.G_Y_EXPECTED:.0f}")
            print(f"    g_z: mean={np.mean(best_gz_vals):.0f}  "
                  f"std={np.std(best_gz_vals):.0f}  "
                  f"expected={ThrowAnalyzer.G_Z_EXPECTED:.0f}")

        # Aggregate velocity errors (only throws with RT comparison, per-arc ref)
        rt_errs = [a['rt_comparison'] for a in valid
                   if a.get('rt_comparison') is not None]
        if rt_errs:
            vx_errs = [r['err_vx'] for r in rt_errs]
            vy_errs = [r['err_vy'] for r in rt_errs]
            vz_errs = [r['err_vz'] for r in rt_errs]
            print(f"  Vel err (per-arc ref): Vx={np.mean(vx_errs):.1f}  "
                  f"Vy={np.mean(vy_errs):.1f}  "
                  f"Vz={np.mean(vz_errs):.1f} cm/s  ({len(rt_errs)} throws)")

        # Aggregate per-arc forward prediction
        all_arc_means = []
        for a in valid:
            pa_pe = a.get('per_arc_pred_errors', [])
            for p in pa_pe:
                if p['n_points'] > 0:
                    all_arc_means.append(p['mean'])
        if all_arc_means:
            print(f"  Per-arc fwd pred: mean={np.mean(all_arc_means):.1f}cm  "
                  f"worst={np.max(all_arc_means):.1f}cm  "
                  f"({len(all_arc_means)} arcs)")

        print(f"  {'═'*56}")

    # ================================================================
    # DISPLAY
    # ================================================================

    def _build_display(self, result):
        lv, rv = self.tri.draw_results(result)

        # FPS counter
        self._fps_n += 1
        if self._fps_n % 30 == 0:
            now = time.perf_counter()
            self._fps = 30.0 / (now - self._fps_t)
            self._fps_t = now

        # Camera overlays
        if result['found_3d']:
            x, y, z = result['position_3d']
            cv2.putText(lv, f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        wu = self.tri.warmup_status()
        if not wu['ready']:
            cv2.putText(lv, f"Warmup {wu['progress']*100:.0f}%",
                        (10, lv.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ps = self.pred.get_stats()
        cv2.putText(lv, f"FPS:{self._fps:.0f} Buf:{ps['buffer_size']} "
                    f"Rej:{ps['rejected']}",
                    (10, lv.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

        vel = self.pred.get_velocity()
        if vel['valid'] and self._throw_active:
            cv2.putText(lv, f"Spd:{vel['speed']:.0f}cm/s",
                        (10, lv.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

        # Status info on right panel
        self._draw_status(rv, result)
        cv2.putText(rv, "q r b SPACE p",
                    (10, self.CAM_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        lv = cv2.resize(lv, (self.CAM_W, self.CAM_H))
        rv = cv2.resize(rv, (self.CAM_W, self.CAM_H))
        cam_row = np.hstack([lv, rv])

        # Chart panel (or placeholder)
        if self._chart_img is not None:
            chart = self._chart_img
        else:
            chart = np.full((self.CHART_H, self.CHART_W, 3), 20, dtype=np.uint8)
            cv2.putText(chart, "Toss ball — analysis chart appears after throw",
                        (30, self.CHART_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        if cam_row.shape[1] != chart.shape[1]:
            chart = cv2.resize(chart, (cam_row.shape[1], chart.shape[0]))

        return np.vstack([cam_row, chart])

    def _draw_status(self, frame, result):
        y = 20
        ps = self.pred.get_stats()
        vel = self.pred.get_velocity()

        if result['found_3d']:
            x3, y3, z3 = result['position_3d']
            cv2.putText(frame, "Ball (cm):", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y += 18
            cv2.putText(frame, f" X:{x3:7.1f}  Y:{y3:7.1f}  Z:{z3:7.1f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No 3D detection", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        y += 22
        if vel['valid']:
            cv2.putText(frame,
                        f"Vel: vx={vel['vx']:+.0f} vy={vel['vy']:+.0f} "
                        f"vz={vel['vz']:+.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 0), 1)
            y += 16
            cv2.putText(frame, f"Speed: {vel['speed']:.0f} cm/s",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 0), 1)
        else:
            cv2.putText(frame, "Vel: --", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        y += 22
        total = ps['accepted'] + ps['rejected']
        rej_pct = ps['rejected'] / max(total, 1) * 100
        clr = ((0, 200, 0) if rej_pct < 30 else
               (0, 200, 255) if rej_pct < 60 else (0, 0, 255))
        cv2.putText(frame,
                    f"Buf:{ps['buffer_size']}/{self.pred.buffer_size} "
                    f"Acc:{ps['accepted']} Rej:{ps['rejected']}({rej_pct:.0f}%)",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, clr, 1)

        y += 18
        if ps['is_ready']:
            cv2.putText(frame, "PREDICTION READY", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "NOT READY", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        y += 22
        cv2.putText(frame,
                    f"Throw #{self._throw_count}  "
                    f"Pts: {len(self._throw_positions)}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 255), 1)

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def run(self):
        print("\n" + "=" * 60)
        print("  VELOCITY ESTIMATION VALIDATION")
        print("=" * 60)
        print(f"  Cams: L={self.cam_l} R={self.cam_r}")
        print(f"  Toss ball after warmup — velocity analysis after each throw")
        print(f"  q=quit  r=reset  b=bg_reset  SPACE=skip_warmup  p=summary")
        print("=" * 60)

        if not self.check_calibration():
            return

        try:
            self.tri = StereoTriangulator(
                calibration_dir=self.calib_dir,
                cam_left_id=self.cam_l, cam_right_id=self.cam_r)
        except Exception as e:
            print(f"ERROR loading calibration: {e}")
            return

        # Print calibration details for debug log
        print(f"\n  Calibration:")
        print(f"    cam0: fx={self.tri.cmtx0[0,0]:.1f}  fy={self.tri.cmtx0[1,1]:.1f}  "
              f"cx={self.tri.cmtx0[0,2]:.1f}  cy={self.tri.cmtx0[1,2]:.1f}")
        print(f"    cam1: fx={self.tri.cmtx1[0,0]:.1f}  fy={self.tri.cmtx1[1,1]:.1f}  "
              f"cx={self.tri.cmtx1[0,2]:.1f}  cy={self.tri.cmtx1[1,2]:.1f}")
        print(f"    cam0 dist: {self.tri.dist0.flatten()}")
        print(f"    cam1 dist: {self.tri.dist1.flatten()}")
        baseline = self.tri.get_baseline()
        rect_f = self.tri.P_rect0[0, 0]
        rect_b = abs(self.tri.P_rect1[0, 3]) / rect_f
        print(f"    baseline={baseline:.2f}cm  rect_f={rect_f:.1f}px  rect_b={rect_b:.2f}cm")
        print(f"    reproj_thresh={self.tri.MAX_REPROJ_ERR}px  "
              f"epipolar_thresh={self.tri.MAX_EPIPOLAR_ERR}px")

        try:
            self.tri.start_cameras(self.fw, self.fh)
        except RuntimeError as e:
            print(f"ERROR opening cameras: {e}")
            return

        if not self.warmup():
            self.tri.stop_cameras()
            cv2.destroyAllWindows()
            return

        self.pred = TrajectoryPredictor(
            buffer_size=15, min_points=6, velocity_method='regression',
            gravity=981.0, y_down=True, enable_drag=True,
            camera_pitch_deg=20.0)

        print("--- Ready! Toss ball to begin tracking ---\n")

        try:
            while True:
                result = self.tri.update()
                if result['left_frame'] is None:
                    continue

                has_det = (result['left_detection'] is not None or
                           result['right_detection'] is not None)

                # --- Ball found in 3D ---
                if result['found_3d']:
                    x, y, z = result['position_3d']

                    if not self._throw_active:
                        self._start_throw()

                    self._diag_frame += 1
                    disp = result.get('disparity') or 0
                    reproj = result.get('reproj_err') or 0

                    accepted = self.pred.add_position(x, y, z)

                    if accepted:
                        t_now = self.pred.position_buffer.get_latest()['t']
                        self._throw_positions.append(
                            (float(x), float(y), float(z), float(t_now)))
                        self._lost_count = 0

                        # Track t0 for relative time display
                        if self._throw_t0 is None:
                            self._throw_t0 = t_now
                        t_rel = t_now - self._throw_t0

                        # Inter-frame dt
                        dt_ms = ((t_now - self._prev_accept_t) * 1000
                                 if self._prev_accept_t else 0.0)
                        self._prev_accept_t = t_now

                        # Snapshot RT velocity at frame SNAP_AFTER
                        n_pts = len(self._throw_positions)
                        if (not self._rt_snapped and
                                n_pts >= self.SNAP_AFTER and
                                self.pred.is_ready()):
                            self._rt_velocity = self.pred.get_velocity()
                            self._rt_corrected = self.pred._get_corrected_velocity()
                            self._rt_snap_time = t_rel
                            self._predictor_was_ready = True
                            self._rt_snapped = True

                        vel = self.pred.get_velocity()
                        status = "OK"
                        if vel['valid']:
                            corrected = self.pred._get_corrected_velocity()
                            status += (f" vel=({vel['vx']:+.0f},"
                                       f"{corrected[1]:+.0f},{vel['vz']:+.0f})")
                            # Store velocity snapshot for convergence analysis
                            self._velocity_snapshots.append({
                                't': t_rel,
                                'n_pts': n_pts,
                                'vx': vel['vx'],
                                'vy_raw': vel['vy'],
                                'vy_corr': corrected[1],
                                'vz': vel['vz'],
                            })
                        bounces = self.pred._bounce_count
                        status += f" [buf={len(self.pred.position_buffer)}"
                        if bounces > 0:
                            status += f" B={bounces}"
                        status += "]"

                        print(f"  {self._diag_frame:6d}  {t_rel:8.4f}  {dt_ms:5.1f}  "
                              f"{x:8.1f}  {y:8.1f}  {z:8.1f}  |  "
                              f"{disp:5.1f}  {reproj:5.1f}  |  {status}")
                    else:
                        status = f"REJ: {self.pred._last_reject_reason}"
                        print(f"  {self._diag_frame:6d}  {'---':>8}  {'---':>5}  "
                              f"{x:8.1f}  {y:8.1f}  {z:8.1f}  |  "
                              f"{disp:5.1f}  {reproj:5.1f}  |  {status}")

                elif has_det and self._throw_active:
                    reason = result.get('reject_reason', 'no_match')
                    self._diag_frame += 1
                    print(f"  {self._diag_frame:6d}  {'---':>8}  {'---':>5}  "
                          f"{'---':>8}  {'---':>8}  {'---':>8}  |  "
                          f"{'---':>5}  {'---':>5}  |  STEREO FAIL: {reason}")

                # --- Ball lost ---
                if not result['found_3d']:
                    if self._throw_active:
                        self._lost_count += 1
                        if (self._lost_count >= self.LOST_THRESHOLD and
                                len(self._throw_positions) > 0):
                            self._end_throw()

                # --- Display ---
                display = self._build_display(result)
                cv2.imshow('Velocity Validation', display)

                # --- Keys ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_throw()
                elif key == ord('b'):
                    self.tri.reset_background()
                    self._reset_throw()
                    self._chart_img = None
                    print("\n  [BG RESET]")
                    self.warmup()
                elif key == ord('p'):
                    self._print_summary()

        except KeyboardInterrupt:
            pass
        finally:
            self.tri.stop_cameras()
            cv2.destroyAllWindows()

        if self._history:
            self._print_summary()
        print("\nDone!")


def main():
    VelocityValidator().run()


if __name__ == '__main__':
    main()
