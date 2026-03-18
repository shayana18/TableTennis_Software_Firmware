"""
Simple live trajectory comparison.

Pipeline:
  1. StereoTriangulator finds the ball in camera frame
  2. cam_to_robot converts the point to robot frame
  3. RobotPredictor runs twice on the same points:
     - blue: Kalman-filtered state
     - red: legacy regression + raw-position start state
  4. Both predicted trajectories are drawn on a Y-Z plot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from comm_function.points_based_transform import (
    load_points_based_transform,
    cam_to_robot,
)
from config.camera_config import load_camera_settings
from tracking.stereo_triangulator import StereoTriangulator
from trajectory.ball_state_estimation import BallStateEstimator3D
from trajectory.robot_predictor import RobotPredictor
from trajectory.workspace import (
    ELLIPSE_B,
    Z_MIN,
    Z_MAX,
    Z_TABLE_SURFACE,
    MAX_BOUNCES,
)


def _print(*args, **kwargs):
    print("from planner in terminal", *args, **kwargs)


class TrajectoryComparison:
    PLOT_W = 900
    PLOT_H = 540
    PAD = 50
    VIEW_Y_MAX = 1800.0
    VIEW_Y_MIN = -250.0
    VIEW_Z_MAX = -200.0
    VIEW_Z_MIN = Z_TABLE_SURFACE - 80.0

    def __init__(self, warmup_s: float = 2.0) -> None:
        self.base_dir = PARENT_DIR
        self.calibration_dir = os.path.join(
            self.base_dir, "camera_calibration", "camera_parameters"
        )

        cam = load_camera_settings()
        self.frame_width = cam["frame_width"]
        self.frame_height = cam["frame_height"]
        self.cam_left_id = cam["camera0"]
        self.cam_right_id = cam["camera1"]

        self.triangulator: Optional[StereoTriangulator] = None
        self.predictor_kf = RobotPredictor()
        self.predictor_legacy = self._make_legacy_predictor()
        self.warmup_s = max(0.0, float(warmup_s))
        self.tracking_enabled = True

        # The comparison is only valid if blue really runs the KF path.
        if not isinstance(self.predictor_kf.state_estimator, BallStateEstimator3D):
            raise RuntimeError(
                "BallStateEstimator3D is unavailable. Install filterpy first."
            )

        tf = load_points_based_transform()
        self.R = tf["rotation"]
        self.t_vec = tf["translation"]
        self.cam_scale = tf["camera_scale_to_robot_units"]

    @staticmethod
    def _make_legacy_predictor() -> RobotPredictor:
        """Return a predictor forced onto the legacy regression path."""
        predictor = RobotPredictor()
        predictor.state_estimator = None
        predictor._using_state_estimator = False
        predictor.velocity = None
        return predictor

    @staticmethod
    def _sample_trajectory(predictor: RobotPredictor) -> list[tuple[float, float, float]]:
        """Sample a forward trajectory from the predictor's current state."""
        if not predictor.is_ready():
            return []

        state = predictor._get_prediction_state()
        if state is None:
            return []

        # Reuse the live predictor physics so the plot matches interception logic.
        x, y, z, vx, vy, vz = state
        points = [(x, y, z)]
        step = predictor.SCAN_DT
        t = 0.0
        bounces = 0

        while t < predictor.SCAN_DURATION:
            x_prev, y_prev, z_prev = x, y, z
            x, y, z, vx, vy, vz = RobotPredictor._step_euler(
                x, y, z, vx, vy, vz, step
            )

            if bounces < MAX_BOUNCES:
                x, y, z, vx, vy, vz, did_bounce = RobotPredictor._apply_bounce(
                    x_prev, y_prev, z_prev, x, y, z, vx, vy, vz, step
                )
                if did_bounce:
                    bounces += 1

            points.append((x, y, z))
            t += step

        return points

    @classmethod
    def _to_plot_px(cls, y_mm: float, z_mm: float) -> tuple[int, int]:
        """Map robot-frame Y/Z to plot pixels."""
        inner_w = cls.PLOT_W - 2 * cls.PAD
        inner_h = cls.PLOT_H - 2 * cls.PAD

        u = cls.PAD + (cls.VIEW_Y_MAX - y_mm) * inner_w / (cls.VIEW_Y_MAX - cls.VIEW_Y_MIN)
        v = cls.PAD + (cls.VIEW_Z_MAX - z_mm) * inner_h / (cls.VIEW_Z_MAX - cls.VIEW_Z_MIN)

        u = int(round(np.clip(u, cls.PAD, cls.PLOT_W - cls.PAD)))
        v = int(round(np.clip(v, cls.PAD, cls.PLOT_H - cls.PAD)))
        return u, v

    @classmethod
    def _draw_polyline(
        cls,
        canvas: np.ndarray,
        points_xyz: list[tuple[float, float, float]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a Y-Z polyline from robot-frame points."""
        if len(points_xyz) < 2:
            return
        pts = np.array(
            [cls._to_plot_px(y, z) for _, y, z in points_xyz],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

    @classmethod
    def _draw_points(
        cls,
        canvas: np.ndarray,
        points_xyz: list[tuple[float, float, float, float]],
    ) -> None:
        """Draw raw robot-frame measurements on the Y-Z plot."""
        for _, y, z, _ in points_xyz:
            px, py = cls._to_plot_px(y, z)
            cv2.circle(canvas, (px, py), 3, (220, 220, 220), -1, cv2.LINE_AA)

    @classmethod
    def _make_plot(
        cls,
        raw_points: list[tuple[float, float, float, float]],
        traj_kf: list[tuple[float, float, float]],
        traj_legacy: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """Build the trajectory comparison canvas."""
        canvas = np.full((cls.PLOT_H, cls.PLOT_W, 3), 18, dtype=np.uint8)

        cv2.rectangle(
            canvas,
            (cls.PAD, cls.PAD),
            (cls.PLOT_W - cls.PAD, cls.PLOT_H - cls.PAD),
            (70, 70, 70),
            1,
        )

        table_y = cls._to_plot_px(0.0, Z_TABLE_SURFACE)[1]
        z_min_y = cls._to_plot_px(0.0, Z_MIN)[1]
        z_max_y = cls._to_plot_px(0.0, Z_MAX)[1]
        y_robot_x = cls._to_plot_px(0.0, Z_TABLE_SURFACE)[0]
        y_edge_x = cls._to_plot_px(ELLIPSE_B, Z_TABLE_SURFACE)[0]

        cv2.line(canvas, (cls.PAD, table_y), (cls.PLOT_W - cls.PAD, table_y), (70, 120, 70), 1)
        cv2.line(canvas, (cls.PAD, z_min_y), (cls.PLOT_W - cls.PAD, z_min_y), (60, 60, 110), 1)
        cv2.line(canvas, (cls.PAD, z_max_y), (cls.PLOT_W - cls.PAD, z_max_y), (60, 60, 110), 1)
        cv2.line(canvas, (y_robot_x, cls.PAD), (y_robot_x, cls.PLOT_H - cls.PAD), (80, 80, 80), 1)
        cv2.line(canvas, (y_edge_x, cls.PAD), (y_edge_x, cls.PLOT_H - cls.PAD), (55, 55, 55), 1)

        cls._draw_points(canvas, raw_points)
        cls._draw_polyline(canvas, traj_legacy, (0, 0, 255), 2)
        cls._draw_polyline(canvas, traj_kf, (255, 0, 0), 2)

        cv2.putText(canvas, "Y-Z Trajectory View (robot frame)", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(canvas, "Blue: KF trajectory", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(canvas, "Red: legacy trajectory", (20, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "White: measured points", (20, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(canvas, f"Table Z={Z_TABLE_SURFACE:.0f} mm", (cls.PLOT_W - 240, table_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 160, 70), 1)
        cv2.putText(canvas, "toward robot", (cls.PLOT_W - 150, cls.PLOT_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(canvas, "Z", (10, cls.PAD - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(canvas, "Y", (cls.PLOT_W - 18, cls.PLOT_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return canvas

    def check_calibration(self) -> bool:
        """Ensure required calibration files exist."""
        required = [
            "camera0_intrinsics.dat",
            "camera1_intrinsics.dat",
            "camera0_rot_trans.dat",
            "camera1_rot_trans.dat",
        ]
        for name in required:
            if not os.path.exists(os.path.join(self.calibration_dir, name)):
                _print(f"ERROR: Missing {name}")
                return False
        return True

    def init_triangulator(self) -> bool:
        """Create the stereo triangulator."""
        try:
            self.triangulator = StereoTriangulator(
                calibration_dir=self.calibration_dir,
                cam_left_id=self.cam_left_id,
                cam_right_id=self.cam_right_id,
            )
            return True
        except Exception as exc:
            _print(f"ERROR init triangulator: {exc}")
            return False

    def warmup_background(self) -> bool:
        """Learn the static background before tracking."""
        if self.warmup_s <= 0.0:
            return True

        _print("Remove ball. Learning background (SPACE=skip)...")
        t0 = time.time()
        while time.time() - t0 < self.warmup_s:
            if not self.triangulator.cap_left.grab():
                continue
            if not self.triangulator.cap_right.grab():
                continue
            _, fl = self.triangulator.cap_left.retrieve()
            _, fr = self.triangulator.cap_right.retrieve()
            if fl is None or fr is None:
                continue

            self.triangulator.build_background(fl, fr)
            vis = cv2.resize(fl, (640, int(640 * self.frame_height / self.frame_width)))
            progress = min((time.time() - t0) / self.warmup_s, 1.0)
            h = vis.shape[0]
            bw = int(progress * (vis.shape[1] - 40))
            cv2.rectangle(vis, (20, h - 30), (20 + bw, h - 15), (0, 255, 255), -1)
            cv2.putText(vis, f"BG: {progress*100:.0f}%", (20, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("Trajectory Compare", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                return False

        _print("Background ready.")
        return True

    def reset_predictors(self) -> None:
        """Reset both predictors to the same empty state."""
        self.predictor_kf.reset()
        self.predictor_legacy.reset()

    def run(self) -> None:
        """Run the live comparison loop."""
        _print("\n" + "=" * 60)
        _print(" TRAJECTORY COMPARISON ")
        _print("=" * 60)
        _print(f"Cameras: L={self.cam_left_id} R={self.cam_right_id}")
        _print("Blue = KF state estimate")
        _print("Red  = legacy regression + raw start state")
        _print("Controls: g=toggle  r=reset  b=bg reset  q=quit")
        _print("=" * 60)

        if not self.check_calibration():
            return

        if not self.init_triangulator():
            return

        try:
            self.triangulator.start_cameras(self.frame_width, self.frame_height)
        except Exception as exc:
            _print(f"ERROR starting cameras: {exc}")
            return

        if not self.warmup_background():
            self.triangulator.stop_cameras()
            cv2.destroyAllWindows()
            return

        fps_t0 = time.time()
        fps = 0.0
        frame_count = 0

        try:
            while True:
                result = self.triangulator.update()
                if result["left_frame"] is None:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30.0 / max(1e-6, time.time() - fps_t0)
                    fps_t0 = time.time()

                robot_pos = None
                if self.tracking_enabled and result["found_3d"]:
                    cx, cy, cz = result["position_3d"]
                    frame_ts = result.get("capture_time", time.perf_counter())
                    rx, ry, rz = cam_to_robot(self.R, self.t_vec, self.cam_scale, cx, cy, cz)
                    robot_pos = (rx, ry, rz)
                    self.predictor_kf.add_position(rx, ry, rz, frame_ts)
                    self.predictor_legacy.add_position(rx, ry, rz, frame_ts)

                traj_kf = self._sample_trajectory(self.predictor_kf)
                traj_legacy = self._sample_trajectory(self.predictor_legacy)

                left_vis, right_vis = self.triangulator.draw_results(result)
                state_kf = self.predictor_kf._get_prediction_state()
                state_legacy = self.predictor_legacy._get_prediction_state()

                cv2.putText(left_vis, f"FPS:{fps:.0f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.putText(left_vis, f"Track:{'ON' if self.tracking_enabled else 'OFF'}", (90, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.putText(left_vis, f"KF:{'READY' if self.predictor_kf.is_ready() else '--'}", (190, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                cv2.putText(left_vis, f"LEG:{'READY' if self.predictor_legacy.is_ready() else '--'}", (290, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                if robot_pos is not None:
                    cv2.putText(left_vis,
                                f"Robot(mm): X={robot_pos[0]:+.0f} Y={robot_pos[1]:+.0f} Z={robot_pos[2]:+.0f}",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

                if state_kf is not None:
                    cv2.putText(left_vis,
                                f"KF vy={state_kf[4]:+.0f} vz={state_kf[5]:+.0f}",
                                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)
                if state_legacy is not None:
                    cv2.putText(left_vis,
                                f"Legacy vy={state_legacy[4]:+.0f} vz={state_legacy[5]:+.0f}",
                                (10, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

                plot = self._make_plot(list(self.predictor_kf.positions), traj_kf, traj_legacy)
                raw_count = len(self.predictor_kf.positions)
                cv2.putText(plot, f"Samples: {raw_count}", (20, self.PLOT_H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                dw = 640
                dh = int(dw * self.frame_height / self.frame_width)
                stereo_view = cv2.hconcat([
                    cv2.resize(left_vis, (dw, dh)),
                    cv2.resize(right_vis, (dw, dh)),
                ])

                cv2.imshow("Stereo Compare", stereo_view)
                cv2.imshow("Trajectory Compare", plot)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("g"):
                    self.tracking_enabled = not self.tracking_enabled
                    if self.tracking_enabled:
                        self.reset_predictors()
                        _print("[GATE] ON")
                    else:
                        _print("[GATE] OFF")
                if key == ord("r"):
                    self.reset_predictors()
                    _print("[RESET] Predictors reset")
                if key == ord("b"):
                    self.reset_predictors()
                    self.triangulator.reset_background()
                    _print("[BG RESET] Re-learning...")
                    if not self.warmup_background():
                        break

        except KeyboardInterrupt:
            pass
        finally:
            if self.triangulator is not None:
                try:
                    self.triangulator.stop_cameras()
                except Exception:
                    pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple live KF vs legacy trajectory comparison"
    )
    parser.add_argument("--warmup-s", type=float, default=2.0)
    args = parser.parse_args()

    try:
        app = TrajectoryComparison(warmup_s=args.warmup_s)
    except RuntimeError as exc:
        _print(f"ERROR: {exc}")
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
