"""
Simple bounce-detection logger.

Records one throw as robot-frame raw positions plus KF position/velocity, and
marks the exact accepted frame where RobotPredictor records a bounce.

Output:
    test_scripts/bounce_detection.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CAMERA_PROPERTIES_DIR = os.path.join(PARENT_DIR, "camera_params", "camera_properties")
BOUNCE_DATA_DIR = os.path.join(PARENT_DIR, "test_data", "bounce_data")

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ball_tracking.stereo_triangulator import StereoTriangulator
from comm_functions.points_based_transform import load_points_based_transform, cam_to_robot
from config.camera_config import load_camera_settings
from estimation.robot_predictor import RobotPredictor


REPROJ_ERR_MAX = 10.0
WARMUP_S = 2.0
LOST_FRAMES_TO_STOP = 30


CSV_COLUMNS = [
    "frame_idx",
    "timestamp_s",
    "raw_x_mm",
    "raw_y_mm",
    "raw_z_mm",
    "accepted",
    "reject_reason",
    "kf_x_mm",
    "kf_y_mm",
    "kf_z_mm",
    "vel_x_mm_s",
    "vel_y_mm_s",
    "vel_z_mm_s",
    "kf_ready",
    "kf_updates",
    "buffer_size",
    "bounce_detected",
    "bounce_count",
    "rising_count",
    "z_min_since_reset_mm",
    "bounce_fall_from_first_mm",
]


def _print(*args, **kwargs):
    print("[bounce-test]", *args, **kwargs)


def next_output_csv_path():
    os.makedirs(BOUNCE_DATA_DIR, exist_ok=True)

    prefix = "bounce_detection_session_"
    suffix = ".csv"
    max_session = 0

    for name in os.listdir(BOUNCE_DATA_DIR):
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        session_str = name[len(prefix):-len(suffix)]
        if session_str.isdigit():
            max_session = max(max_session, int(session_str))

    next_session = max_session + 1
    return os.path.join(BOUNCE_DATA_DIR, f"{prefix}{next_session}.csv")


def warmup_background(triangulator, frame_width, frame_height, warmup_s):
    if warmup_s <= 0.0:
        return True

    _print("Remove ball. Learning background...")
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        if not triangulator.cap_left.grab():
            continue
        if not triangulator.cap_right.grab():
            continue

        ok_left, frame_left = triangulator.cap_left.retrieve()
        ok_right, frame_right = triangulator.cap_right.retrieve()
        if not ok_left or not ok_right or frame_left is None or frame_right is None:
            continue

        triangulator.build_background(frame_left, frame_right)

        vis = cv2.resize(frame_left, (640, int(640 * frame_height / frame_width)))
        progress = min((time.time() - t0) / warmup_s, 1.0)
        h = vis.shape[0]
        bar_w = int(progress * (vis.shape[1] - 40))
        cv2.rectangle(vis, (20, h - 30), (20 + bar_w, h - 15), (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"BG: {progress * 100:.0f}%",
            (20, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        cv2.imshow("Bounce Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    _print("Background ready.")
    return True


def save_csv(rows, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _print(f"Saved {len(rows)} rows -> {output_csv}")


def main():
    output_csv = next_output_csv_path()
    cam = load_camera_settings()
    frame_width = cam["frame_width"]
    frame_height = cam["frame_height"]
    cam_left_id = cam["camera0"]
    cam_right_id = cam["camera1"]

    tf = load_points_based_transform()
    rotation = tf["rotation"]
    translation = tf["translation"]
    cam_scale = tf["camera_scale_to_robot_units"]

    predictor = RobotPredictor()
    triangulator = StereoTriangulator(
        calibration_dir=CAMERA_PROPERTIES_DIR,
        cam_left_id=cam_left_id,
        cam_right_id=cam_right_id,
    )

    rows = []
    frame_idx = 0
    prev_bounce_count = 0
    throw_started = False
    lost_frames = 0
    bounce_flash_frames = 0

    triangulator.start_cameras(frame_width, frame_height)
    cv2.namedWindow("Bounce Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bounce Detection", 1280, 800)

    try:
        if not warmup_background(triangulator, frame_width, frame_height, WARMUP_S):
            return

        _print("Tracking live. Throw one ball. Press 'q' to stop, 'r' to reset.")

        while True:
            result = triangulator.update()
            frame_ts = result.get("capture_time", time.perf_counter())

            if result["left_frame"] is None or result["right_frame"] is None:
                continue

            left_vis, right_vis = triangulator.draw_results(result)

            if result["found_3d"]:
                reproj = result.get("reproj_err", 0.0)
                if reproj <= REPROJ_ERR_MAX:
                    cx, cy, cz = result["position_3d"]
                    raw_x, raw_y, raw_z = cam_to_robot(
                        rotation, translation, cam_scale, cx, cy, cz
                    )

                    bounce_count_before = predictor._bounce_count
                    accepted = predictor.add_position(raw_x, raw_y, raw_z, frame_ts)
                    stats = predictor.get_stats()
                    kf_pos = (
                        predictor.state_estimator.get_position()
                        if predictor.state_estimator is not None
                        else predictor.get_current_position()
                    )
                    vel = (
                        predictor.state_estimator.get_velocity()
                        if predictor.state_estimator is not None
                        else predictor.velocity
                    )

                    bounce_count = predictor._bounce_count
                    bounce_recorded = bounce_count > bounce_count_before
                    if bounce_recorded:
                        bounce_flash_frames = 20
                    prev_bounce_count = bounce_count

                    z_min_since_reset = predictor._z_min_since_reset
                    bounce_fall_from_first = ""
                    if predictor.positions and z_min_since_reset is not None:
                        bounce_fall_from_first = round(
                            predictor.positions[0][2] - z_min_since_reset, 3
                        )

                    frame_idx += 1
                    rows.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_s": round(frame_ts, 6),
                            "raw_x_mm": round(raw_x, 3),
                            "raw_y_mm": round(raw_y, 3),
                            "raw_z_mm": round(raw_z, 3),
                            "accepted": accepted,
                            "reject_reason": "" if accepted else predictor._last_reject_reason,
                            "kf_x_mm": round(kf_pos[0], 3) if kf_pos is not None else "",
                            "kf_y_mm": round(kf_pos[1], 3) if kf_pos is not None else "",
                            "kf_z_mm": round(kf_pos[2], 3) if kf_pos is not None else "",
                            "vel_x_mm_s": round(vel[0], 3) if vel is not None else "",
                            "vel_y_mm_s": round(vel[1], 3) if vel is not None else "",
                            "vel_z_mm_s": round(vel[2], 3) if vel is not None else "",
                            "kf_ready": stats["kf_ready"],
                            "kf_updates": stats["kf_updates"],
                            "buffer_size": stats["buffer"],
                            "bounce_detected": bool(bounce_recorded),
                            "bounce_count": bounce_count,
                            "rising_count": predictor._rising_count,
                            "z_min_since_reset_mm": (
                                round(z_min_since_reset, 3)
                                if z_min_since_reset is not None
                                else ""
                            ),
                            "bounce_fall_from_first_mm": bounce_fall_from_first,
                        }
                    )

                    throw_started = True
                    lost_frames = 0
                else:
                    result["reject_reason"] = f"reproj({reproj:.1f}px)"

            if throw_started and not result["found_3d"]:
                lost_frames += 1
                if lost_frames >= LOST_FRAMES_TO_STOP:
                    _print("Lost track of throw. Stopping and saving log.")
                    break

            overlay_lines = [
                f"Logged frames: {len(rows)}",
                f"Predictor buffer: {len(predictor.positions)}",
                f"Bounces: {predictor._bounce_count}",
                f"Rise count: {predictor._rising_count}",
            ]
            if result.get("reject_reason"):
                overlay_lines.append(f"Reject: {result['reject_reason']}")

            for i, text in enumerate(overlay_lines):
                y = 30 + i * 24
                cv2.putText(left_vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(left_vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if bounce_flash_frames > 0:
                cv2.putText(
                    left_vis,
                    "BOUNCE RECORDED",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                bounce_flash_frames -= 1

            left_vis = cv2.resize(left_vis, (640, int(640 * left_vis.shape[0] / left_vis.shape[1])))
            right_vis = cv2.resize(right_vis, (640, int(640 * right_vis.shape[0] / right_vis.shape[1])))
            live_view = np.hstack((left_vis, right_vis))
            cv2.imshow("Bounce Detection", live_view)

            key = cv2.waitKeyEx(1)
            if 0 <= key <= 255:
                key = ord(chr(key).lower())

            if key == ord("q"):
                break
            if key == ord("r"):
                _print("Resetting predictor and current log.")
                predictor.reset()
                rows.clear()
                frame_idx = 0
                prev_bounce_count = 0
                throw_started = False
                lost_frames = 0
                bounce_flash_frames = 0

    finally:
        triangulator.stop_cameras()
        cv2.destroyAllWindows()
        save_csv(rows, output_csv)


if __name__ == "__main__":
    main()
