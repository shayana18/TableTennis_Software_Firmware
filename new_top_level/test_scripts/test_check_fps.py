"""
Measure real delivered stereo frame-pair FPS for the current pipeline.

Computation logic:
1. Both cameras are hardware-synced, so each successful left/right grab-retrieve
   cycle is treated as one frame pair.
2. We still grab and retrieve both cameras on every cycle to keep the pipeline
   behavior realistic and to verify that both cameras are delivering frames.
3. The timestamp associated with a frame pair is taken immediately after the
   RIGHT camera retrieve completes.
4. Because the right retrieve happens after the left retrieve in software, this
   timestamp is a conservative lower-bound arrival time for the pair. It avoids
   overstating delivered FPS.
5. We exclude an initial warmup period from the recorded data so startup effects
   do not skew the measurement.
6. FPS is computed from delivered frame-pair timing, not requested camera FPS:
      fps = (n_pairs - 1) / (last_pair_time - first_pair_time)

The JSON output stores pair-delivery deltas rather than absolute arrival times.
"""

import csv
import sys
import time
from collections import OrderedDict
from pathlib import Path
from statistics import median

import cv2
import yaml


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SETTINGS_PATH = PROJECT_DIR / "camera_params" / "calibration_settings.yaml"
OUTPUT_PATH = PROJECT_DIR / "camera_params" / "camera_properties" / "fps_timings.csv"
CAPTURE_DURATION_S = 120.0
WARMUP_DURATION_S = 2.0
DEFAULT_FPS = 100
STALL_TIMEOUT_S = 5.0


def load_settings():
    with SETTINGS_PATH.open("r", encoding="utf-8") as infile:
        settings = yaml.safe_load(infile)

    required = ("camera0", "camera1", "frame_width", "frame_height")
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"Missing keys in {SETTINGS_PATH}: {missing}")

    return settings


def fourcc_to_str(value):
    value = int(value)
    chars = [(value >> (8 * i)) & 0xFF for i in range(4)]
    return "".join(chr(c) if 32 <= c <= 126 else "." for c in chars)


def open_camera(device_id, width, height, fps):
    if sys.platform.startswith("win"):
        backend_candidates = (
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
            ("DEFAULT", None),
        )
    elif sys.platform == "darwin":
        backend_candidates = (
            ("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION),
            ("DEFAULT", None),
        )
    else:
        backend_candidates = (
            ("CAP_V4L2", getattr(cv2, "CAP_V4L2", None)),
            ("DEFAULT", None),
        )

    for backend_name, backend in backend_candidates:
        cap = cv2.VideoCapture(device_id) if backend is None else cv2.VideoCapture(device_id, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        warmed = False
        for _ in range(10):
            ok, _ = cap.read()
            if ok:
                warmed = True

        if warmed:
            return cap, backend_name

        cap.release()

    raise RuntimeError(f"Failed to open camera {device_id}")


def calculate_fps_from_sum(pair_count, total_delta_ns):
    if pair_count < 2:
        return 0.0

    if total_delta_ns <= 0:
        return 0.0

    return (pair_count - 1) * 1_000_000_000.0 / total_delta_ns


def summarize_deltas_ms(delta_ms):
    if not delta_ms:
        return OrderedDict(
            [
                ("interval_count", 0),
                ("mean_delta_ms", 0.0),
                ("median_delta_ms", 0.0),
                ("min_delta_ms", 0.0),
                ("max_delta_ms", 0.0),
            ]
        )

    mean_ms = sum(delta_ms) / len(delta_ms)
    return OrderedDict(
        [
            ("interval_count", len(delta_ms)),
            ("mean_delta_ms", round(mean_ms, 6)),
            ("median_delta_ms", round(median(delta_ms), 6)),
            ("min_delta_ms", round(min(delta_ms), 6)),
            ("max_delta_ms", round(max(delta_ms), 6)),
        ]
    )


def build_pair_summary(pair_count, pair_indices, pair_deltas_ms, total_delta_ns, failure_count):
    pair_fps = calculate_fps_from_sum(pair_count, total_delta_ns)
    return OrderedDict(
        [
            ("frame_pair_count", pair_count),
            ("successful_cycles", pair_count),
            ("failed_cycles", failure_count),
            ("calculated_pair_fps", round(pair_fps, 6)),
            ("pair_delta_summary", summarize_deltas_ms(pair_deltas_ms)),
            ("frame_pair_indices", pair_indices),
            ("frame_pair_deltas_ms", pair_deltas_ms),
        ]
    )


def write_csv(output_path, summary, pair_summary, camera0_info, camera1_info):
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)

        writer.writerow(["section", "key", "value"])
        for key, value in summary.items():
            writer.writerow(["fps_summary", key, value])

        for key, value in camera0_info.items():
            writer.writerow(["camera0", key, value])

        for key, value in camera1_info.items():
            writer.writerow(["camera1", key, value])

        for key, value in pair_summary.items():
            if key in {"frame_pair_indices", "frame_pair_deltas_ms"}:
                continue
            writer.writerow(["pair_delivery", key, value])

        writer.writerow([])
        writer.writerow(["frame_pair_index", "frame_delta_ms"])
        for frame_pair_index, frame_delta_ns in zip(
            pair_summary["frame_pair_indices"][1:],
            pair_summary["frame_pair_deltas_ms"],
        ):
            writer.writerow([frame_pair_index, frame_delta_ns])


def capture_frame_pair(left_cap, right_cap):
    """
    Capture one stereo pair.

    Use only grab/retrieve so the measurement reflects the intended stereo
    capture path. Rare transient misses are counted as failed cycles and do not
    invalidate the run unless they become a sustained stall.

    Returns:
        (success, pair_arrival_ns, method, failure_stage)
    """
    grabbed_left = left_cap.grab()
    grabbed_right = right_cap.grab()

    if not grabbed_left or not grabbed_right:
        return False, None, None, "grab_failed"

    ok_left, _ = left_cap.retrieve()
    if not ok_left:
        return False, None, None, "retrieve_left"

    ok_right, _ = right_cap.retrieve()
    pair_arrival_ns = time.perf_counter_ns()
    if not ok_right:
        return False, None, None, "retrieve_right"

    return True, pair_arrival_ns, "grab_retrieve", None


def main():
    settings = load_settings()
    width = int(settings["frame_width"])
    height = int(settings["frame_height"])
    target_fps = int(settings.get("camera_fps", DEFAULT_FPS))
    left_id = int(settings["camera0"])
    right_id = int(settings["camera1"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    left_cap = None
    right_cap = None

    try:
        left_cap, left_backend = open_camera(left_id, width, height, target_fps)
        right_cap, right_backend = open_camera(right_id, width, height, target_fps)

        run_start_ns = time.perf_counter_ns()
        warmup_end_ns = run_start_ns + int(WARMUP_DURATION_S * 1_000_000_000)
        capture_end_ns = warmup_end_ns + int(CAPTURE_DURATION_S * 1_000_000_000)

        pair_indices = []
        pair_deltas_ms = []
        pair_counter = 0
        failed_cycles = 0
        last_success_ns = run_start_ns
        previous_recorded_pair_ns = None
        total_delta_ns = 0
        method_counts = OrderedDict(
            [
                ("grab_retrieve", 0),
            ]
        )
        failure_counts = OrderedDict(
            [
                ("grab_failed", 0),
                ("retrieve_left", 0),
                ("retrieve_right", 0),
                ("stall_timeout", 0),
            ]
        )

        while time.perf_counter_ns() < capture_end_ns:
            success, pair_arrival_ns, method, failure_stage = capture_frame_pair(left_cap, right_cap)

            if not success:
                failed_cycles += 1
                if failure_stage in failure_counts:
                    failure_counts[failure_stage] += 1

                if time.perf_counter_ns() - last_success_ns > int(STALL_TIMEOUT_S * 1_000_000_000):
                    failure_counts["stall_timeout"] += 1
                    raise RuntimeError(
                        f"No successful frame pairs received for {STALL_TIMEOUT_S:.1f} seconds"
                    )

                time.sleep(0.001)
                continue

            method_counts[method] += 1
            last_success_ns = pair_arrival_ns
            pair_counter += 1

            if pair_arrival_ns < warmup_end_ns:
                continue

            pair_indices.append(pair_counter)
            if previous_recorded_pair_ns is not None:
                delta_ns = pair_arrival_ns - previous_recorded_pair_ns
                total_delta_ns += delta_ns
                pair_deltas_ms.append(round(delta_ns / 1_000_000.0, 6))
            previous_recorded_pair_ns = pair_arrival_ns

        summary = OrderedDict(
            [
                ("compute_logic", "Each successful left/right cycle is treated as one hardware-synced frame pair. The pair timestamp is taken immediately after the right-camera retrieve completes, making it a conservative lower-bound arrival time for the pair. FPS is computed from pair-delivery timing after excluding the warmup period."),
                ("warmup_duration_seconds", WARMUP_DURATION_S),
                ("capture_duration_seconds", CAPTURE_DURATION_S),
                ("requested_width", width),
                ("requested_height", height),
                ("requested_fps", target_fps),
                ("camera0_device_id", left_id),
                ("camera1_device_id", right_id),
                ("camera0_backend", left_backend),
                ("camera1_backend", right_backend),
                ("run_start_perf_counter_ns", run_start_ns),
                ("warmup_end_perf_counter_ns", warmup_end_ns),
                ("capture_end_perf_counter_ns", time.perf_counter_ns()),
                ("frame_pair_fps", round(calculate_fps_from_sum(len(pair_indices), total_delta_ns), 6)),
                ("frame_pair_count", len(pair_indices)),
                ("failed_cycles", failed_cycles),
                ("capture_methods", method_counts),
                ("failure_counts", failure_counts),
            ]
        )

        pair_summary = build_pair_summary(
            pair_count=len(pair_indices),
            pair_indices=pair_indices,
            pair_deltas_ms=pair_deltas_ms,
            total_delta_ns=total_delta_ns,
            failure_count=failed_cycles,
        )
        camera0_info = OrderedDict(
            [
                ("device_id", left_id),
                ("backend", left_backend),
                ("reported_width", int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                ("reported_height", int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                ("reported_fps", float(left_cap.get(cv2.CAP_PROP_FPS))),
                ("reported_fourcc", fourcc_to_str(left_cap.get(cv2.CAP_PROP_FOURCC))),
            ]
        )
        camera1_info = OrderedDict(
            [
                ("device_id", right_id),
                ("backend", right_backend),
                ("reported_width", int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                ("reported_height", int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                ("reported_fps", float(right_cap.get(cv2.CAP_PROP_FPS))),
                ("reported_fourcc", fourcc_to_str(right_cap.get(cv2.CAP_PROP_FOURCC))),
            ]
        )

        write_csv(OUTPUT_PATH, summary, pair_summary, camera0_info, camera1_info)

        print(f"frame-pair delivered fps: {summary['frame_pair_fps']:.6f}")
        print(f"frame pairs recorded: {summary['frame_pair_count']}")
        print(f"failed cycles during run: {summary['failed_cycles']}")
        print(f"Saved timings to: {OUTPUT_PATH}")
    finally:
        if left_cap is not None:
            left_cap.release()
        if right_cap is not None:
            right_cap.release()


if __name__ == "__main__":
    main()
