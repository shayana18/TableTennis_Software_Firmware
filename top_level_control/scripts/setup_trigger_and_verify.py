"""
Setup External Trigger Mode & Verify Sync
==========================================
Enables external trigger mode on BOTH Arducam OV9782 cameras, then
verifies hardware synchronization by measuring frame-pair time deltas.

Run this after plugging in cameras (replaces having to use the AMCap app).

WHAT IT DOES:
  1. Sets MJPG 1280x800@100fps on both cameras
  2. Enables external trigger mode (backlight=ON, manual exposure, exposure=-6)
  3. Runs sync verification (measures L-R frame delta)

NOTE: Trigger mode is now handled by the shared configure_camera() in
      config/camera_config.py. The trigger_mode setting in
      calibration_settings.yaml controls whether it's enabled.

CAMERA: Arducam OV9782 Global Shutter USB Camera
        1MP, 100fps @ 1280x800 MJPG (external trigger: up to 90fps)

USAGE:
    cd top_level_control
    python -m scripts.setup_trigger_and_verify

CONTROLS:
    q - Quit and show results graph
    SPACE - Quit and show results graph
"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.camera_config import load_camera_settings, configure_camera


def main():
    cam_settings = load_camera_settings()
    cam_left_id = cam_settings['camera0']
    cam_right_id = cam_settings['camera1']
    CAMERA_WIDTH = cam_settings['frame_width']
    CAMERA_HEIGHT = cam_settings['frame_height']

    print("=" * 60)
    print("SETUP TRIGGER MODE & VERIFY SYNC")
    print("=" * 60)
    print(f"  Left camera:  ID {cam_left_id}")
    print(f"  Right camera: ID {cam_right_id}")
    print(f"  Target: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ 100fps MJPG")
    print(f"  Trigger mode: {'ON' if cam_settings['trigger_mode'] else 'OFF'}")
    print(f"\nOpening cameras...")

    cap_left = cv2.VideoCapture(cam_left_id)
    cap_right = cv2.VideoCapture(cam_right_id)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("ERROR: Failed to open one or both cameras")
        return

    # --- Step 1: Configure resolution + codec + trigger mode ---
    # configure_camera() reads trigger_mode from calibration_settings.yaml
    print(f"\n--- Step 1: Configure cameras (MJPG + trigger mode) ---")
    s_left = configure_camera(cap_left, CAMERA_WIDTH, CAMERA_HEIGHT)
    s_right = configure_camera(cap_right, CAMERA_WIDTH, CAMERA_HEIGHT)

    print(f"  LEFT:  {s_left['width']}x{s_left['height']} @ {s_left['fps']:.0f}fps ({s_left['fourcc']})")
    if s_left.get('trigger_mode'):
        print(f"         Trigger: {'OK' if s_left.get('trigger_ok') else 'FAILED'}")

    print(f"  RIGHT: {s_right['width']}x{s_right['height']} @ {s_right['fps']:.0f}fps ({s_right['fourcc']})")
    if s_right.get('trigger_mode'):
        print(f"         Trigger: {'OK' if s_right.get('trigger_ok') else 'FAILED'}")

    if s_left.get('trigger_ok') and s_right.get('trigger_ok'):
        print(f"\n  External trigger mode enabled on BOTH cameras.")
    elif s_left.get('trigger_mode'):
        print(f"\n  WARNING: Could not confirm trigger mode on one or both cameras.")
        print(f"  The test will still run -- check results to verify sync.")

    # --- Step 2: Sync verification ---
    print(f"\n--- Step 2: Sync verification ---")
    print(f"Collecting frames... Press 'q' or SPACE to stop and see results.")
    print("(Ensure your external trigger signal is connected and running)\n")

    # Data collection
    deltas_ms = []
    trigger_intervals = []
    frame_count = 0
    last_pair_time = None

    # Warm up - discard first few frames
    for _ in range(5):
        cap_left.read()
        cap_right.read()

    try:
        while True:
            t_before_left = time.perf_counter()
            ret_left, frame_left = cap_left.read()
            t_after_left = time.perf_counter()

            t_before_right = time.perf_counter()
            ret_right, frame_right = cap_right.read()
            t_after_right = time.perf_counter()

            if not ret_left or not ret_right:
                continue

            delta = (t_after_right - t_after_left) * 1000
            deltas_ms.append(delta)

            pair_time = (t_after_left + t_after_right) / 2
            if last_pair_time is not None:
                interval = (pair_time - last_pair_time) * 1000
                trigger_intervals.append(interval)
            last_pair_time = pair_time

            frame_count += 1

            # Live display
            display_w = 480
            display_h = int(display_w * s_left['height'] / s_left['width'])
            left_small = cv2.resize(frame_left, (display_w, display_h))
            right_small = cv2.resize(frame_right, (display_w, display_h))

            cv2.putText(left_small, f"Frame #{frame_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(left_small, f"L-R delta: {delta:.2f}ms", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if len(trigger_intervals) > 0:
                fps_actual = 1000.0 / trigger_intervals[-1] if trigger_intervals[-1] > 0 else 0
                cv2.putText(left_small, f"Trigger rate: {fps_actual:.1f}fps", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if len(deltas_ms) > 10:
                avg = np.mean(deltas_ms[-50:])
                std = np.std(deltas_ms[-50:])
                sync_status = "SYNCED" if std < 2.0 else "CHECK SYNC"
                color = (0, 255, 0) if std < 2.0 else (0, 0, 255)
                cv2.putText(right_small, f"Avg delta: {avg:.2f}ms", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(right_small, f"Std dev:   {std:.2f}ms", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(right_small, sync_status, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(right_small, "Press 'q' to see results", (10, display_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            combined = cv2.hconcat([left_small, right_small])
            cv2.imshow('Trigger Setup & Sync Verify', combined)

            if frame_count % 50 == 0:
                avg = np.mean(deltas_ms)
                std = np.std(deltas_ms)
                print(f"  Frame {frame_count:4d} | "
                      f"Avg delta: {avg:.2f}ms | "
                      f"Std: {std:.2f}ms | "
                      f"Min: {min(deltas_ms):.2f}ms | "
                      f"Max: {max(deltas_ms):.2f}ms")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord(' '):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    # --- Results ---
    if len(deltas_ms) < 10:
        print(f"\nOnly {len(deltas_ms)} frames collected - not enough data.")
        return

    deltas = np.array(deltas_ms)
    intervals = np.array(trigger_intervals) if trigger_intervals else np.array([0])

    print("\n" + "=" * 60)
    print("SYNC VERIFICATION RESULTS")
    print("=" * 60)
    print(f"\nFrames collected: {frame_count}")
    print(f"\nLeft-Right Frame Delta (ms):")
    print(f"  Mean:   {np.mean(deltas):7.2f} ms")
    print(f"  Std:    {np.std(deltas):7.2f} ms")
    print(f"  Min:    {np.min(deltas):7.2f} ms")
    print(f"  Max:    {np.max(deltas):7.2f} ms")
    print(f"  Median: {np.median(deltas):7.2f} ms")

    if len(intervals) > 1:
        effective_fps = 1000.0 / np.mean(intervals)
        print(f"\nTrigger Interval (ms):")
        print(f"  Mean:   {np.mean(intervals):7.2f} ms  ({effective_fps:.1f} fps)")
        print(f"  Std:    {np.std(intervals):7.2f} ms")
        print(f"  Min:    {np.min(intervals):7.2f} ms")
        print(f"  Max:    {np.max(intervals):7.2f} ms")

    print(f"\n{'=' * 60}")
    if np.std(deltas) < 2.0 and np.max(deltas) < 10.0:
        print("RESULT: CAMERAS ARE SYNCED")
        print("  Low, consistent delta = hardware trigger is working.")
    elif np.std(deltas) < 5.0:
        print("RESULT: LIKELY SYNCED (minor jitter)")
        print("  Small variation is normal from USB scheduling.")
    else:
        print("RESULT: SYNC ISSUE DETECTED")
        print("  High variation suggests cameras are NOT hardware-synced.")
        print("  Check: trigger wiring, external trigger mode enabled on BOTH cameras.")
    print("=" * 60)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Trigger Setup & Sync Verification', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(deltas, linewidth=0.8, color='#2196F3')
        ax.axhline(y=np.mean(deltas), color='red', linestyle='--', linewidth=1,
                    label=f'Mean: {np.mean(deltas):.2f}ms')
        ax.set_xlabel('Frame #')
        ax.set_ylabel('L-R Delta (ms)')
        ax.set_title('Frame Pair Time Delta')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.hist(deltas, bins=50, color='#4CAF50', edgecolor='white', alpha=0.8)
        ax.axvline(x=np.mean(deltas), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean: {np.mean(deltas):.2f}ms')
        ax.axvline(x=np.median(deltas), color='orange', linestyle='--', linewidth=1.5,
                    label=f'Median: {np.median(deltas):.2f}ms')
        ax.set_xlabel('Delta (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Delta Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if len(intervals) > 1:
            ax.plot(intervals, linewidth=0.8, color='#FF9800')
            ax.axhline(y=np.mean(intervals), color='red', linestyle='--', linewidth=1,
                        label=f'Mean: {np.mean(intervals):.1f}ms ({1000/np.mean(intervals):.1f}fps)')
            ax.set_ylabel('Interval (ms)')
            ax.legend()
        ax.set_xlabel('Frame #')
        ax.set_title('Trigger Interval (time between frames)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')

        if np.std(deltas) < 2.0 and np.max(deltas) < 10.0:
            verdict = "SYNCED"
            verdict_color = '#4CAF50'
        elif np.std(deltas) < 5.0:
            verdict = "LIKELY SYNCED"
            verdict_color = '#FF9800'
        else:
            verdict = "SYNC ISSUE"
            verdict_color = '#F44336'

        summary = (
            f"Frames: {frame_count}\n\n"
            f"L-R Delta:\n"
            f"  Mean:   {np.mean(deltas):.2f} ms\n"
            f"  Std:    {np.std(deltas):.2f} ms\n"
            f"  Min:    {np.min(deltas):.2f} ms\n"
            f"  Max:    {np.max(deltas):.2f} ms\n"
        )
        if len(intervals) > 1:
            summary += (
                f"\nTrigger Rate:\n"
                f"  {1000/np.mean(intervals):.1f} fps\n"
            )

        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.5, 0.05, verdict, transform=ax.transAxes,
                fontsize=24, fontweight='bold', color=verdict_color,
                ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\n(matplotlib not available - install it for graphs: pip install matplotlib)")
    except Exception as e:
        print(f"\n(Could not show plot: {e})")


if __name__ == '__main__':
    main()
