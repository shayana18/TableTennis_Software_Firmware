"""
Test Stereo Ball Detection

Runs the stereo tracker and displays:
- Left and right camera views with ball detection
- 3D position coordinates
- Disparity and depth information

Usage:
    python test_stereo_detection.py

Controls:
    'q' - Quit
    'r' - Reset tracker state
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking.stereo_tracker import StereoTracker


def main():
    # Config path
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "stereo_config.yaml"

    print("=" * 50)
    print("STEREO BALL DETECTION TEST")
    print("=" * 50)
    print(f"\nConfig: {config_path}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset tracker\n")

    # Initialize stereo tracker
    try:
        tracker = StereoTracker(str(config_path))
        print("Stereo tracker initialized")

        if not tracker.is_calibrated:
            print("\nWARNING: Extrinsics not calibrated!")
            print("3D positions will be inaccurate.")
            print("Run calibrate_stereo_extrinsics.py first.\n")
        else:
            print(f"Baseline: {tracker.baseline_mm:.1f} mm\n")

    except Exception as e:
        print(f"ERROR: Failed to initialize tracker: {e}")
        return

    # Start cameras
    try:
        tracker.start_cameras()
        print("Cameras started")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("\nRunning... Show the orange ball to both cameras.\n")

    while True:
        # Get stereo detection
        result = tracker.update()

        if result['left_frame'] is None:
            continue

        # Draw detections on frames
        left_display, right_display = tracker.draw_detections(
            result['left_frame'],
            result['right_frame'],
            result
        )

        # Add velocity info if available
        velocity = tracker.get_velocity_3d()
        if velocity and result['found']:
            vx, vy, vz = velocity
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            vel_text = f"Vel: ({vx:.0f}, {vy:.0f}, {vz:.0f}) mm/f | Speed: {speed:.0f}"
            cv2.putText(left_display, vel_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # Add labels
        cv2.putText(left_display, "LEFT", (10, left_display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(right_display, "RIGHT", (10, right_display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Combine views side by side
        combined = np.hstack([left_display, right_display])

        # Resize if too wide
        h, w = combined.shape[:2]
        if w > 1920:
            scale = 1920 / w
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Stereo Detection", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            print("Tracker reset")

    # Cleanup
    tracker.stop_cameras()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
