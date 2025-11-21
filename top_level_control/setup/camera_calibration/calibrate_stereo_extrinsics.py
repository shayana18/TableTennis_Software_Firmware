"""
Stereo Extrinsic Calibration

Captures stereo image pairs of a checkerboard and calculates
the rotation (R) and translation (T) between cameras.

Usage:
    python calibrate_stereo_extrinsics.py

Controls:
    'c' - Capture current stereo pair
    'q' - Quit and run calibration
    'r' - Reset captured pairs
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_intrinsics(calibration_dir):
    """Load camera matrix and distortion from calibration directory."""
    cal_path = Path(calibration_dir)

    camera_matrix = np.loadtxt(cal_path / "camera_matrix.txt").reshape(3, 3)
    distortion = np.loadtxt(cal_path / "distortion_coefficients.txt")

    return camera_matrix, distortion


def main():
    # Configuration
    CHECKERBOARD = (10, 7)  # Inner corners (columns, rows)
    SQUARE_SIZE = 1.5     # Square size in CM

    # Camera IDs - update if needed
    CAM_LEFT_ID = 0
    CAM_RIGHT_ID = 2

    # Paths
    script_dir = Path(__file__).parent.parent.parent
    intrinsics_left = script_dir / "output_cam1"
    intrinsics_right = script_dir / "output_cam2_new"
    output_dir = script_dir / "output_stereo"

    print("=" * 50)
    print("STEREO EXTRINSIC CALIBRATION")
    print("=" * 50)
    print(f"\nCheckerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} corners")
    print(f"Square size: {SQUARE_SIZE} mm")
    print(f"\nControls:")
    print("  'c' - Capture stereo pair")
    print("  'q' - Quit and calibrate")
    print("  'r' - Reset captures")
    print("\nNeed 15-20 pairs from different angles.\n")

    # Load intrinsics
    print("Loading intrinsics...")
    K_left, dist_left = load_intrinsics(intrinsics_left)
    K_right, dist_right = load_intrinsics(intrinsics_right)

    print(f"\nLeft camera ({intrinsics_left}):")
    print(f"  Camera matrix:\n{K_left}")
    print(f"  Distortion: {dist_left}")

    print(f"\nRight camera ({intrinsics_right}):")
    print(f"  Camera matrix:\n{K_right}")
    print(f"  Distortion: {dist_right}\n")

    # Open cameras
    cap_left = cv2.VideoCapture(CAM_LEFT_ID)
    cap_right = cv2.VideoCapture(CAM_RIGHT_ID)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("ERROR: Could not open cameras!")
        return

    # Prepare object points (checkerboard corners in 3D)
    # e.g., (0,0,0), (25,0,0), (50,0,0), ... for 25mm squares
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Storage for calibration data
    obj_points = []     # 3D points
    img_points_left = []   # 2D points in left camera
    img_points_right = []  # 2D points in right camera

    print("Starting capture... Position checkerboard so BOTH cameras see it.\n")

    while True:
        # Capture frames
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            continue

        # Convert to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        found_left, corners_left = cv2.findChessboardCorners(
            gray_left, CHECKERBOARD, None
        )
        found_right, corners_right = cv2.findChessboardCorners(
            gray_right, CHECKERBOARD, None
        )

        # Draw corners for visualization
        display_left = frame_left.copy()
        display_right = frame_right.copy()

        if found_left:
            cv2.drawChessboardCorners(display_left, CHECKERBOARD, corners_left, found_left)
        if found_right:
            cv2.drawChessboardCorners(display_right, CHECKERBOARD, corners_right, found_right)

        # Status text
        status = "READY" if (found_left and found_right) else "Position checkerboard"
        color = (0, 255, 0) if (found_left and found_right) else (0, 0, 255)

        cv2.putText(display_left, f"Left - {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_right, f"Right - {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_left, f"Pairs captured: {len(obj_points)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Resize to same height if different
        h1, w1 = display_left.shape[:2]
        h2, w2 = display_right.shape[:2]
        if h1 != h2:
            target_h = max(h1, h2)
            display_left = cv2.resize(display_left, (int(w1 * target_h / h1), target_h))
            display_right = cv2.resize(display_right, (int(w2 * target_h / h2), target_h))

        # Show side by side
        combined = np.hstack([display_left, display_right])
        # Resize if too large
        h, w = combined.shape[:2]
        if w > 1920:
            scale = 1920 / w
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        cv2.imshow("Stereo Calibration", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture pair
            if found_left and found_right:
                # Refine corners for accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points_left.append(corners_left)
                img_points_right.append(corners_right)

                print(f"  Captured pair {len(obj_points)}")
            else:
                print("  Cannot capture - checkerboard not found in both cameras")

        elif key == ord('r'):
            # Reset
            obj_points = []
            img_points_left = []
            img_points_right = []
            print("\n  Reset - all captures cleared\n")

        elif key == ord('q'):
            break

    # Cleanup cameras
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    # Run calibration
    if len(obj_points) < 5:
        print(f"\nERROR: Need at least 5 pairs, got {len(obj_points)}")
        return

    print(f"\n\nRunning stereo calibration with {len(obj_points)} pairs...")
    print("This may take a moment...\n")

    # Get image size
    h, w = gray_left.shape

    # Stereo calibration
    # This calculates R and T between the cameras
    flags = cv2.CALIB_FIX_INTRINSIC  # Use our existing intrinsics

    ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        K_left, dist_left,
        K_right, dist_right,
        (w, h),
        flags=flags
    )

    # Calculate baseline
    baseline_mm = np.linalg.norm(T)

    print("=" * 50)
    print("CALIBRATION COMPLETE")
    print("=" * 50)
    print(f"\nRMS Error: {ret:.4f} (lower is better, <1.0 is good)")
    print(f"\nBaseline: {baseline_mm:.2f} mm")
    print(f"\nRotation matrix R:")
    print(R)
    print(f"\nTranslation vector T (mm):")
    print(T.flatten())

    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\nSaving to {output_dir}/...")

    # Save calibration files
    np.savetxt(output_dir / "R.txt", R)
    np.savetxt(output_dir / "T.txt", T.flatten())
    np.savetxt(output_dir / "baseline.txt", [baseline_mm])
    np.savetxt(output_dir / "E.txt", E)
    np.savetxt(output_dir / "F.txt", F)
    np.savetxt(output_dir / "rms_error.txt", [ret])

    print("Files saved:")
    print("  - R.txt (rotation matrix)")
    print("  - T.txt (translation vector)")
    print("  - baseline.txt (distance between cameras)")
    print("  - E.txt (essential matrix)")
    print("  - F.txt (fundamental matrix)")
    print("  - rms_error.txt")
    print("\nYou can now use StereoTracker for 3D ball tracking.")
    print("=" * 50)


if __name__ == "__main__":
    main()
