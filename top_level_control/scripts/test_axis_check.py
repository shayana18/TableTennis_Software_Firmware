"""
Axis Verification — Confirm camera coordinate mapping.

Hold the ball in view and move it in known physical directions.
Watch which axis changes to confirm the mapping:

  EXPECTED (cameras on SIDE of table, looking ACROSS width):
    Move ball along table LENGTH (toward Player A/B) → X changes
    Move ball across table WIDTH (closer/further)    → Z changes
    Move ball UP/DOWN                                → Y changes (down = +Y)

CONTROLS:
    q - Quit
    r - Reset reference point (set current position as origin)
    SPACE - Skip warmup
"""

import cv2
import sys
import os
import time
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tracking.stereo_triangulator import StereoTriangulator
from config.camera_config import load_camera_settings


def main():
    cam = load_camera_settings()
    calib_dir = os.path.join(parent_dir, 'camera_calibration', 'camera_parameters')

    tri = StereoTriangulator(
        calibration_dir=calib_dir,
        cam_left_id=cam['camera0'], cam_right_id=cam['camera1'])

    tri.start_cameras(cam['frame_width'], cam['frame_height'])

    # Warmup
    print("\n  Remove ball from view. Learning background...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        if not tri.cap_left.grab() or not tri.cap_right.grab():
            continue
        _, fl = tri.cap_left.retrieve()
        _, fr = tri.cap_right.retrieve()
        if fl is not None and fr is not None:
            tri.build_background(fl, fr)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            break
        elif k == ord('q'):
            tri.stop_cameras()
            return

    print("  Ready! Hold ball in view and move in known directions.\n")
    print("  EXPECTED MAPPING:")
    print("    Along table LENGTH  → X changes")
    print("    Across table WIDTH  → Z changes")
    print("    UP / DOWN           → Y changes (down = +Y)")
    print()

    ref_pos = None    # reference point for delta display
    prev_pos = None   # previous position for movement arrows
    W, H = 480, 360

    try:
        while True:
            result = tri.update()
            if result['left_frame'] is None:
                continue

            frame = result['left_frame'].copy()
            frame = cv2.resize(frame, (W, H))

            if result['found_3d']:
                x, y, z = result['position_3d']

                # Delta from reference
                if ref_pos is not None:
                    dx = x - ref_pos[0]
                    dy = y - ref_pos[1]
                    dz = z - ref_pos[2]
                else:
                    dx, dy, dz = 0, 0, 0

                # Find dominant axis of movement (from previous frame)
                dominant = ""
                if prev_pos is not None:
                    mx = abs(x - prev_pos[0])
                    my = abs(y - prev_pos[1])
                    mz = abs(z - prev_pos[2])
                    total = mx + my + mz
                    if total > 0.5:  # minimum movement threshold
                        if mx > my and mx > mz:
                            dominant = "X (LENGTH)"
                        elif my > mx and my > mz:
                            dominant = "Y (VERTICAL)"
                        else:
                            dominant = "Z (WIDTH/DEPTH)"
                prev_pos = (x, y, z)

                # -- Draw coordinates --
                # Absolute position
                cv2.putText(frame, f"X: {x:+8.1f} cm", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Y: {y:+8.1f} cm", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Z: {z:+8.1f} cm", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 50), 2)

                # Delta from reference
                if ref_pos is not None:
                    cv2.putText(frame, f"dX:{dx:+6.1f}", (280, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.putText(frame, f"dY:{dy:+6.1f}", (280, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    cv2.putText(frame, f"dZ:{dz:+6.1f}", (280, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 50), 1)

                # Dominant movement axis
                if dominant:
                    cv2.putText(frame, f"Moving: {dominant}", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Reproj quality
                reproj = result.get('reproj_err', 0)
                cv2.putText(frame, f"Reproj: {reproj:.1f}px", (10, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            else:
                cv2.putText(frame, "No 3D detection", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                prev_pos = None

            # Instructions at bottom
            cv2.putText(frame, "LENGTH->X  WIDTH->Z  UP/DOWN->Y",
                        (10, H - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(frame, "q=quit  r=set reference point",
                        (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

            cv2.imshow('Axis Check', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if result['found_3d']:
                    ref_pos = result['position_3d']
                    print(f"  Reference set: X={ref_pos[0]:.1f}  "
                          f"Y={ref_pos[1]:.1f}  Z={ref_pos[2]:.1f}")
                else:
                    print("  No ball detected — can't set reference")

    except KeyboardInterrupt:
        pass
    finally:
        tri.stop_cameras()
        cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == '__main__':
    main()
