"""
Standalone Extrinsic Calibration (Extrinsics Only)

Loads existing intrinsic parameters and runs ONLY the stereo calibration steps.
Use this when intrinsics are already calibrated but you need to redo extrinsics.

All settings come from calibration_settings.yaml — no camera_config dependency.
"""

import cv2 as cv
import glob
import numpy as np
import sys
import time
from scipy import linalg
import yaml
import os

calibration_settings = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
CAMERA_PROPERTIES_DIR = os.path.join(PROJECT_DIR, 'camera_params', 'camera_properties')


# ===============================
# DLT (unused here but kept from your original)
# ===============================
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3,0:3]/Vh[3,3]


# ===============================
# FOURCC debug helpers
# ===============================
def fourcc_debug(v):
    v = int(v)
    b = [(v >> (8*i)) & 0xFF for i in range(4)]
    s = "".join([chr(x) if 32 <= x <= 126 else "." for x in b])
    return s, f"{v:#010x}"


# ===============================
# Settings parser
# ===============================
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()

    print('Using calibration settings:', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if 'camera0' not in calibration_settings.keys() or 'camera1' not in calibration_settings.keys():
        print('camera0/camera1 keys not found in the settings file.')
        quit()

    print(f"  camera0: {calibration_settings['camera0']}")
    print(f"  camera1: {calibration_settings['camera1']}")
    print(f"  resolution: {calibration_settings['frame_width']}x{calibration_settings['frame_height']}")
    if 'camera_fps' in calibration_settings:
        print(f"  camera_fps: {calibration_settings['camera_fps']}")


# ===============================
# Improved camera open (debug + retry)
# ===============================
def open_camera(device_id, width, height, fps=100):
    print(f"\nOpening camera {device_id}...")

    cap = cv.VideoCapture(device_id, cv.CAP_DSHOW)

    # Force MJPG first
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Then resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # Then FPS
    cap.set(cv.CAP_PROP_FPS, fps)

    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Let driver settle
    last = None
    ret = False
    for _ in range(10):
        ret, last = cap.read()

    if not ret:
        print(f"Camera {device_id}: failed to read frames.")
        quit()

    actual_h, actual_w = last.shape[:2]
    fcc_s, fcc_hex = fourcc_debug(cap.get(cv.CAP_PROP_FOURCC))
    fps_rb = cap.get(cv.CAP_PROP_FPS)
    width_rb = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height_rb = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print("  FIRST APPLY RESULT:")
    print(f"    requested: {width}x{height}@{fps} MJPG")
    print(f"    actual frame: {actual_w}x{actual_h}")
    print(f"    readback: {width_rb}x{height_rb}")
    print(f"    fps readback (often unreliable): {fps_rb}")
    print(f"    fourcc printable: {fcc_s}  fourcc hex: {fcc_hex}")

    # Retry once if it didn't stick
    if (actual_w != width) or (actual_h != height) or (fcc_s != "MJPG"):
        print("  Retrying format apply...")

        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS, fps)

        for _ in range(10):
            ret, last = cap.read()

        if not ret:
            print(f"Camera {device_id}: failed to read frames after retry.")
            quit()

        actual_h, actual_w = last.shape[:2]
        fcc_s, fcc_hex = fourcc_debug(cap.get(cv.CAP_PROP_FOURCC))
        fps_rb = cap.get(cv.CAP_PROP_FPS)
        width_rb = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height_rb = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        print("  AFTER RETRY:")
        print(f"    actual frame: {actual_w}x{actual_h}")
        print(f"    readback: {width_rb}x{height_rb}")
        print(f"    fps readback (often unreliable): {fps_rb}")
        print(f"    fourcc printable: {fcc_s}  fourcc hex: {fcc_hex}")

    return cap


# ===============================
# Load intrinsics
# ===============================
def load_camera_intrinsics(camera_name):
    filepath = os.path.join(CAMERA_PROPERTIES_DIR, camera_name + '_intrinsics.dat')

    if not os.path.exists(filepath):
        print(f'ERROR: Intrinsic file not found: {filepath}')
        print('Run full calibration first to generate intrinsics.')
        quit()

    cmtx = []
    dist = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    reading_intrinsic = False
    reading_distortion = False

    for line in lines:
        line = line.strip()
        if line == 'intrinsic:':
            reading_intrinsic = True
            reading_distortion = False
            continue
        elif line == 'distortion:':
            reading_intrinsic = False
            reading_distortion = True
            continue

        if reading_intrinsic and line:
            cmtx.append([float(x) for x in line.split()])
        elif reading_distortion and line:
            dist = [float(x) for x in line.split()]

    cmtx = np.array(cmtx, dtype=np.float64)
    dist = np.array([dist], dtype=np.float64)

    print(f'Loaded {camera_name} intrinsics from {filepath}')
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist


# ===============================
# Capture frames with LIVE measured FPS overlay
# ===============================
def save_frames_two_cams(camera0_name, camera1_name):
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    target_fps = calibration_settings.get('camera_fps', 100)

    cap0 = open_camera(calibration_settings[camera0_name], width, height, fps=target_fps)
    cap1 = open_camera(calibration_settings[camera1_name], width, height, fps=target_fps)

    # Rolling FPS measurement (measured via timestamps)
    t0_last = None
    t1_last = None
    fps0 = 0.0
    fps1 = 0.0
    alpha = 0.10  # smoothing factor

    saved = 0
    number_to_save = calibration_settings['stereo_calibration_frames']

    print("\nPreview running.")
    print("  SPACE = save frame pair")
    print("  ESC   = quit\n")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Camera read failed.")
            break

        # Timestamp-based FPS (per camera)
        now0 = time.perf_counter()
        if t0_last is not None:
            inst0 = 1.0 / max(1e-9, (now0 - t0_last))
            fps0 = (1 - alpha) * fps0 + alpha * inst0
        t0_last = now0

        now1 = time.perf_counter()
        if t1_last is not None:
            inst1 = 1.0 / max(1e-9, (now1 - t1_last))
            fps1 = (1 - alpha) * fps1 + alpha * inst1
        t1_last = now1

        # Overlay info
        h0, w0 = frame0.shape[:2]
        h1, w1 = frame1.shape[:2]
        fcc0, hx0 = fourcc_debug(cap0.get(cv.CAP_PROP_FOURCC))
        fcc1, hx1 = fourcc_debug(cap1.get(cv.CAP_PROP_FOURCC))

        overlay0 = [
            f"Cam0 idx={calibration_settings[camera0_name]}",
            f"Measured FPS: {fps0:5.1f}",
            f"Res: {w0}x{h0}",
            f"FOURCC: {fcc0} ({hx0})",
            f"Saved pairs: {saved}/{number_to_save}",
        ]
        overlay1 = [
            f"Cam1 idx={calibration_settings[camera1_name]}",
            f"Measured FPS: {fps1:5.1f}",
            f"Res: {w1}x{h1}",
            f"FOURCC: {fcc1} ({hx1})",
            f"Saved pairs: {saved}/{number_to_save}",
        ]

        y = 25
        for line in overlay0:
            cv.putText(frame0, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            y += 25

        y = 25
        for line in overlay1:
            cv.putText(frame1, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            y += 25

        cv.imshow("cam0", frame0)
        cv.imshow("cam1", frame1)

        k = cv.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break

        if k == 32:  # SPACE
            cv.imwrite(f"frames_pair/camera0_{saved}.png", frame0)
            cv.imwrite(f"frames_pair/camera1_{saved}.png", frame1)
            print(f"Saved frame pair {saved}")
            saved += 1

            if saved >= number_to_save:
                break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


# ===============================
# Stereo calibration
# ===============================
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    if len(c0_images_names) == 0 or len(c1_images_names) == 0:
        print("No images found in frames_pair/. Did you capture frames?")
        quit()

    if len(c0_images_names) != len(c1_images_names):
        print("WARNING: number of camera0 and camera1 frames differ.")
        print(f"  camera0 frames: {len(c0_images_names)}")
        print(f"  camera1 frames: {len(c1_images_names)}")

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            # show corners and allow skipping
            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.imshow('img_cam0', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv.imshow('img_cam1', frame1)

            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                print("Skipping this pair")
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()

    if len(objpoints) < 5:
        print(f"Not enough valid chessboard detections: {len(objpoints)}")
        print("Try capturing more frames and/or improving lighting/pattern visibility.")
        quit()

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC

    ret, CM1, dist0_out, CM2, dist1_out, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx0, dist0, mtx1, dist1,
        (width, height),
        criteria=criteria,
        flags=stereocalibration_flags
    )

    print('Stereo calibration RMSE:', ret)
    return R, T


# ===============================
# Projection + visual check (from your original)
# ===============================
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4, 4))
    P[:3, :3] = R
    P[:3, 3] = t.reshape(3)
    P[3, 3] = 1
    return P


def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3, :]
    return P


def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.):
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    coordinate_points = np.array([[0., 0., 0.],
                                  [1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])
    z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift

    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])

        uv = P0 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera0.append(uv)

        uv = P1 @ X
        uv = np.array([uv[0], uv[1]]) / uv[2]
        pixel_points_camera1.append(uv)

    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    target_fps = calibration_settings.get('camera_fps', 100)

    cap0 = open_camera(calibration_settings[camera0_name], width, height, fps=target_fps)
    cap1 = open_camera(calibration_settings[camera1_name], width, height, fps=target_fps)

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            break

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)

        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0_axes', frame0)
        cv.imshow('frame1_axes', frame1)

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


# ===============================
# Save extrinsics (from your original)
# ===============================
def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    os.makedirs(CAMERA_PROPERTIES_DIR, exist_ok=True)

    camera0_rot_trans_filename = os.path.join(CAMERA_PROPERTIES_DIR, prefix + 'camera0_rot_trans.dat')
    with open(camera0_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R0:
            outf.write(" ".join(str(en) for en in l) + '\n')
        outf.write('T:\n')
        for l in T0:
            outf.write(" ".join(str(en) for en in l) + '\n')

    camera1_rot_trans_filename = os.path.join(CAMERA_PROPERTIES_DIR, prefix + 'camera1_rot_trans.dat')
    with open(camera1_rot_trans_filename, 'w') as outf:
        outf.write('R:\n')
        for l in R1:
            outf.write(" ".join(str(en) for en in l) + '\n')
        outf.write('T:\n')
        for l in T1:
            outf.write(" ".join(str(en) for en in l) + '\n')

    return R0, T0, R1, T1


# ===============================
# Main
# ===============================
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python calib_extrinsic_only.py calibration_settings.yaml"')
        quit()

    parse_calibration_settings_file(sys.argv[1])

    print('\n' + '='*60)
    print('EXTRINSIC CALIBRATION ONLY')
    print('Loading existing intrinsics, skipping intrinsic calibration')
    print('='*60)

    print('\n>>> Loading camera0 intrinsics...')
    cmtx0, dist0 = load_camera_intrinsics('camera0')

    print('\n>>> Loading camera1 intrinsics...')
    cmtx1, dist1 = load_camera_intrinsics('camera1')

    print('\n>>> Capturing stereo calibration frames...')
    save_frames_two_cams('camera0', 'camera1')

    print('\n>>> Running stereo calibration...')
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T)
    R1 = R
    T1 = T

    print('\n>>> Extrinsic parameters saved!')
    print(f'    {os.path.join(CAMERA_PROPERTIES_DIR, "camera0_rot_trans.dat")}')
    print(f'    {os.path.join(CAMERA_PROPERTIES_DIR, "camera1_rot_trans.dat")}')

    print('\n>>> Visual calibration check (press ESC to exit)...')
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift=60.)

    print('\n' + '='*60)
    print('EXTRINSIC CALIBRATION COMPLETE!')
    print('='*60)
