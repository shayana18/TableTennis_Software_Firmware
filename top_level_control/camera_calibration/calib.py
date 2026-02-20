import cv2 as cv
import glob
import numpy as np
import sys
import time
from scipy import linalg
import yaml
import os
from datetime import datetime

# Add parent dir so we can import shared config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.camera_config import (
    configure_camera, CAMERA_LEFT_ID, CAMERA_RIGHT_ID, FRAME_WIDTH, FRAME_HEIGHT
)

# This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

# Store RMSE values for later saving
calibration_rmse = {
    'camera0_intrinsic': None,
    'camera1_intrinsic': None,
    'stereo': None
}


def print_camera_settings(camera_name, settings):
    """
    Pretty print camera settings for verification.
    Uses the dict returned by configure_camera() from camera_config.py.
    """
    print("\n" + "="*60)
    print(f"CAMERA SETTINGS: {camera_name}")
    print("="*60)
    print(f"  Resolution:  {settings['width']}x{settings['height']}")
    print(f"  FOURCC:      {settings['fourcc']}")
    print(f"  FPS:         {settings['fps']}")
    if settings.get('trigger_mode'):
        trigger_status = "ON" if settings.get('trigger_ok') else "FAILED"
        print(f"  Trigger:     {trigger_status}")

    if settings['settings_match']:
        print("  STATUS: Settings accepted correctly!")
    else:
        print("  STATUS: WARNING - Settings may not match requested values!")
    print("="*60 + "\n")

    return settings['settings_match']


def save_rmse_to_file(rmse_value, filepath, description=""):
    """
    Save RMSE value to a separate file for easy reference.
    """
    with open(filepath, 'w') as f:
        f.write(f"# {description}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RMSE: {rmse_value}\n")
    print(f"  Saved RMSE ({rmse_value:.6f}) to: {filepath}")


# Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]


# Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings:', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    # Camera settings come from camera_config.py (single source of truth)
    calibration_settings['camera0'] = CAMERA_LEFT_ID
    calibration_settings['camera1'] = CAMERA_RIGHT_ID
    calibration_settings['frame_width'] = FRAME_WIDTH
    calibration_settings['frame_height'] = FRAME_HEIGHT

    # Print loaded settings for verification
    print("\n" + "="*60)
    print("LOADED CALIBRATION SETTINGS")
    print("="*60)
    for key, value in calibration_settings.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")


# Open camera stream and save frames
def save_frames_single_camera(camera_name):
    # Create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    # Get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Open video stream
    cap = cv.VideoCapture(camera_device_id)
    
    # Configure camera with MJPG and verify settings
    settings = configure_camera(cap, width, height)
    settings_ok = print_camera_settings(camera_name, settings)
    
    if not settings_ok:
        print("WARNING: Camera settings don't match requested values.")
        print("         Calibration may still work but verify image quality.")
        user_input = input("         Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            cap.release()
            quit()
    
    cooldown = cooldown_time
    start = False
    countdown_active = False
    countdown_start = 0
    COUNTDOWN_SECONDS = 10
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        if not start and not countdown_active:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        if countdown_active and not start:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining <= 0:
                start = True
                countdown_active = False
                print("\n[GO] Starting frame collection!")
            else:
                cv.putText(frame_small, f"Starting in {int(remaining)+1}...", (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (0,165,255), 3)
                cv.putText(frame_small, "Get checkerboard in position!", (50,120), cv.FONT_HERSHEY_COMPLEX, 1, (0,165,255), 2)

        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            # Save the frame when cooldown reaches 0
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)

        if k == 27:
            quit()

        if k == 32 and not start and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            print(f"\n[COUNTDOWN] {COUNTDOWN_SECONDS} seconds to get checkerboard in position...")

        if saved_count == number_to_save:
            break

    cap.release()
    cv.destroyAllWindows()


# Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix, camera_name="camera"):
    global calibration_rmse
    
    images_names = glob.glob(images_prefix)
    
    if len(images_names) == 0:
        print(f"ERROR: No images found matching pattern: {images_prefix}")
        quit()

    # Read all frames
    images = [cv.imread(imname, 1) for imname in images_names]
    
    print(f"\nCalibrating {camera_name} with {len(images)} images...")

    # Criteria used by checkerboard pattern detector
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # Coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    # Frame dimensions
    width = images[0].shape[1]
    height = images[0].shape[0]
    
    print(f"  Image dimensions: {width}x{height}")

    # Pixel coordinates of checkerboards
    imgpoints = []
    objpoints = []

    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            conv_size = (11, 11)
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print(f'  Skipping frame {i}')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"  WARNING: Could not find checkerboard in frame {i}")

    cv.destroyAllWindows()
    
    if len(objpoints) < 3:
        print("ERROR: Not enough valid calibration frames (need at least 3)")
        quit()
    
    print(f"  Using {len(objpoints)} valid frames for calibration...")
    
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    
    # Store RMSE for later
    if 'camera0' in camera_name:
        calibration_rmse['camera0_intrinsic'] = ret
    elif 'camera1' in camera_name:
        calibration_rmse['camera1_intrinsic'] = ret
    
    print(f'\n  {camera_name} Intrinsic Calibration Results:')
    print(f'  RMSE: {ret:.6f}')
    print(f'  Camera matrix:\n{cmtx}')
    print(f'  Distortion coeffs: {dist}')

    return cmtx, dist, ret


# Save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, rmse_value):
    # Create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')
    outf.close()
    
    print(f"  Saved intrinsics to: {out_filename}")
    
    # Save RMSE to separate file
    rmse_filename = os.path.join('camera_parameters', camera_name + '_intrinsics_rmse.dat')
    save_rmse_to_file(rmse_value, rmse_filename, f"{camera_name} Intrinsic Calibration RMSE")


# Open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):
    # Create frames directory
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    # Settings for taking data
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']

    # Open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    # Configure both cameras with MJPG and verify settings
    print("\nConfiguring cameras for stereo frame capture...")
    settings0 = configure_camera(cap0, width, height)
    settings1 = configure_camera(cap1, width, height)
    
    settings0_ok = print_camera_settings(camera0_name, settings0)
    settings1_ok = print_camera_settings(camera1_name, settings1)
    
    if not (settings0_ok and settings1_ok):
        print("WARNING: One or both camera settings don't match requested values.")
        user_input = input("         Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            cap0.release()
            cap1.release()
            quit()

    cooldown = cooldown_time
    start = False
    countdown_active = False
    countdown_start = 0
    COUNTDOWN_SECONDS = 10
    saved_count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start and not countdown_active:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

        if countdown_active and not start:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining <= 0:
                start = True
                countdown_active = False
                print("\n[GO] Starting frame collection!")
            else:
                cv.putText(frame0_small, f"Starting in {int(remaining)+1}...", (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (0,165,255), 3)
                cv.putText(frame0_small, "Get checkerboard in position!", (50,120), cv.FONT_HERSHEY_COMPLEX, 1, (0,165,255), 2)
                cv.putText(frame1_small, f"Starting in {int(remaining)+1}...", (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (0,165,255), 3)
                cv.putText(frame1_small, "Get checkerboard in position!", (50,120), cv.FONT_HERSHEY_COMPLEX, 1, (0,165,255), 2)

        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            if cooldown <= 0:
                savename = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        if k == 27:
            quit()

        if k == 32 and not start and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
            print(f"\n[COUNTDOWN] {COUNTDOWN_SECONDS} seconds to get checkerboard in position...")

        if saved_count == number_to_save:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


# Open paired calibration frames and stereo calibrate for cam0 to cam1 coordinate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    global calibration_rmse
    
    # Read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))
    
    if len(c0_images_names) == 0 or len(c1_images_names) == 0:
        print("ERROR: No stereo calibration images found!")
        quit()

    # Open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    
    print(f"\nStereo calibrating with {len(c0_images)} image pairs...")

    # Criteria for calibration
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # Coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    # Frame dimensions
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for i, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print(f'  Skipping frame pair {i}')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
        else:
            print(f"  WARNING: Could not find checkerboard in frame pair {i}")

    cv.destroyAllWindows()
    
    if len(objpoints) < 3:
        print("ERROR: Not enough valid stereo calibration frame pairs (need at least 3)")
        quit()
    
    print(f"  Using {len(objpoints)} valid frame pairs for stereo calibration...")
    
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, 
        mtx0, dist0, mtx1, dist1, 
        (width, height), 
        criteria=criteria, 
        flags=stereocalibration_flags
    )
    
    # Store RMSE
    calibration_rmse['stereo'] = ret

    print(f'\n  Stereo Calibration Results:')
    print(f'  RMSE: {ret:.6f}')
    print(f'  Rotation matrix R:\n{R}')
    print(f'  Translation vector T:\n{T.flatten()}')
    
    return R, T, ret


# Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P


# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P


# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
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

    # Define coordinate axes in 3D space
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift

    # Project 3D points to each camera view
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    # Open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    # Configure cameras with MJPG
    settings0 = configure_camera(cap0, width, height)
    settings1 = configure_camera(cap1, width, height)
    print_camera_settings(camera0_name + " (verification)", settings0)
    print_camera_settings(camera1_name + " (verification)", settings1)

    print("\nShowing calibration verification - press ESC to exit")
    print("RGB axes should appear correctly oriented in both views")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        colors = [(0,0,255), (0,255,0), (255,0,0)]
        
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27:
            break

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()


def get_world_space_origin(cmtx, dist, img_path):
    frame = cv.imread(img_path, 1)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _ = cv.Rodrigues(rvec)

    return R, tvec


def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):
    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, stereo_rmse=None, prefix=''):
    # Create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()
    
    print(f"  Saved extrinsics to: {camera0_rot_trans_filename}")

    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()
    
    print(f"  Saved extrinsics to: {camera1_rot_trans_filename}")
    
    # Save stereo RMSE to separate file
    if stereo_rmse is not None:
        rmse_filename = os.path.join('camera_parameters', prefix + 'stereo_calibration_rmse.dat')
        save_rmse_to_file(stereo_rmse, rmse_filename, "Stereo Calibration RMSE")

    return R0, T0, R1, T1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()
    
    print("\n" + "="*60)
    print("STEREO CAMERA CALIBRATION PIPELINE")
    print("Optimized for Arducam OV9782 Global Shutter Cameras")
    print("="*60)
    
    # Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])

    """Step1. Save calibration frames for single cameras"""
    print("\n>>> STEP 1: Capturing mono calibration frames")
    save_frames_single_camera('camera0')
    save_frames_single_camera('camera1')

    """Step2. Obtain camera intrinsic matrices and save them"""
    print("\n>>> STEP 2: Computing intrinsic parameters")
    # Camera0 intrinsics
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0, rmse0 = calibrate_camera_for_intrinsic_parameters(images_prefix, 'camera0') 
    save_camera_intrinsics(cmtx0, dist0, 'camera0', rmse0)
    
    # Camera1 intrinsics
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1, rmse1 = calibrate_camera_for_intrinsic_parameters(images_prefix, 'camera1')
    save_camera_intrinsics(cmtx1, dist1, 'camera1', rmse1)

    """Step3. Save calibration frames for both cameras simultaneously"""
    print("\n>>> STEP 3: Capturing stereo calibration frames")
    save_frames_two_cams('camera0', 'camera1')

    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    print("\n>>> STEP 4: Computing stereo calibration (extrinsics)")
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T, stereo_rmse = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    """Step5. Save calibration data where camera0 defines the world space origin."""
    print("\n>>> STEP 5: Saving calibration parameters")
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T, stereo_rmse)
    R1 = R
    T1 = T
    
    # Check calibration visually
    print(">>> STEP 6: Visual verification")
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)

    print("\nCalibration complete! Files saved to 'camera_parameters/' directory.")