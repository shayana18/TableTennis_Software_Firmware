"""
Stereo Camera Calibration for Basler acA1920-150uc Cameras

Same calibration logic as Arducam version, but uses pypylon SDK for camera access.

CAMERA: Basler acA1920-150uc USB 3.0
    - Resolution: 1920x1200
    - Frame Rate: up to 150fps
    - Global Shutter

REQUIREMENTS:
    pip install pypylon

USAGE:
    python calib.py calibration_settings.yaml
"""

import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

try:
    from pypylon import pylon
except ImportError:
    print("ERROR: pypylon not installed. Run: pip install pypylon")
    sys.exit(1)


# Global calibration settings
calibration_settings = {}


# ============================================================================
# BASLER CAMERA HELPER FUNCTIONS
# ============================================================================

def get_basler_camera(serial=None, device_index=0):
    """
    Get a Basler camera by serial number or device index.
    
    Args:
        serial: Camera serial number (string) - preferred method
        device_index: Device index if serial not provided
    
    Returns:
        pylon.InstantCamera object
    """
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    
    if len(devices) == 0:
        raise RuntimeError("No Basler cameras found")
    
    # Find by serial
    if serial and serial.strip():
        for device in devices:
            if device.GetSerialNumber() == serial:
                camera = pylon.InstantCamera(tlFactory.CreateDevice(device))
                return camera
        raise RuntimeError(f"Camera with serial '{serial}' not found. "
                          f"Available: {[d.GetSerialNumber() for d in devices]}")
    
    # Find by index
    if device_index >= len(devices):
        raise RuntimeError(f"Device index {device_index} out of range. Found {len(devices)} cameras.")
    
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[device_index]))
    return camera


def configure_basler_camera(camera, width, height):
    """
    Configure Basler camera settings.
    
    Args:
        camera: pylon.InstantCamera object
        width: Desired width
        height: Desired height
    """
    camera.Open()
    
    # Set resolution
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)
    
    # Center ROI
    max_width = camera.WidthMax.GetValue()
    max_height = camera.HeightMax.GetValue()
    offset_x = (max_width - width) // 2
    offset_y = (max_height - height) // 2
    camera.OffsetX.SetValue(offset_x)
    camera.OffsetY.SetValue(offset_y)
    
    # Exposure and gain
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(5000)  # 5ms
    camera.GainAuto.SetValue("Off")
    camera.Gain.SetValue(0)
    
    # Print actual settings
    actual_width = camera.Width.GetValue()
    actual_height = camera.Height.GetValue()
    print(f"    Resolution: {actual_width}x{actual_height}")


def create_basler_converter():
    """Create image format converter for BGR output."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def grab_frame_basler(camera, converter):
    """
    Grab a single frame from Basler camera.
    
    Returns:
        numpy array (BGR) or None
    """
    if not camera.IsGrabbing():
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grab_result.GrabSucceeded():
        image = converter.Convert(grab_result)
        frame = image.GetArray()
        grab_result.Release()
        return frame
    
    grab_result.Release()
    return None


# ============================================================================
# CALIBRATION FUNCTIONS (Same logic as Arducam version)
# ============================================================================

def DLT(P1, P2, point1, point2):
    """Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point."""
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3,0:3]/Vh[3,3]


def parse_calibration_settings_file(filename):
    """Open and load the calibration_settings.yaml file."""
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings:', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file.')
        quit()


def save_frames_single_camera(camera_name):
    """Open camera stream and save calibration frames."""
    
    if not os.path.exists('frames'):
        os.mkdir('frames')

    # Get settings
    camera_serial = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Determine device index from camera name
    device_index = 0 if camera_name == 'camera0' else 1

    # Open Basler camera
    print(f"\n>>> Opening {camera_name}...")
    camera = get_basler_camera(serial=camera_serial if camera_serial else None, 
                                device_index=device_index)
    configure_basler_camera(camera, width, height)
    converter = create_basler_converter()
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        frame = grab_frame_basler(camera, converter)
        if frame is None:
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx=1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            camera.Close()
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    camera.Close()
    cv.destroyAllWindows()


def calibrate_camera_for_intrinsic_parameters(images_prefix):
    """Calibrate single camera to obtain camera intrinsic parameters from saved frames."""
    
    images_names = glob.glob(images_prefix)
    images = [cv.imread(imname, 1) for imname in images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    width = images[0].shape[1]
    height = images[0].shape[0]

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
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    rmse, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', rmse)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist, rmse


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, rmse=None):
    """Save camera intrinsic parameters and RMSE to file."""

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

    if rmse is not None:
        outf.write('reprojection error:\n')
        outf.write(str(rmse) + '\n')

    outf.close()

    # Also save RMSE to separate file for easy access
    rmse_filename = os.path.join('camera_parameters', camera_name + '_intrinsics_rmse.dat')
    with open(rmse_filename, 'w') as f:
        f.write(f'rmse: {rmse}\n' if rmse else 'rmse: N/A\n')
        if rmse is not None:
            quality = "excellent" if rmse < 0.5 else "good" if rmse < 1.0 else "acceptable" if rmse < 2.0 else "poor"
            f.write(f'quality: {quality}\n')


def save_frames_two_cams(camera0_name, camera1_name):
    """Open both cameras and take calibration frames."""
    
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    
    serial0 = calibration_settings[camera0_name]
    serial1 = calibration_settings[camera1_name]

    # Open both Basler cameras
    print("\n>>> Opening cameras for stereo capture...")
    cam0 = get_basler_camera(serial=serial0 if serial0 else None, device_index=0)
    cam1 = get_basler_camera(serial=serial1 if serial1 else None, device_index=1)
    
    configure_basler_camera(cam0, width, height)
    configure_basler_camera(cam1, width, height)
    
    converter = create_basler_converter()

    cooldown = cooldown_time
    start = False
    saved_count = 0
    
    while True:
        frame0 = grab_frame_basler(cam0, converter)
        frame1 = grab_frame_basler(cam1, converter)

        if frame0 is None or frame1 is None:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
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

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cam0.Close()
    cam1.Close()
    cv.destroyAllWindows()


def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    """Open paired calibration frames and stereo calibrate for cam0 to cam1 coordinate transformations."""
    
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
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
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria=criteria, flags=stereocalibration_flags)

    print('rmse:', ret)
    cv.destroyAllWindows()
    return R, T


def _make_homogeneous_rep_matrix(R, t):
    """Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix."""
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P


def get_projection_matrix(cmtx, R, T):
    """Turn camera calibration data into projection matrix."""
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P


def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.):
    """After calibrating, we can see shifted coordinate axes in the video feeds directly."""
    
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

    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    draw_axes_points = 5 * coordinate_points + z_shift

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

    # Open Basler cameras
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    serial0 = calibration_settings[camera0_name]
    serial1 = calibration_settings[camera1_name]
    
    cam0 = get_basler_camera(serial=serial0 if serial0 else None, device_index=0)
    cam1 = get_basler_camera(serial=serial1 if serial1 else None, device_index=1)
    configure_basler_camera(cam0, width, height)
    configure_basler_camera(cam1, width, height)
    converter = create_basler_converter()

    while True:
        frame0 = grab_frame_basler(cam0, converter)
        frame1 = grab_frame_basler(cam1, converter)

        if frame0 is None or frame1 is None:
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

    cam0.Close()
    cam1.Close()
    cv.destroyAllWindows()


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    """Save extrinsic calibration parameters."""
    
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

    return R0, T0, R1, T1


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python calib.py calibration_settings.yaml"')
        quit()
    
    parse_calibration_settings_file(sys.argv[1])

    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0')
    save_frames_single_camera('camera1')

    """Step2. Obtain camera intrinsic matrices and save them"""
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0, rmse0 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx0, dist0, 'camera0', rmse0)

    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1, rmse1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, 'camera1', rmse1)

    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1')

    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)

    """Step5. Save calibration data where camera0 defines the world space origin."""
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T)
    R1 = R; T1 = T
    
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift=60.)