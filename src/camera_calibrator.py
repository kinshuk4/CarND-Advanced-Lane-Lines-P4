import numpy as np
import cv2
import glob

DEFAULT_NX = 9
DEFAULT_NY = 6


def find_and_draw_chessboard_corner_for_image(image, nx=DEFAULT_NX, ny=DEFAULT_NY,
                                              is_gray_image=False):
    """
    Load chessboard image files and compute mappings
    between image points and object points.
    Args:
        image: File name pattern of chessboard images.
        nx (int): num_x_points - The number of corners in x-coordinate of a chessboard image.
        ny (int): num_y_points - The number of corners in y-coordinate of a chessboard image.
        is_gray_image: Whether image is already gray scaled or not
    Returns:
        image_corners
    """
    if is_gray_image is False:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    image_corners = None

    # If found, add object points, image points
    if ret is True:
        image_corners = cv2.drawChessboardCorners(image, (9, 6), corners, ret)

    return image_corners


def find_chessboard_corner_for_image(image, nx=DEFAULT_NX, ny=DEFAULT_NY,
                                     is_gray_image=False):
    """
    Load chessboard image files and compute mappings
    between image points and object points.
    Args:
        image: File name pattern of chessboard images.
        nx (int): The number of corners in x-coordinate of a chessboard image.
        ny (int): The number of corners in y-coordinate of a chessboard image.
        is_gray_image: Whether image is already gray scaled or not
    Returns:
        image_corners
    """
    if is_gray_image is False:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    image_corners = None

    # If found, add object points, image points
    if ret is True:
        return corners
    else:
        return None


def find_chessboard_corners(image_file_pattern, nx=DEFAULT_NX, ny=DEFAULT_NY):
    """
    Load chessboard image files and compute mappings
    between image points and object points.
    Args:
        image_file_pattern (str): File name pattern of chessboard images.
        nx (int): The number of corners in x-coordinate of a chessboard image.
        ny (int): The number of corners in y-coordinate of a chessboard image.
    Returns:
        mtx (numpy ndarray): Camera matrix.
        dist (numpy ndarray): Distortion coefficients.
        objpoints: 3d points in real world space
        imgpoints: 2d points in image plane.
    """
    objpoints = []
    imgpoints = []

    image_files = glob.glob(image_file_pattern)

    # prepare object points
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x,y coordinate

    if len(image_files) == 0:
        raise FileNotFoundError('No image files: {}'.format(image_file_pattern))
    image_size = None
    # find chessborad corners for each image
    for image_file in image_files:
        image = cv2.imread(image_file)

        image_size = (image.shape[1], image.shape[0])

        # find chessboard corners
        image_corners = find_chessboard_corner_for_image(image, nx=nx,
                                                         ny=ny)
        if image_corners is not None:
            imgpoints.append(image_corners)
            objpoints.append(objp)
        else:
            print("Failed to calibrate for: " + image_file)

    return objpoints, imgpoints, image_size


def calibrate_camera(image_file_pattern, nx=DEFAULT_NX, ny=DEFAULT_NY):
    objpoints, imgpoints, image_size = find_chessboard_corners(image_file_pattern, nx=nx,
                                                               ny=ny)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return mtx, dist


def undistort_image1(image, objpoints, imgpoints):
    '''
    Undistort and image based on the provided camera calibration factors.
    
    :param image:
        Image (RGB)
        
    :param objpoints:
        Object points calibration factors
        
    :param imgpoints:
        Image points calibration factors
        
    :return:
        Undistorted image
    '''

    image_size = (image.shape[1], image.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    undist = undistort_image(image, mtx, dist)

    return undist


def undistort_image(image, mtx, dist):
    '''
    Undistort and image based on the provided camera calibration factors.
    
    :param image:
        Image (RGB)
        
    :param objp:
        Object points calibration factors
        
    :param imgp:
        Image points calibration factors
        
    :return:
        Undistorted image
    '''

    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist


def undistort_image_file(image_file, mtx, dist):
    '''
    Undistort and image based on the provided camera calibration factors.
    
    :param image:
        Image (RGB)
        
    :param objp:
        Object points calibration factors
        
    :param imgp:
        Image points calibration factors
        
    :return:
        Undistorted image
    '''
    image = cv2.imread(image_file)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    return undist
