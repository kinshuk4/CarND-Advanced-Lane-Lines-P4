import cv2
import numpy as np

DEFAULT_NX = 9
DEFAULT_NY = 6


def unwarp_image(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    unwarped = cv2.warpPerspective(img, M, (h, w), flags=cv2.INTER_LINEAR)
    return unwarped, M, Minv


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, mtx, dist, nx=DEFAULT_NX, ny=DEFAULT_NY):
    """
    Args:
        mtx: camera matrix
    """
    # remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    warped = None
    M = None
    Minv = None

    if ret is True:
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv


def transform(img, src_corners, dst_corners, is_gray=False):
    if is_gray is False:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    img_size = (gray.shape[1], gray.shape[0])

    src = np.float32(src_corners)
    dst = np.float32(dst_corners)

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Use this to revert transformation
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    binary_warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
    return binary_warped, Minv


def transform_with_offset(img, src_corners, offset=(300, 0), is_gray=False):
    offset_x = offset[0]  # offset for dst points
    offset_y = offset[1]
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # For source points I'm grabbing the outer four detected corners

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst_corners = [
        [offset_x, offset_y],  # top left
        [img_size[0] - offset_x, offset_y],  # top right
        [img_size[0] - offset_x, img_size[1] - offset_y],  # bottom right
        [offset_x, img_size[1] - offset_y]  # bottom left
    ]

    return transform(img, src_corners, dst_corners, is_gray=is_gray)
