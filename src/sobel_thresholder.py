import cv2
import numpy as np

'''
Module to group the static methods transforming the images to binary format after thresholding
'''
THRES_RANGE = (0, 255)
DEFAULT_ABS_THRES = None
DEFAULT_MAG_THRES = (50, 200)
DEFAULT_DIR_THRES = (0.7, 1.2)
DEFAULT_SOBEL_KERNEL = None
DEFAULT_MAG_KERNEL = 9
DEFAULT_DIR_KERNEL = 15


def abs_sobel_threshold(img, orient='x', sobel_kernel=DEFAULT_SOBEL_KERNEL, thres=DEFAULT_ABS_THRES, is_gray=False):
    """Apply the absolute Sobel filter.
    Sobel operator detects gradients in x and y directions.
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    if is_gray is True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('orient must be "x" or "y".')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    if thres is not None:
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= thres[0]) & (scaled_sobel <= thres[1])] = 1
        # 6) Return this mask as binary_output image
        binary_output = sbinary
    else:
        binary_output = scaled_sobel
    return binary_output


def mag_threshold(img, sobel_kernel=DEFAULT_MAG_KERNEL, thres=DEFAULT_MAG_THRES, is_gray=False):
    """Apply filter according to magnituide of gradient."""
    # Apply the following steps to img
    # 1) Convert to grayscale
    if is_gray is True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    # 5) Create a binary mask where mag thresholds are met
    binary_gradmag = np.zeros_like(scaled_gradmag)
    binary_gradmag[(scaled_gradmag >= thres[0]) & (scaled_gradmag <= thres[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = binary_gradmag
    return binary_output


def dir_threshold(img, sobel_kernel=DEFAULT_DIR_KERNEL, thres=DEFAULT_DIR_THRES, is_gray=False):
    # Apply the following steps to img
    # 1) Convert to grayscale
    if is_gray is True:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_dir = np.zeros_like(grad_direction)
    binary_dir[(grad_direction >= thres[0]) & (grad_direction <= thres[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = binary_dir  # Remove this line
    return binary_output


def apply_gradient_filters(image, sobel_kernel=DEFAULT_SOBEL_KERNEL, abs_thres=DEFAULT_ABS_THRES,
                           mag_thres=DEFAULT_MAG_THRES,
                           dir_thres=DEFAULT_DIR_THRES, dir_kernel=DEFAULT_DIR_KERNEL, mag_kernel=DEFAULT_MAG_KERNEL,
                           is_gray=False):
    # Choose a Sobel kernel size
    # Choose a larger sobel_kernel as odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=sobel_kernel, thres=abs_thres, is_gray=is_gray)
    grady = abs_sobel_threshold(image, orient='y', sobel_kernel=sobel_kernel, thres=abs_thres, is_gray=is_gray)
    mag_binary = mag_threshold(image, sobel_kernel=mag_kernel, thres=mag_thres, is_gray=is_gray)
    dir_binary = dir_threshold(image, sobel_kernel=dir_kernel, thres=dir_thres, is_gray=is_gray)
    # combine filters
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined
