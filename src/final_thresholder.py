import cv2
import numpy as np

import src.color_thresholder as cth
import src.sobel_thresholder as sth

ABS_THRES = (0, 255)
MAG_THRES = (0, 255)
DIR_THRES = np.pi / 2
SOBEL_KERNEL = 15


def grad_color_threshold(img, s_thres=(170, 255), sx_thres=(20, 100)):
    """
    Applies varies transformation on an image and combines them in one binary output
    Parameters
    ----------
    image : numpy array
        The image to process
    Returns
    -------
    image : numpy array
        The binary output image
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Apply sobel x to L channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thres[0]) & (scaled_sobel <= sx_thres[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thres[0]) & (s_channel <= s_thres[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary
