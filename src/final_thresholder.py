import cv2
import numpy as np

import src.color_thresholder as cth
import src.sobel_thresholder as sth

ABS_THRES = (0, 255)
MAG_THRES = (0, 255)
DIR_THRES = np.pi / 2
SOBEL_KERNEL = 15

DEFAULT_SX_THRES = (20, 100)
THRES_RANGE = (0, 255)
DEFAULT_ABS_THRES = (50, 200)
DEFAULT_MAG_THRES = (10, 80)
DEFAULT_DIR_THRES = (0.0, 0.3)
DEFAULT_SOBEL_KERNEL = 3
DEFAULT_MAG_KERNEL = 5
DEFAULT_DIR_KERNEL = 5
DEFAULT_HLS_S_THRES = (170, 255)


def grad_color_threshold(img, s_thres=DEFAULT_HLS_S_THRES, sx_thres=DEFAULT_SX_THRES,
                         sobel_kernel=DEFAULT_SOBEL_KERNEL):
    img = np.copy(img)
    img = cv2.GaussianBlur(img, (sobel_kernel, sobel_kernel), 0)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobel_l = sth.abs_sobel_threshold(l_channel, sobel_kernel=sobel_kernel)
    sobel_s = sth.abs_sobel_threshold(s_channel, sobel_kernel=sobel_kernel)

    # combine l and s
    sobel_l_and_s = cv2.bitwise_or(sobel_l, sobel_s)

    # Threshold x gradient
    sxbinary = np.zeros_like(sobel_l_and_s)
    sxbinary[(sobel_l_and_s >= sx_thres[0]) & (sobel_l_and_s <= sx_thres[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thres[0]) & (s_channel <= s_thres[1])] = 1

    # Stack each channel
    color_binary = np.dstack((np.ones_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary
