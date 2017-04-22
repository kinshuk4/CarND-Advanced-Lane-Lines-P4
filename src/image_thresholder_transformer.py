import cv2
import numpy as np

import src.color_thresholder as cth
import src.sobel_thresholder as sth
import src.final_thresholder as fth
import src.perspective_transformer as ppt


def undistort_threshold_transform_image1(img, mtx, dist, corners):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Apply gradient and color filters
    _, combined_binary = fth.grad_color_threshold(img)

    # Transform perspective
    binary_warped, Minv = ppt.transform_with_offset(combined_binary.astype(np.uint8), corners)
    return binary_warped, Minv


def undistort_threshold_transform_image2(img, mtx, dist, corners):
    # Undistort
    img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
    # Perspective Transform
    img_unwarp, Minv = ppt.transform_with_offset(img_undistort, corners, is_gray=True)

    # HLS L-channel Threshold (using default parameters)
    img_LThresh = cth.hls_lthreshold(img_unwarp)

    # Lab B-channel Threshold (using default parameters)
    img_BThresh = cth.lab_bthreshold(img_unwarp)

    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_BThresh)
    combined[(img_LThresh == 1) | (img_BThresh == 1)] = 1
    return combined, Minv, img_undistort
