import cv2
import numpy as np

COLOR_SPACES = ['RGB', 'HLS', 'HSV']


def convert_color(image, dest_color_space='HLS'):
    if dest_color_space == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif dest_color_space == 'YUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif dest_color_space == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif dest_color_space == 'HLS':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif dest_color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif dest_color_space == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def split_channels(image, color_space='HLS'):
    converted_image = convert_color(image, dest_color_space=color_space)
    image_as_np_array = converted_image.astype(np.float)
    ch1 = image_as_np_array[:, :, 0]
    ch2 = image_as_np_array[:, :, 1]
    ch3 = image_as_np_array[:, :, 2]
    return ch1, ch2, ch3


def hls_select(img, thres=(0, 255), channel=2):
    '''
        Channel 2 means we are selecting s
    '''
    # 1) Convert to HLS color space
    S = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = S.copy()
    binary_output[(S < thres[0]) | (S > thres[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def bgr_select(image, thres=(0, 255), channel=2):
    '''
        Channel 2 means we are selecting R
    '''
    # 1) Convert to HLS color space
    R = image[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = R.copy()
    binary_output[(R < thres[0]) | (R > thres[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def yCrCb_select(image, thresh=(0, 255), channel=0):
    # 1) Convert to HLS color space
    S = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = S.copy()
    binary_output[(S < thresh[0]) | (S > thresh[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def lab_select(image, thresh=(90, 255), channel=2):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # channel = 2 for B
    B = lab[:, :, channel]
    binary = np.zeros_like(B)
    binary[(B > thresh[0]) & (B <= thresh[1])] = 1
    return binary
