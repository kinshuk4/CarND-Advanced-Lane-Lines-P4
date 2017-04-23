import cv2
import numpy as np

COLOR_SPACES = ['RGB', 'HLS', 'HSV']


def convert_color(img, dest_color_space='HLS'):
    if dest_color_space == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif dest_color_space == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif dest_color_space == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif dest_color_space == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif dest_color_space == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif dest_color_space == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif dest_color_space == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img


def split_channels(img, color_space='HLS'):
    converted_image = convert_color(img, dest_color_space=color_space)
    image_as_np_array = converted_image.astype(np.float)
    ch1 = image_as_np_array[:, :, 0]
    ch2 = image_as_np_array[:, :, 1]
    ch3 = image_as_np_array[:, :, 2]
    return ch1, ch2, ch3


def get_channel(img, color_space='HLS', channel=2):
    ch1, ch2, ch3 = split_channels(img, color_space=color_space)
    if channel is 0:
        return ch1
    elif channel is 1:
        return ch2
    elif channel is 2:
        return ch3


DEFAULT_HLS_S_THRES = (100, 255)
DEFAULT_HLS_L_THRES = (220, 255)
DEFAULT_LAB_B_THRES = (190, 255)


def hls_select(img, thres=DEFAULT_HLS_S_THRES, channel=2):
    '''
        Channel 2 means we are selecting s
    '''
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S < thres[0]) | (S > thres[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def hls_sthreshold(img, thres=(125, 255)):
    return hls_select(img, thres=thres, channel=2)


def hls_lthreshold(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:, :, 1]
    hls_l = hls_l * (255 / np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def bgr_select(img, thres=(0, 255), channel=2):
    '''
        Channel 2 means we are selecting R
    '''
    # 1) Convert to HLS color space
    R = img[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = R.copy()
    binary_output[(R < thres[0]) | (R > thres[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def yCrCb_select(img, thresh=(0, 255), channel=0):
    # 1) Convert to HLS color space
    S = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, channel]
    # 2) Apply a threshold to the S channel
    binary_output = S.copy()
    binary_output[(S < thresh[0]) | (S > thresh[1])] = 0
    # 3) Return a binary image of threshold result
    return binary_output


def lab_select(img, thres=(90, 255), channel=2):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    # channel = 2 for B
    B = lab[:, :, channel]

    # don't normalize if there are no yellows in the image
    if np.max(B) > 175:
        lab_b = B * (255 / np.max(B))

    binary = np.zeros_like(B)
    binary[(B > thres[0]) & (B <= thres[1])] = 1
    return binary


def lab_bthreshold(img, thres=DEFAULT_LAB_B_THRES):
    return lab_select(img, thres=thres, channel=2)
