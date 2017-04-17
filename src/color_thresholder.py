import cv2
import numpy as np

class ColorThresholder:
    @staticmethod
    def hls_threshold(img, thresh=(0, 255), channel=2):
        '''
        Channel 2 means we are selecting s
        '''
        # 1) Convert to HLS color space
        S = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,channel]
        # 2) Apply a threshold to the S channel
        binary_output = S.copy()
        binary_output[(S < thresh[0]) | (S > thresh[1])] = 0
        # 3) Return a binary image of threshold result
        return binary_output
    @staticmethod
    def red_select(img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        R = img[:,:,2]
        # 2) Apply a threshold to the S channel
        binary_output = R.copy()
        binary_output[(S < thresh[0]) | (S > thresh[1])] = 0
        # 3) Return a binary image of threshold result
        return binary_output
    @staticmethod
    def yCrCb_select(img, thresh=(0, 255), channel=0):
        # 1) Convert to HLS color space
        S = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:,:,channel]
        # 2) Apply a threshold to the S channel
        binary_output = S.copy()
        binary_output[(S < thresh[0]) | (S > thresh[1])] = 0
        # 3) Return a binary image of threshold result
        return binary_output
    @staticmethod
    def grad_color_threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
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
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Apply sobel x to L channel
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        return color_binary