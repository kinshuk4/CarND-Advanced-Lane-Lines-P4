import cv2
import numpy as np

class SobelThresholder:
    '''
    Class to group the static methods transforming the images to binary format after thresholding
    '''

    @staticmethod
    def abs_sobel_thresh(img, sobel_kernel=3, thresh=[0, 255], orient='x'):
        '''
        Calculate directional gradient
        '''
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as binary_output image
        return binary_output

    @staticmethod
    def mag_thresh(img, sobel_kernel=3, thresh=(30, 255)):
        '''
        Calculate gradient magnitude
        '''
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
        # 6) Create a binary mask where mag thresholds are met
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # 7) Return this mask as your binary_output image
        binary_output = np.copy(img) # Remove this line
        return binary_output

    @staticmethod
    def dir_threshold(img, sobel_kernel=15, thresh=(0, np.pi/2)):
        '''
        Calculate gradient direction
        '''
        # Apply the following steps to img
        # 1) Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        gradient = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        sbinary = np.zeros_like(gradient)
        sbinary[(gradient >= thresh[0]) & (gradient <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return sbinary

    @staticmethod
    def apply_gradient_filters(image, ksize=15):
        '''
        param ksize integer denoting Sobel kernel size
        Choose a larger odd number to smooth gradient measurements
        '''
        gradx = SobelThresholder.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
        grady = SobelThresholder.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
        mag_binary = SobelThresholder.mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
        dir_binary = SobelThresholder.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
        # combine filters
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined