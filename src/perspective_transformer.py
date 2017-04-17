import cv2
import numpy as np

class PerspectiveTransformer:
    @staticmethod
    def transform(img, src_corners):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        offset_x = 300 # offset for dst points
        offset_y = 0
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
        # For source points I'm grabbing the outer four detected corners
        src = np.float32(src_corners)
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([
            [offset_x, offset_y],  # top left
            [img_size[0] - offset_x, offset_y],  # top right
            [img_size[0] - offset_x, img_size[1] - offset_y], # bottom right
            [offset_x, img_size[1] - offset_y]  # bottom left
        ])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Use this to revert transformation
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        binary_warped = cv2.warpPerspective(gray, M, img_size)
        return binary_warped, Minv

