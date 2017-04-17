import cv2
import numpy as np
from lane_detector import LaneDetector
from final_thresholder import FinalThresholder
from perspective_transformer import PerspectiveTransformer

class Pipeline:
    @staticmethod
    def add_detected_lanes(img, mtx, dist, corners):
        # undistort image
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Apply gradient and color filters
        filters = FinalThresholder.grad_color_threshold(img)
        combined_binary = np.zeros_like(filters)
        combined_binary[(filters[:,:,1] == 1) | (filters[:,:,2] == 1)] = 1
        # Transform perspective
        binary_warped, Minv = PerspectiveTransformer.transform(combined_binary.astype(np.uint8), corners)
        # Find lanes
        left_fitx, right_fitx, ploty, _, _ = LaneDetector.find_lane_lines(binary_warped)
        # Draw predicted lane area
        result = LaneDetector.show_inside_lane(undist, binary_warped, Minv, left_fitx, right_fitx, ploty)
        # show curvature
        curve_rad = LaneDetector.get_curvature_radius(left_fitx, right_fitx, ploty)
        cv2.putText(result, 'Radius of curvature (m): {:.2f}'.format(curve_rad),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        return result
