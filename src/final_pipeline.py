import cv2
import numpy as np
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

import src.lane_detector as ldt
import src.final_thresholder as fth
import src.perspective_transformer as ppt


def pipeline_for_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Apply gradient and color filters
    _, combined_binary = fth.grad_color_threshold(img)

    # Transform perspective
    # binary_warped, Minv, M = ppt.transform_with_offset(combined_binary.astype(np.uint8), corners, is_gray=True)
    binary_warped, Minv, M = ppt.transform_img(combined_binary, is_gray=True)
    # Find lanes
    left_fitx, right_fitx, ploty, left_fit, right_fit = ldt.find_lane_lines(binary_warped)
    # Draw predicted lane area
    result = ldt.show_inside_lane(undist, binary_warped, Minv, left_fitx, right_fitx, ploty)

    lane_mid = (left_fitx + right_fitx) / 2.0

    off_center = ldt.dist_from_center(left_fitx, right_fitx)
    # show curvature
    curve_rad, left_curverad, right_curverad = ldt.get_curvature_radius(left_fitx, right_fitx, ploty)

    cv2.putText(result, 'Radius of curvature (m): {:.2f}'.format(curve_rad),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(result, 'Distance from center (m): {:.2f}'.format(off_center),
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def pipeline_for_video(mtx, dist, input_video="./project_video.mp4", output_video='output_video.mp4'):
    clip1 = VideoFileClip(input_video)
    test_clip = clip1.fx(transform_image, mtx, dist)
    test_clip.write_videofile(output_video, audio=False, progress_bar=False)


def transform_image(clip, mtx, dist):
    """Helper function to apply lane detection with parameters."""

    def _transform(img):
        return pipeline_for_image(img, mtx, dist)

    return clip.fl_image(_transform)
