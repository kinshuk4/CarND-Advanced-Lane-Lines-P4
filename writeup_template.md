##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_corners.png "Chessboard Corners"
[image2]: ./output_images/chessboard_undistort.png "Undistorted Chessboard"
[image3]: ./output_images/test1_undistort.png "Undistorted image"
[image4]: ./output_images/combined_filters.png "Color and Gradient Filters"
[image5]: ./output_images/warped.png "Perspective Transformation"
[image6]: ./output_images/fit_lanes.png "Fit Lines to Lanes"
[image7]: ./output_images/plot_lanes.png "Detected Lanes on a Test Image"
[image8]: ./output_images/src_corners.png "Finding Source Corners"
[image9]: ./output_images/pipeline_result.png "Pipeline Result"
[video1]: ./project_video.mp4 "Project Video"
[video2]: ./output_video.mp4 "Output Video"
[video3]: ./challenge_video.mp4 "Challenge 1 Project Video"
[video4]: ./out_challenge_video.mp4 "Harder Challenge Output Video"
[video5]: ./harder_challenge_video.mp4 "Harder Challenge Project Video"
[video6]: ./out_harder_challenge_video.mp4 "Harder Challenge Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Files

The list of files of this project is as follows:

- [writeup.md](/writeup.md): This document. Summary of the project.
- [AdvancedLaneFindingWorkbook.ipynb](/AdvancedLaneFindingWorkbook.ipynb): IPython notebook file that contains implementation of lane lines detection.
- [project_video.mp4](/project_video.mp4): Original video data before applying lane lines detection.
- [output_video.mp4](/output_video.mp4): Final result video of the lane lines detection.
- [camera_cal](/camera_cal): Contains chessboard images that is used to calibrate the camera.
- [output_images](/output_images): Contains intermediate output images generated in the IPython notebook.
- [test_images](/test_images): Contains sample input images used in the IPython notebook.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./AdvancedLaneFindingWorkbook.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Image points are corners in the chessboard images. To detect corners in those chessboards, I used `cv2.findChessboardCorners()` function:

![alt text][image1]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

###Pipeline (single images)
Following are the steps in the pipeline:
* Undistort an input image
* Apply gradient and color filter i.e. use color transforms, gradients or other methods to create a thresholded binary image
* Perspective transformation
* Lane detection
* Calculate curvature of the lane
* Plot area between lanes

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image in `FinalThresholder.grad_color_threshold()`, finally getting the final filtered image via `FinalThresholder.get_thresholded()`. Here is the image:

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `PerspectiveTransformator.transform()`.  The function takes as inputs an image (`img`), as well as source (`src`) points.  I chose the hardcode the source points in the following manner:

![alt text][image8]

Here are the source points:

```
src_corners = [
    [600, 450],  # top left
    [700, 450],  # top right
    [1150, 719], # bottom right
    [255, 719]   # bottom left
]
```

Then I mapped these corners to the destination points, which is hard-coded in the `transorm_perspective()` function as:

```
offset_x = 300
offset_y = 0
dst = np.float32([
	[offset_x, offset_y],  # top left
	[img_size[0] - offset_x, offset_y],  # top right 
	[img_size[0] - offset_x, img_size[1] - offset_y], # bottom right
	[offset_x, img_size[1] - offset_y]  # bottom left
])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 300, 0        | 
| 700, 450      | 980, 0        |
| 1150, 719     | 980, 720      |
| 255, 719      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lane lines in the transformed image, I used histograms to detect color peak in the transformed image. This is `LaneDetector.find_lanes_first()` function in the "Fit polynomial" section in the IPython notebook. By checking histograms from bottom to top within sliding windows, I can detect curved lane lines. After that, I fit polynomial to the detected lines in the sliding windows, a smooth curve can be detected. The detected image is visualized as yellow lines in the image below.


Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

A function to calculate radius of curvature (`LaneDetector.get_curvature_radius()`) is implemeted in the "Fit Polynomial" section in the IPython notebook. This function takes plotting data of the lane lines, and returns radius of curvature in meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, I implemented `Pipeline.add_detected_lanes()` in the "Pipeline" section of the IPython notebook. This function applies all steps described above to one image. This is output of this function with a test image:

![alt text][image9]


---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a input video:
 ![alt text][video1]
 
 Here is a output video:
 ![alt text][video2]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


