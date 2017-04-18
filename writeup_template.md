## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image3]: ./output_images/test_undistort_all.png "Undistorted Test Images"
[image4]: ./output_images/combined_filters.png "Color and Gradient Filters"
[image5]: ./output_images/warped.png "Perspective Transformation"
[image6]: ./output_images/sliding_windows1.png "Fit Lines to Lanes"
[image7]: ./output_images/pipeline_result.png "Detected Lanes on a Test Image"
[image8]: ./output_images/src_corners.png "Finding Source Corners"
[video1]: ./project_video.mp4 "Project Video"
[video2]: ./output_video.mp4 "Output Video"
[video3]: ./challenge_video.mp4 "Challenge 1 Project Video"
[video4]: ./out_challenge_video.mp4 "Harder Challenge Output Video"
[video5]: ./harder_challenge_video.mp4 "Harder Challenge Project Video"
[video6]: ./out_harder_challenge_video.mp4 "Harder Challenge Output Video"

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

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

### Camera Calibration

#### Code Info

Note that all the code is located in [AdvancedLaneFindingWorkbook.ipynb](/AdvancedLaneFindingWorkbook.ipynb) and I will talking about that code only. Some of the reusable components of this jupyter file are present in the [/src](/src) directory, which we will not talk about in this writeup as it is duplicate. 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

To undistort images, I computed camera matrix and distortion coefficients. To do so, we have to perform camera calibration. 
To calibrate camera, I need image and object points. 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.

 `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. To get image points, I converted images to gray scale, and then applied `cv2.findChessboardCorners` on them which returned corners i.e. image points. Image points are corners in the chessboard images.  

After that I use `cv2.calibrateCamera` on image points and object points to get the calibration matrix, i.e. camera calibration and distortion coefficients. 

I used `cv2.findChessboardCorners()` function:

![alt text][image1]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)
Following are the steps in the pipeline:
* Undistort an input image
* Apply gradient and color filter i.e. use color transforms, gradients or other methods to create a thresholded binary image
* Perspective transformation
* Lane detection
* Calculate curvature of the lane
* Plot area between lanes

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to the images:

![alt text][image3]

Check the section "Undistort the Test Images" for this.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image in `FinalThresholder.grad_color_threshold()`, finally getting the final filtered image via `FinalThresholder.get_thresholded()`. 
Here are the transforms I used:
* Sobel gradient in x/y direction
* Magnitude of Sobel gradient
* Direction of Sobel gradient
* Using only the S channel from the HLS colorspac

Here is the final result for test1 image:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `PerspectiveTransformer.transform()`.  The function takes as inputs an image (`img`), as well as source (`src`) points.  I chose the hardcode the source and points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 300, 0        | 
| 700, 450      | 980, 0        |
| 1150, 719     | 980, 720      |
| 255, 719      | 300, 720      |

Then I mapped these corners to the destination points, which is hard-coded in the `PerspectiveTransformer.transform()` function as:


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

Sample image to show the trapezoid:
![alt text][image8]


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find lane lines in the transformed image, I used histograms to detect color peak in the transformed image. This is `LaneDetector.find_lane_lines()` function in the "Fit polynomial" section in the IPython notebook. 

To find the lane pixels, we use sliding window algorithm. In this algorithm, we check the histograms from bottom to top within sliding windows and detect curved lane lines. After that, I fit polynomial to the detected lines in the sliding windows, a smooth curve can be detected. The detected image is visualized as yellow lines in the image below.


Then we use 2nd order polynomial to fit the lane:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

A function to calculate radius of curvature (`LaneDetector.get_curvature_radius()`) is implemented in the "Fit Polynomial" section in the IPython notebook. This function takes plotting data of the lane lines, and returns radius of curvature in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, I implemented `Pipeline.add_detected_lanes()` in the "Pipeline" section of the IPython notebook. This function applies all steps described above to one image. This is output of this function with a test image:

![alt text][image7]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a input video:
 ![alt text][video1]
 
 Here is a output video:
 ![alt text][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Couple of Points:
* Selecting the threshold values for color and sobel gradient is challenging and time consuming. While S channel is good, but it still has problems when there is high contrast in color of the road or when there is shadow on the road or there is vehicle going through the road.
* Dealing with curvature is problematic, and when I tried the harder challenge videos, it certainly failed to find the lane.
* Sometimes, lane have some markings in the middle and this approach assumes only the small part i.e. half of the lane is actual lane, which can be challenging when driving the actual vehicle.
