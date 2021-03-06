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

[image1]: ./camera_cal/calibration1.jpg "Undistorted"
[image2]: ./output_images/undist_calibration1.jpg "Undistorted"
[image3]: ./output_images/original_undistorted_mixed.jpg "Undistorted mixed"
[image4]: ./output_images/binary_straight_lines1.jpg "Binary"
[image5]: ./output_images/original_rectangle_straight_lines1.jpg "Binary Example"
[image6]: ./output_images/warped_straight_lines1.jpg "Warp Example"
[image7]: ./output_images/sliding_windows_straight_lines1.jpg "Fit Visual"
[image8]: ./output_images/final_image_straight_lines2.jpg "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

In the image I mixed the original and the undistorted images in order to see the differences. We can observe a kind of tunnel effect, as the center of the image is quite similar, but in the borders it's more fuzzy, and for example the positions from traffic signs changes.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of S, L, R and G channels thresholds to generate a binary image after transforming the original image to HLS (thresholding steps at lines 61 through 90 in `advanced_lane_lines.py`).  Here's an example of my output for this step.

![alt text][image4]

I also created a Region Of Interest in the bottom part of the image, starting at y=440 (lines 33-58).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform()`, which appears in lines 92 through 107 in the file `advanced_lane_lines.py` . The `transform()` function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points were calculated using Photoshop, but it's accuracy is not very good as the process was made by hand.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 599, 446      | 310, 0        | 
| 200, 720      | 310, 720      |
| 1100, 720     | 1100, 720     |
| 680, 446      | 1100, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then using the code provided in the lesson 33 from the project, using first a histogram I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

This can be found in the function `detect_lines()` in lines 109-256 in file `advanced_lane_lines.py`.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 235 through 239 in my code in `advanced_lane_lines.py`, again using the code provided in lesson 35. The final curvature is calculated as the average of both values.

```python
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
curvature = (left_curverad + right_curverad) / 2
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 295 through 303 in my code in `advanced_lane_lines.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

To begin, yes, the pipeline will fail as proved testing the program against more difficult videos like _challenge___video.mp4_ or _harderchallengevideo.mp4_.

The main issue while developing the program was the huge difference in results that one obtains when changing a little bit values like src or dst coordinates or their offset.

Also, the detected lines in previous frames where used always. A better approach would be to check with the curvature and distance to lane center if the detection is good enough and maintain an arry of n last detected lanes to have an average value. 
