import pickle
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from collections import deque

# Define a class to receive the characteristics of each line detection
class Lane():
	def __init__(self):
		self.detected = False  
		self.left_fit = None
		self.right_fit = None  
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the lane
		self.lane_center_meters = None 

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "./camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
input_folder = './test_images/'
output_folder = './output_images/'
# Make a list of calibration images
images = glob.glob(input_folder + '*.jpg')
filename = 'test_image.jpg'
save_images = True
is_video = False
lanes = deque(maxlen=6)

def region_of_interest(img):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	imshape = img.shape
	vertices = np.array([[(0,imshape[0]),(0,440),(imshape[1],440),(imshape[1],imshape[0])]], dtype=np.int32)
	
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

# Combined color and gradient thresholds
def threshold(img, s_thresh=(120, 255), sx_thresh=(20, 60)):

	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	
	L = hls[:,:,1]
	S = hls[:,:,2]
	
	l_thresh = (40, 170)
	l_binary = np.zeros_like(L)
	l_binary[(L > l_thresh[0]) & (L <= l_thresh[1])] = 1
	
	s_thresh = (70, 255)
	s_binary = np.zeros_like(S)
	s_binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1
	
	G = img[:,:,1]
	R = img[:,:,2]
	
	R_thresh = (210, 255)
	R_binary = np.zeros_like(R)
	R_binary[(R > R_thresh[0]) & (R <= R_thresh[1])] = 1
	
	G_thresh = (195, 255)
	G_binary = np.zeros_like(G)
	G_binary[(G > G_thresh[0]) & (G <= G_thresh[1])] = 1
	
	combined = np.zeros_like(l_binary)
	combined[((s_binary == 1) & (l_binary==1)) | (R_binary == 1) | (G_binary == 1)] = 1
	
	return combined
	
def transform(img):
	img_size = (img.shape[1], img.shape[0])
	offset = 200
	x1 = 200
	x3 = 1100
	# We calculated the source points on a straight lines image using Photoshop
	src = np.float32([[x1,720], [599,446], [680,446], [x3,720]])
	# c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
	dst = np.float32([[x1+offset,720], [x1+offset,0], [x3-offset,0], [x3-offset,720]])
	# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	# e) use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	
	return warped, Minv
	
def detect_lines(binary_warped,lane):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	
	if save_images:
		# Save the histogram
		plt.plot(histogram)
		write_name = output_folder + 'histogram_' + filename
		plt.savefig(write_name)
		plt.clf()

	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	
	if not(lane.detected):
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:		
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds) 

	else:
		# Assume you now have a new warped binary image 
		# from the next frame of video (also called "binary_warped")
		# It's now much easier to find line pixels!
		left_fit_average = 0
		right_fit_average = 0
		for average_lane in lanes:
			left_fit_average += average_lane.left_fit
			right_fit_average += average_lane.right_fit
		left_fit_average = left_fit_average / len(lanes)
		right_fit_average = right_fit_average / len(lanes)
		left_lane_inds = ((nonzerox > (left_fit_average[0]*(nonzeroy**2) + left_fit_average[1]*nonzeroy + left_fit_average[2] - margin)) & (nonzerox < (left_fit_average[0]*(nonzeroy**2) + left_fit_average[1]*nonzeroy + left_fit_average[2] + margin))) 
		right_lane_inds = ((nonzerox > (lane.right_fit[0]*(nonzeroy**2) + lane.right_fit[1]*nonzeroy + lane.right_fit[2] - margin)) & (nonzerox < (lane.right_fit[0]*(nonzeroy**2) + lane.right_fit[1]*nonzeroy + lane.right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	if is_video:
		lane.detected = True
		lane.left_fit = left_fit
		lane.right_fit = right_fit
		lanes.append(lane)
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, binary_warped.shape[1])
	plt.ylim(binary_warped.shape[0], 0)
	
	if save_images:
		write_name = output_folder + 'sliding_windows_' + filename
		plt.savefig(write_name)
		plt.clf()

	# Calculate curvature
	y_eval = np.max(ploty)
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	curvature = (left_curverad + right_curverad) / 2
	
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	lane_center = (right_fitx[-1] + left_fitx[-1])/2
	center_offset_pixels = abs(binary_warped.shape[1]/2 - lane_center)
	lane_center_meters = center_offset_pixels*xm_per_pix
	
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	return color_warp, curvature, lane_center_meters

def pipeline(img):
	
	global lane

	img = np.copy(img)
	
	# STEP A.2 UNDISTORT
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	
	masked = region_of_interest(undist)
	
	if save_images:
		write_name = output_folder + 'masked_' + filename
		cv2.imwrite(write_name,masked)
	
	# STEP A.3 Color/gradient threshold
	result = threshold(masked)
	
	if save_images:
		# Save binary file
		final_image_RGB = np.dstack((result, result, result))*255
		write_name = output_folder + 'binary_' + filename
		#cv2.imwrite(write_name,final_image_RGB)
	
	# STEP A.4 Perspective transform
	result, Minv = transform(result)
	
	if save_images:
		# Save binary file
		final_image_RGB = np.dstack((result, result, result))*255
		write_name = output_folder + 'binary_warp_' + filename
		cv2.imwrite(write_name,final_image_RGB)
	

	# STEP B Detect lane lines and curvature
	color_warp, curvature, lane_center_meters = detect_lines(result,lane)
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	text = ('Radius of curvature: ' + str(round(curvature,3)) + ' m')
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result,text,(300,70), font, 1,(255,255,255),2)
	text = ('Distance from lane center: ' + str(round(lane_center_meters,3)) + ' m')
	cv2.putText(result,text,(300,120), font, 1,(255,255,255),2)
	
	if save_images:
		# Save final image
		write_name = output_folder + 'final_image_' + filename
		cv2.imwrite(write_name,result)
	
	return result
	
lane = Lane()
	
for fname in images:

	filename = os.path.basename(fname)
	print('Processing file', filename)

	# Read in an image
	img = cv2.imread(fname)
	
	converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	s_channel = converted[:,:,2]
	write_name = output_folder + 'hls_' + filename
	cv2.imwrite(write_name,s_channel)
	
	if save_images:
		# Warp original
		undist = cv2.undistort(img, mtx, dist, None, mtx)
		cv2.line(undist, (210, 720), (599, 446), color=[255,0,0], thickness=1)
		cv2.line(undist, (599, 446), (680, 446), color=[255,0,0], thickness=1)
		cv2.line(undist, (680, 446), (1110, 720), color=[255,0,0], thickness=1)
		cv2.line(undist, (1110, 720), (210, 720), color=[255,0,0], thickness=1)
		write_name = output_folder + 'original_rectangle_' + filename
		cv2.imwrite(write_name,undist)
		warped, Minv = transform(undist)
		write_name = output_folder + 'warped_' + filename
		cv2.imwrite(write_name,warped)

	# Process the image
	result = pipeline(img)
	
	# Save output
	#cv2.imshow('Original image', img)
	#cv2.imshow('Result', result)
	#cv2.waitKey(1000
	#cv2.destroyAllWindows()
	
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

save_images = False
is_video = True
white_output = 'project_video_result.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)