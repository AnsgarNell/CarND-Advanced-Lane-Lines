import pickle
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "./camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
input_folder = './test_images/'
output_folder = './output_images/'

# Combined color and gradient thresholds
def hls_select(img, s_thresh=(170, 255), sx_thresh=(20, 100), l_thresh=(40,255)):
	# Convert to HSV color space and separate the S channel
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
	l_channel = hsv[:,:,1]
	s_channel = hsv[:,:,2]
	
	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	
	# Threshold saturation channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	
	# Threshold lightness
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
	
	# Combine the binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
	return combined_binary
	
def transform(img):
	img_size = (img.shape[1], img.shape[0])
	offset = 150
	x1 = 200
	x3 = 1130
	# We calculated the source points on a straight lines image using Photoshop
	src = np.float32([[x1,720], [565,470], [725,470], [x3,720]])
	# c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
	dst = np.float32([[x1+offset,720], [x1+offset,0], [x3-offset,0], [x3-offset,720]])
	# d) use cv2.getPerspectiveTransform() to get M, the transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	# e) use cv2.warpPerspective() to warp your image to a top-down view
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, Minv
	
def detect_lines(binary_warped, filename):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Save the histogram
	plt.plot(histogram)
	write_name = './output_images/histogram_' + filename
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

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
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
	radius_curvature = 0.5*(left_curverad + right_curverad)
	
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	return color_warp, radius_curvature

def pipeline(img, filename):
	img = np.copy(img)
	
	# STEP A.2 UNDISTORT
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	
	# STEP A.3 Color/gradient threshold
	result = hls_select(undist)
	
	# STEP A.4 Perspective transform
	result, Minv = transform(result)
	# Save binary file
	final_image_RGB = out_img = np.dstack((result, result, result))*255
	write_name = output_folder + 'binary_warp_' + filename
	cv2.imwrite(write_name,final_image_RGB)
	
	# STEP B Detect lane lines and curvature
	color_warp, radius_curvature = detect_lines(result, filename)
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	print('Radius of curvature: ', radius_curvature, 'm')
	#cv2.putText(result,str,(430,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)   
	
	# Save final image
	write_name = output_folder + 'final_image_' + filename
	cv2.imwrite(write_name,result)

# Make a list of calibration images
images = glob.glob(input_folder + '*.jpg')

for fname in images:

	filename = os.path.basename(fname)
	print('Processing file', filename)

	# Read in an image
	img = cv2.imread(fname)
	
	# Warp original
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	warped, Minv = transform(undist)
	write_name = output_folder + 'warped_' + filename
	cv2.imwrite(write_name,warped)

	# Process the image
	pipeline(img, filename)
	
	# Save output
	#cv2.imshow('Original image', img)
	#cv2.imshow('Undistorted image', final_image)
	
cv2.destroyAllWindows()