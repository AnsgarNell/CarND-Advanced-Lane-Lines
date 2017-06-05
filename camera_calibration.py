import numpy as np
import cv2
import glob
import os

# STEP 1 CALIBRATION

# Number of corners
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
	print('Processing image', fname)
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		# Draw and display the corners
		cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		filename = os.path.basename(fname)
		write_name = './output_images/' + filename
		cv2.imwrite(write_name, img)

import pickle

# We take the image calibration1 as 
img = cv2.imread('camera_cal/calibration3.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("camera_cal/wide_dist_pickle.p", "wb" ))

# Sa
for fname in images:
	print('Processing undistortion of image', fname)
	img = cv2.imread(fname)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	filename = os.path.basename(fname)
	write_name = './output_images/undist_' + filename
	cv2.imwrite(write_name,dst)