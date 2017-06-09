#https://stackoverflow.com/questions/42169247/apply-opencv-look-up-table-lut-to-an-image

import pickle
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

input_folder = './test_images/'
output_folder = './project_video/'
# Make a list of calibration images
images = glob.glob(input_folder + '*.jpg')

for fname in images:

	filename = os.path.basename(fname)
	print('Processing file', filename)

	# Read in an image
	img = cv2.imread(fname)
	
	converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	
	r_thresh=(210,255)
	
	R = img[:,:,2]
	
	thresh = (210, 255)
	binary = np.zeros_like(R)
	binary[(R > thresh[0]) & (R <= thresh[1])] = 1
	
	cv2.imshow('Result', binary)
	cv2.waitKey(1000)
	cv2.destroyAllWindows()
	
"""
	# Load the image
	img = cv2.imread('input.jpg',cv2.IMREAD_COLOR)
	dstImage = cv2.LUT(img, lut)
	cv2.imwrite('output.jpg', dstImage)
"""