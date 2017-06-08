#https://stackoverflow.com/questions/42169247/apply-opencv-look-up-table-lut-to-an-image

import pickle
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

input_folder = './project_video/'
output_folder = './project_video/'
# Make a list of calibration images
images = glob.glob(input_folder + 'filename02*.jpg')


lut = np.zeros((256, 1, 1), dtype=np.uint8)

lut[:, 0, 0] = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,251,249,247,245,242,241,238,237,235,233,231,229,227,225,223,221,219,217,215,213,211,209,207,205,203,201,199,197,195,193,191,189,187,185,183,181,179,177,175,173,171,169,167,165,163,161,159,157,155,153,151,149,147,145,143,141,138,136,134,132,131,129,126,125,122,121,118,116,115,113,111,109,107,105,102,100,98,97,94,93,91,89,87,84,83,81,79,77,75,73,70,68,66,64,63,61,59,57,54,52,51,49,47,44,42,40,39,37,34,33,31,29,27,25,22,20,18,17,14,13,11,9,6,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    



for fname in images:

	filename = os.path.basename(fname)
	print('Processing file', filename)

	# Read in an image
	img = cv2.imread(fname,cv2.IMREAD_COLOR)
	
	im_color = cv2.LUT(img, lut)
	
	converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	
	# Save output
	#cv2.imshow('Original image', img)
	#cv2.imshow('Result', im_color)
	#cv2.waitKey(1000)
	#cv2.destroyAllWindows()
	
	write_name = output_folder + 'LUT_' + filename
	cv2.imwrite(write_name,im_color)
	
	write_name = output_folder + 'HLS_' + filename
	cv2.imwrite(write_name,converted)
	
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