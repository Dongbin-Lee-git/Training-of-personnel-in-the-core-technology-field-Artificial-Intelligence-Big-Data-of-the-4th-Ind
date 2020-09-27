import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Boundary value Defined
lower_r = np.array([0,0,249]).astype('uint8')
lower_g = np.array([57,190,14]).astype('uint8')
lower_y = np.array([0,198,255]).astype('uint8')
upper_r = np.array([128,128,252]).astype('uint8')
upper_g = np.array([107,206,74]).astype('uint8')
upper_y = np.array([128,227,255]).astype('uint8')

# read image using opencv module
input_img = cv2.imread('P2_Input_Image.png')

# Raw image converted to get binary image
# All the pixel value in the range have value 255 and other have 0
mask_r = cv2.inRange(input_img, lower_r, upper_r)
mask_g = cv2.inRange(input_img, lower_g, upper_g)
mask_y = cv2.inRange(input_img, lower_y, upper_y)

red = np.where(mask_r!=0)
red = len(red[0])

green = np.where(mask_g!=0)
green = len(green[0])

yellow = np.where(mask_y!=0)
yellow = len(yellow[0])

total = red + green + yellow

# Congestion Index
Ci = np.around(red/total, decimals=4)
print(Ci)


