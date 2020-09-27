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

# Bitwise multiplication is done between input image and masked image
# resulting in image having only specific color
output_r = cv2.bitwise_and(input_img, input_img, mask=mask_r)

mask_g = cv2.inRange(input_img, lower_g, upper_g)
output_g = cv2.bitwise_and(input_img, input_img, mask=mask_g)

mask_y = cv2.inRange(input_img, lower_y, upper_y)
output_y = cv2.bitwise_and(input_img, input_img, mask=mask_y)

# add all the individual color into one
combine_img = cv2.add(output_r, output_g)
output_RGY = cv2.add(combine_img, output_y)
# Show the image using Opencv function
cv2.imshow('o', output_RGY)
cv2.waitKey(0)



