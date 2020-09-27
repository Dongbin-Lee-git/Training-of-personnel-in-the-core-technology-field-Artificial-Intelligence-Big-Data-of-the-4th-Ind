import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/metro_woman.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,None, fx=0.3, fy=0.3)
cv2.imshow(img)

img_equalized = cv2.equalizeHist(img)
cv2.imshow(img_equalized)

hist_org = cv2.calcHist([img], [0], None, [256], [0,256])
hist_equalized = cv2.calcHist([img_equalized], [0], None, [256], [0,256])
plt.plot(hist_org, color='b')
plt.show()
plt.plot(hist_equalized, color='r')
plt.show()

