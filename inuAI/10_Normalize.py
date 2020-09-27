import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/man_alone_train.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow(img)

img_normalized = cv2.normalize(img, None, 100,255, cv2.NORM_MINMAX)
cv2.imshow(img_normalized)

hist_org = cv2.calcHist([img], [0], None, [256], [0,256])
hist_norm = cv2.calcHist([img_normalized], [0], None, [256], [0,256])
plt.plot(hist_org, color='b')
plt.show()
plt.plot(hist_norm, color='r')
plt.show()

