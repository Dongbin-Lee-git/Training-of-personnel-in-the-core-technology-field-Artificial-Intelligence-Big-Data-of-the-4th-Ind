import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/red_car.jpg')
img = cv2.resize(img,None, fx=0.3, fy=0.3)
cv2.imshow(img)

# * 이미지 중앙을 기준으로 45도 회전시키면서, 크기는 그대로 유지

h, w, _ =img.shape
m45 = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
img45 = cv2.warpAffine(img, m45, (w, h))
cv2.imshow(img45)

# * 이미지 중앙을 기준으로 반시계 방향 90도 회전시키면서, 크기는 그대로
#

m90 = cv2.getRotationMatrix2D((w//2, h//2), 90, 1.0)
img90 = cv2.warpAffine(img, m90, (w, h))
cv2.imshow(img90)

