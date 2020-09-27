import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/red_car.jpg')
img = cv2.resize(img,None, fx=0.5, fy=0.5)
cv2.imshow(img)

# * line 1-5: 변환 전의 3점의 좌표
# * line 6-12: 변환 후의 3점의 좌표

pts1 = np.float32([
                  [100,50],
                  [200,50],
                  [100,200]
])
pts2 = np.float32(
  [
   [80,70],
   [210,60],
   [250,120]
  ]
)

# * line 1-3: 변환 전의 3점을 빨간색, 녹색, 청색으로 표시한다.
# * line 7: pt1과 pts2 사이의 affine transformation을 계산한다.
# * line 8: affine transformation을 적용하여 결과 이미지 dst를 생성한다.

cv2.circle(img, tuple(pts1[0]), 5, (0,0,255), -1)
cv2.circle(img, tuple(pts1[1]), 5, (0,255,0), -1)
cv2.circle(img, tuple(pts1[2]), 5, (255,0,0), -1)

h, w, _ = img.shape

mtrx = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, mtrx, (2*w, h) )
cv2.imshow(img)
cv2.imshow(dst)

