import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/building.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
print(img.shape)
cv2.imshow(img)

# * line 1-6: 변환 전의 4 점의 좌표
# * line 7 - 13: 변환 후의 4 점의 좌표

pts1 = np.float32([
                  [0,0],
                  [479,0],
                  [0,487],
                  [479,487]
])
pts2 = np.float32(
  [
   [150,0],
   [300,0],
   [0,487],
   [479,487]
  ]
)

# * line 1-4: 변환 전의 4 점의 좌표를 색점으로 표시
# * line 8: perspective transformation을 계산
# * line 9: 변환을 적용

cv2.circle(img, tuple(pts1[0]), 5, (0,0,255), -1)
cv2.circle(img, tuple(pts1[1]), 5, (0,255,0), -1)
cv2.circle(img, tuple(pts1[2]), 5, (255,0,0), -1)
cv2.circle(img, tuple(pts1[3]), 5, (255,0,0), -1)

h, w, _ = img.shape

mtrx = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, mtrx, (w, h) )
cv2.imshow(img)
cv2.imshow(dst)

