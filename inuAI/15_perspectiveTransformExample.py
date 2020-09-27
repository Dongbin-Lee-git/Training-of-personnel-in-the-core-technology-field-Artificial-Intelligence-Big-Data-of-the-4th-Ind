import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/document.jpg')
cv2.imshow(img)

# * line 1-6: 이미지 내 문서의 네 꼭지점을 pts1으로 설정한다.
# * line 7-13: 네 꼭지점을 이미지의 네 꼭지점으로 설정. 이미지 크기를 360 x 480으로 설정 

pts1 = np.float32([
                  [112,12],
                  [345,83],
                  [90,472],
                  [330,421]
])
pts2 = np.float32(
  [
   [0,0],
   [359,0],
   [0,479],
   [359,479]
  ]
)
cv2.circle(img, tuple(pts1[0]), 5, (0,0,255), -1)
cv2.circle(img, tuple(pts1[1]), 5, (0,255,0), -1)
cv2.circle(img, tuple(pts1[2]), 5, (255,0,0), -1)
cv2.circle(img, tuple(pts1[3]), 5, (255,0,0), -1)

h, w, _ = img.shape

mtrx = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, mtrx, (w, h) )
cv2.imshow(img)
cv2.imshow(dst)

