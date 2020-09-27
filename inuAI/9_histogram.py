import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/man_alone_train.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow(img)

hist = cv2.calcHist([img], [0], None, [256], [0,256])
print(hist.shape)
plt.plot(hist)
plt.show()

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/man_jump.jpg', 
                 cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow(img)

hist = cv2.calcHist([img], [0], None, [256], [0,256])
print(hist.shape)
plt.plot(hist)
plt.show()

# # Color image에 대한 histogram

cimg = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/strawberry.jpg')
cimg = cv2.resize(cimg, None, fx=0.5, fy=0.5)
cv2.imshow(cimg)

# * line 1: cv2.split은 이미지를 채널별로 나눈다. 나뉜 결과는 1채널을 가진 이미지들
# * line 4: 채널별로 나뉜 이미지에 대해서 히스토그램을 계산

alist = ('a', 'b', 'c')
for _a in alist:
  print(_a)

alist = ('a', 'b', 'c')
blist = ('가', '나', '다')

for _a, _b in zip(alist, blist):
  print(_a, _b)

channels = cv2.split(cimg)
colors = ('b', 'g', 'r')
for ch, color in zip(channels, colors):
  hist = cv2.calcHist([ch], [0], None, [256], [0,256])
  plt.plot(hist, color=color)
plt.show()

