import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/building.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
h, w, _ = img.shape
print(img.shape)
cv2.imshow(img)

# * line 1: sine, cosine 곡선의 높이
# * line 2: sine, cosine 곡선의 파장
# * line 3: 

mapy, mapx = np.indices((h, w), dtype=np.float32)

mapy = mapy + 100
remapped_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, None)

cv2.imshow(remapped_img)

amp = 15
wl = 20
mapy, mapx = np.indices((h, w), dtype=np.float32)

sinx = mapx + amp * np.sin(mapy/wl)
cosy = mapy + amp * np.cos(mapx/wl)

img_x = cv2.remap(img, mapx, cosy, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
cv2.imshow(img_x)

img_y = cv2.remap(img, sinx, mapy, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
cv2.imshow(img_y)

img_xy = cv2.remap(img, sinx, cosy, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
cv2.imshow(img_xy)

