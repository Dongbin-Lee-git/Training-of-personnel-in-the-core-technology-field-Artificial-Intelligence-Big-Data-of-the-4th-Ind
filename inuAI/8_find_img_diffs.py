import cv2
import numpy as np

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/coffee_cake.jpg')
h, w, c = img.shape
print('img shape:', img.shape)
cv2.imshow(img)

# * line 1: img1은 왼쪽 이미지를 나타낸다. //는 정수 나누기를 의미
# * line 2: img2는 오른쪽 이미지를 나타낸다.
# * line 3: cv2.absdiff( )함수를 이용하여 차 영상을 계산한다.

img1 = img[:, :w//2, :]
img2 = img[:, w//2:, :]
diff = cv2.absdiff(img1, img2)
print('diff shape: ',diff.shape)
cv2.imshow(diff)


# * line 9-18: 좌상단에 위치한 이미지의 모서리를 찾는다.

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/pander_two_imgs.jpg')
h, w, c = img.shape
print('img shape:', img.shape)
cv2.imshow(img)

left_upper_x = 0
left_upper_y = 0
found = False
for _w in range(w):
  for _h in range(h):
    if np.array_equal(img[_h,_w, :], [255,255,255]) is False:
      print('Found: ', _w, _h)
      left_upper_x = _w
      left_upper_y = _h
      found = True
      break
  if found is True:
    break

right_upper_x = 0
right_upper_y = 0
found = False
for _w in range(w-1,-1,-1):
  for _h in range(h):
    if np.array_equal(img[_h,_w, :], [255,255,255]) is False:
      print('Found: ', _w, _h)
      right_upper_x = _w
      right_upper_y = _h
      found = True
      break
  if found is True:
    break

left_lower_x = 0
left_lower_y = 0
found = False
for _w in range(w):
  for _h in range(h-1,-1,-1):
    if np.array_equal(img[_h,_w, :], [255,255,255]) is False:
      print('Found: ', _w, _h)
      left_lower_x = _w
      left_lower_y = _h
      found = True
      break
  if found is True:
    break

found = False
right_lower_x = 0
right_lower_y = 0
for _w in range(w-1,-1,-1):
  for _h in range(h-1,-1,-1):
    if np.array_equal(img[_h,_w, :], [255,255,255]) is False:
      print('Found: ', _w, _h)
      right_lower_x = _w
      right_lower_y = _h
      found = True
      break
  if found is True:
    break

for _h in range(left_upper_y,h):
  if np.array_equal(img[_h, left_upper_x, :], [255,255,255]) is True:
    print('Found middle: ', left_upper_x, _h)

img1 = img[16:397, left_upper_x:904, :]
img2 = img[411:792, left_upper_x:904, :]

print('img1 shape: ', img1.shape)
print('img2 shape: ', img2.shape)

cv2.imshow(img1)
cv2.imshow(img2)

diff = cv2.absdiff(img1, img2)
print('diff shape: ',diff.shape)
cv2.imshow(diff)

