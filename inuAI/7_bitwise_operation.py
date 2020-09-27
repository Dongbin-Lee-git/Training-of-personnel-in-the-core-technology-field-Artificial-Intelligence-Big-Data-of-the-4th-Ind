import numpy as np
import cv2

img1 = np.zeros((200,400), dtype=np.uint8)
img2 = np.zeros((200,400), dtype=np.uint8)
print('img1')
cv2.imshow(img1)
print('img2')
cv2.imshow(img2)

img1[:,:200] = 255
img2[100:200,:] = 255
print('img1')
cv2.imshow(img1)
print('img2')
cv2.imshow(img2)

bitAnd = cv2.bitwise_and(img1, img2)
cv2.imshow(bitAnd)

bitOr = cv2.bitwise_or(img1, img2)
cv2.imshow(bitOr)

bitXor = cv2.bitwise_xor(img1, img2)
cv2.imshow(bitXor)

bitNot = cv2.bitwise_not(img1)
cv2.imshow(bitNot)

# * line 1: city.jpg를 읽어들인다.
# * line 2: city.jpg와 같은 크기, 같은 채널의 검정색 이미지를 만든다.
# * line 3: 검정색 이미지의 중심(160,240)에 반지름 100인 하얀색 원을 그리고, 원 내부를 채운다.(-1의 의미)
# * line 4: city 이미지와 mask 이미지를 and 시킨다.

city_img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/city.jpg')
mask = np.zeros_like(city_img)
cv2.circle(mask, (110,240), 100, (255,255,255), -1)
maksed_city = cv2.bitwise_and(city_img, mask)
cv2.imshow(maksed_city)

