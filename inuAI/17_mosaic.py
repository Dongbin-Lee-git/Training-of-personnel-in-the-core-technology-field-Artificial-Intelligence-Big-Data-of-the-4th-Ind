import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/My Drive/Colab Notebooks/PYTORCH강의/ImageProcessing/images/man_with_tattoo.jpg')
h, w, _ = img.shape
print(img.shape)
cv2.imshow(img)

roi = img[265:356, 467:553]
roi = cv2.resize(roi, (int((553-467)/10), int((356-265)/10)))
roi = cv2.resize(roi, ((553-467), (356-265)), interpolation=cv2.INTER_AREA)
print(roi.shape)
img[265:356, 467:553] = roi
cv2.imshow(img)

