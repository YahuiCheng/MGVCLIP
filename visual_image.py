
import os
import cv2
import runpy as np

image = cv2.imread('mvtec/000.png')
image = cv2.resize(image, (240,240))

mask = cv2.imread('mvtec/000_mask.png')
mask = cv2.resize(mask, (240, 240))

mask[:, :, :2] *= 0

mask = mask.astype(image.dtype)

rst = cv2.addWeighted(mask, 0.5, image, 0.5, 0)

cv2.imshow('1', rst)
cv2.imwrite('mvtec/000.png', rst)

cv2.waitKey()
cv2.destroyAllWindows()




