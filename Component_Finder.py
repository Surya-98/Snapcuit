import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('Circuit.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
(thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(type(img_bw))
print(np.where(img_bw == 255))
print(img_bw)