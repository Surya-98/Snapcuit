import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
from sklearn.metrics.pairwise import euclidean_distances

src = cv2.imread('Circuit.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

width = 500
height = 500
dim = (width, height)

# resize image
src = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
bw = cv2.resize(bw, dim, interpolation = cv2.INTER_AREA)

(thresh, bw) = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


kernel = np.ones((20,20),np.uint8)
bw1 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


kernel = np.ones((3,3),np.uint8)
bw2 = cv2.dilate(bw,kernel,iterations = 1)

wires = cv2.subtract(bw1, bw2)

kernel = np.ones((20,20),np.uint8)

wires = cv2.morphologyEx(wires, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((10,10),np.uint8)
wires = cv2.dilate(wires,kernel,iterations = 1)

wires = cv2.subtract(bw,wires)
'''wires = cv2.subtract(wires, bw)
wires = cv2.morphologyEx(wires, cv2.MORPH_CLOSE, kernel)
wires = cv2.subtract(wires, bw)'''

cv2.imshow("img", src)
cv2.imshow('BW1', bw2)
cv2.imshow('BW', bw1)
cv2.imshow('Wires', wires)

cv2.waitKey(0)
cv2.destroyAllWindows()

