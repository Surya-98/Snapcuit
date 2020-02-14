import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
from sklearn.metrics.pairwise import euclidean_distances
import time

name = "Circuit"
'''
if name == "Circuit":
    name_image = name + ".jpg"
else:
    name_image = name + ".png"

src = cv2.imread(name_image)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = np.ones((2,2),np.uint8)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
#10 for Circuit
#the kernal size should be small for small circuits
nodes = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
#detects nodes

print('Original Dimensions : ',bw.shape)

width = 500
height = 500
dim = (width, height)

# resize image
src = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
bw = cv2.resize(bw, dim, interpolation = cv2.INTER_AREA)
nodes = cv2.resize(nodes, dim, interpolation = cv2.INTER_AREA)
(thresh, bw) = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(thresh, nodes) = cv2.threshold(nodes, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print('Resized Dimensions : ',bw.shape)


cv2.imshow('SRC', bw)   

cv2.waitKey(0)
cv2.destroyAllWindows()
time.sleep(5)
'''

name_file = name + ".cir"
f=open(name_file, "r")
contents =f.read()
print contents
