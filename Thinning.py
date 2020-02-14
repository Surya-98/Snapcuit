import cv2
import numpy as np
from matplotlib import pyplot as plt
src = cv2.imread('Circuit.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#kernel = np.ones((2,2),np.uint8)
#bw1 = cv2.erode(bw,kernel,iterations = 1)
'''
kernel = np.array([(0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (1, 1, 1, 1, 1, 1, 1, 1, 1),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0),
                   (0, 0, 0, 0, 1, 0, 0, 0, 0)],np.uint8)

bw1 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
'''

kernel = np.ones((2,2),np.uint8)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
bw1 = bw
bw1 = cv2.ximgproc.thinning(bw, bw1, 0)
kernel = np.ones((3,3),np.uint8)
bw2 = cv2.dilate(bw1,kernel,iterations = 1)


cv2.imshow('BW2', bw2)   
cv2.imshow('BW1', bw1)   
cv2.imshow('BW', bw)   

cv2.waitKey(0)
cv2.destroyAllWindows()
