import numpy as np
import cv2
from scipy import signal
from matplotlib import pyplot as plt
im = cv2.imread("1.png", 0)
im1 = cv2.imread("5.png", 0)
ret, thresh1 = cv2.threshold(im,128,255,cv2.THRESH_BINARY_INV)
ret1, thresh2 = cv2.threshold(im1,128,255,cv2.THRESH_BINARY_INV)
rows,cols = thresh2.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
thresh2 = cv2.warpAffine(thresh2,M,(cols,rows))
k = signal.correlate2d(thresh1, thresh2)
k = k.astype(float)
k = np.multiply(np.divide(k,np.max(k)),255)
k = k.astype(int)
plt.imshow(k, cmap= 'gray')
plt.xticks([]), plt.yticks([])
plt.show()