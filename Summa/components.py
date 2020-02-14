import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
src = cv2.imread('Circuit.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = np.ones((2,2),np.uint8)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


'''
print('Original Dimensions : ',bw.shape)
 
width = 450
height = 450
dim = (width, height)
 
# resize image
bw = cv2.resize(bw, dim, interpolation = cv2.INTER_AREA)
(thresh, bw) = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Resized Dimensions : ',bw.shape)

'''
'''
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#10 for Circuit
#the kernal size should be small for small circuits
bw1 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
#detects nodes
cv2.imshow('BW1', bw1)   
cv2.imshow('BW', bw)   

cv2.waitKey(0)
cv2.destroyAllWindows()
#bw1 = cv2.medianBlur(bw1,9)
#bw1 = cv2.medianBlur(bw1,9)

bw = cv2.subtract(bw, bw1)
#(thresh, bw) = cv2.threshold(~bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
'''

horizontal = bw
vertical = bw
#this number rejects other small lines 
horizontal_size = horizontal.shape[0]/ 15
vertical_size = vertical.shape[1]/ 15
    
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))

horizontal = cv2.erode(horizontal, kernel_h, iterations=1) 
horizontal = cv2.dilate(horizontal, kernel_h, iterations=1) 

vertical = cv2.erode(vertical, kernel_v, iterations=1) 
vertical = cv2.dilate(vertical, kernel_v, iterations=1) 

wires = cv2.add(horizontal, vertical)

wires = cv2.ximgproc.thinning(wires, wires, 0)
kernel = np.ones((3,3),np.uint8)
wires = cv2.dilate(wires,kernel,iterations = 1)


components = bw
components = cv2.ximgproc.thinning(bw, components, 0)
cv2.imwrite("Thin.jpg", components)

components = cv2.subtract(components, wires)

(thresh, components) = cv2.threshold(components, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
components = cv2.dilate(components,kernel,iterations = 1)

'''
components1 = cv2.blur(components,(10,10))
(thresh, components1) = cv2.threshold(components1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
components2 = cv2.medianBlur(components,5)
'''
cv2.imshow('BW', bw)
cv2.imshow('Wires', wires)
cv2.imshow('Components', components)   
#cv2.imshow('Components1', components1)   
#cv2.imshow('Components2', components2)   

cv2.waitKey(0)
cv2.destroyAllWindows()
# find contours in the thresholded image, then initialize the
# digit contours lists

cnts = cv2.findContours(components, cv2.RETR_TREE, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
bk1 = components
cv2.imshow("BW", bw)
cv2.waitKey(0)
#bk1 = bw
# loop over the digit area candidates
i = 0
for c in cnts:
    i = i + 1
	# compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
	    #(255,255,255)
    if w <= 5 and h <= 5:
        cv2.rectangle(bk1, (x, y), (x+w, y+h), (0,0,0), -1)
    else:
        config = ('--psm 10')
        crop_img = bw[y-2:y+h+2, x-2:x+w+2]
        crop_img = (255 - crop_img)
        #crop_img = cv2.blur(crop_img,(3,3))
        #cv2.imshow("cropped", crop_img)
        name = str(i) + ".jpg" 
        cv2.imwrite(name, crop_img)
        #cv2.waitKey(0)
        text = pytesseract.image_to_string(crop_img, config=config)
        print(text)
    #    cv2.rectangle(bk1, (x, y), (x+w, y+h), (255,0,0), 2)

		    
cv2.imshow("BW", bk1)
cv2.waitKey(0)
cv2.destroyAllWindows()