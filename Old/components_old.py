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


print('Original Dimensions : ',bw.shape)

width = 500
height = 500
dim = (width, height)
 
# resize image
bw = cv2.resize(bw, dim, interpolation = cv2.INTER_AREA)
src = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
(thresh, bw) = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Resized Dimensions : ',bw.shape)


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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
bw4 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

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
bw3 = wires
kernel = np.ones((3,3),np.uint8)
wires = cv2.dilate(wires,kernel,iterations = 2)


components = bw
#components = cv2.ximgproc.thinning(bw, components, 0)

components = cv2.subtract(components, wires)

(thresh, components) = cv2.threshold(components, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#kernel = np.ones((3,3),np.uint8)
#components = cv2.dilate(components,kernel,iterations = 1)

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
bk1 = components.copy()

cnts = cv2.findContours(bk1, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
bw1 = wires
kernel = np.ones((3,3),np.uint8)
bw1 = cv2.dilate(bw1,kernel,iterations = 2)

#bk1 = bw

for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
        #(255,255,255)
    if w <= 5 and h <= 5:
        cv2.rectangle(bk1, (x, y), (x+w, y+h), (0,255,0), -1)
    
    else:
        '''
        config = ('-psm 10')
        crop_img = bw[y-2:y+h+2, x-2:x+w+2]
        crop_img = (255 - crop_img)
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(0)
        text = pytesseract.image_to_string(crop_img, config=config)
        #print(text)
        
        '''
        reset = 0
  
        kernele = []
        for k1 in range(0, h):
            for k2 in range(0, w):
                if (k1 == 0 or k1 == h-1):
                    kernele.append(1)
                elif k2 == 0 or k2 == w-1:
                    kernele.append(1)
                else:
                    kernele.append(0)
        kernel = np.asarray(kernele)
        kernel = kernel.reshape(h,w)
        #kernel = np.transpose(kernel)
        check_part = bw1[y:y+h, x:x+w]
        check_part =  np.multiply(kernel, check_part)
        if (check_part.any() == 0):
            reset = 1
        ''' 
        coodsx = []
        coodsy = []
        reset = 0
        a = 5
        for i in range(x-a, x+w+a):
            coodsx.append(i)
            coodsy.append(y-a)
        for i in range(x-a, x+w+a):
            coodsx.append(i)
            coodsy.append(y+a)
        for i in range(y-a, y+h+a):
            coodsx.append(i)
            coodsy.append(x-a)
        for i in range(y-a, y+h+a):
            coodsx.append(i)
            coodsy.append(x+a)
        l = len(coodsx)
        #print (coodsx)
        #print (len(coodsx), len(coodsy))
        for i in range(0, l):
            if (wires[coodsx[i], coodsy[i]] == 255):
                reset = 1
        '''
        cv2.rectangle(bw3, (x, y), (x+w, y+h), (255,255,255), 2)
        if (reset == 1):
            #cv2.rectangle(bw1, (x, y), (x+w, y+h), (0,255,0), -1)
            cv2.rectangle(bk1, (x, y), (x+w, y+h), (0,255,0), -1)

cv2.imshow("Com", components) 
cv2.imshow("BW1", bw1) 
cv2.imshow("Wires", bw3)
cv2.imshow("Bk1", bk1)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5,5),np.uint8)
bk1 = cv2.dilate(bk1,kernel,iterations = 2)
cv2.imshow("Bk1", bk1)
cv2.waitKey(0)
cv2.destroyAllWindows()

bk3 = bk1.copy()

cnts = cv2.findContours(bk3, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#comp_cood
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(src, (x, y), (x+w, y+h), (0,255,0), 2)

bk3 = bw4.copy()

cnts = cv2.findContours(bk3, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(src, (x, y), (x+w, y+h), (0,0,255), 2)


cv2.imshow("Final", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

#code for node detection
