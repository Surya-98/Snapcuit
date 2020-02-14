import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

src = cv2.imread('Circuit.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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


cnts = cv2.findContours(nodes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

node_infos = []
k = 0
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    a = 10
    #cv2.rectangle(src, (x, y), (x+w, y+h), (0,255,255), 2)
    coods = [x+(w/2), y+(h/2)]
    cv2.rectangle(src, (coods[0]-a, coods[1]-a), (coods[0]+a, coods[1]+a), (0,255,255), 2)
    crop_img = bw[coods[1]-a:coods[1]+a, coods[0]-a:coods[0]+a]
    crop_img = cv2.ximgproc.thinning(crop_img, crop_img, 0)
    
    #up, right, down, left
    up = crop_img[0:5, 0:19]    
    right = crop_img[0:19, 15:19]
    down = crop_img[15:19, 0:19]
    left = crop_img[0:19, 0:5]
    #print("u",up,"r", right, "d", down,"l", left)
    node_type = []
    if np.count_nonzero(up) == 0:
        node_type.append(0)
    else:
        node_type.append(1)

    if np.count_nonzero(right) == 0:
        node_type.append(0)
    else:
        node_type.append(1)

    if np.count_nonzero(down) == 0:
        node_type.append(0)
    else:
        node_type.append(1)

    if np.count_nonzero(left) == 0:
        node_type.append(0)
    else:
        node_type.append(1)
    
    node_info = []

    node_info.append(coods)
    node_info.append(node_type)
    node_infos.append(node_info)

    #crop_img = (255 - crop_img)
    #name = str(k) + ".jpg" 
    #cv2.imwrite(name, crop_img)
    k = k+1
    #print node_type
    #cv2.imshow('SRC', crop_img)   
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print(node_infos)
cv2.imshow('Nodes', nodes)   
cv2.imshow('BW', bw)
cv2.imshow('SRC', src)   

cv2.waitKey(0)
cv2.destroyAllWindows()

