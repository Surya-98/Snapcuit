import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
from sklearn.metrics.pairwise import euclidean_distances

src = cv2.imread('Circuit - Copy.jpg')
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


cnts1 = cv2.findContours(nodes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts1 = imutils.grab_contours(cnts1)

node_infos = []
k = 0
for c in cnts1:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    a = 10
    #cv2.rectangle(src, (x, y), (x+w, y+h), (0,255,255), 2)
    coods = [x+(w/2), y+(h/2)]
    #cv2.rectangle(src, (coods[0]-a, coods[1]-a), (coods[0]+a, coods[1]+a), (0,255,255), 2)
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



horizontal = bw
vertical = bw
#this number rejects other small lines 
horizontal_size = horizontal.shape[0]/ 18
vertical_size = vertical.shape[1]/ 18

kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))

horizontal = cv2.erode(horizontal, kernel_h, iterations=1) 
horizontal = cv2.dilate(horizontal, kernel_h, iterations=1) 

vertical = cv2.erode(vertical, kernel_v, iterations=1) 
vertical = cv2.dilate(vertical, kernel_v, iterations=1) 

wires = cv2.add(horizontal, vertical)
wires = cv2.ximgproc.thinning(wires, wires, 0)
bw3 = wires
kernel = np.ones((4,4),np.uint8)
wires = cv2.dilate(wires,kernel,iterations = 2)

cv2.imshow("wire", wires)
cv2.imwrite("wires.jpg", wires)
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
'''
cv2.imshow('BW', bw)
cv2.imshow('Wires', wires)
cv2.imshow('Components', components)   
#cv2.imshow('Components1', components1)   
#cv2.imshow('Components2', components2)   

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# find contours in the thresholded image, then initialize the
# digit contours lists
bk1 = components.copy()

cnts = cv2.findContours(bk1, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
bw1 = wires
kernel = np.ones((3,3),np.uint8)
bw1 = cv2.dilate(bw1,kernel,iterations = 2)
cv2.imwrite("Compstoo.png", bk1)
bk5 = bw
ind = 0
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
            cv2.rectangle(bk5, (x, y), (x+w, y+h), (0,255,0), -1)
            
'''
cv2.imshow("Com", components) 
cv2.imshow("BW1", bw1) 
cv2.imshow("Wires", bw3)
cv2.imshow("Bk1", bk1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.imwrite("Components.png", bk1)

kernel = np.ones((5,5),np.uint8)
bk1 = cv2.dilate(bk1,kernel,iterations = 2)
'''
cv2.imshow("Bk1", bk1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
bk3 = bk1.copy()

cnts = cv2.findContours(bk3, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#comp_cood
comp_center = []

for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    comp_center.append([x+w/2, y+h/2])
    cv2.rectangle(src, (x, y), (x+w, y+h), (0,255,0), 2)
    ind = ind + 1
    crop_img = bk5[y-2:y+h+2, x-2:x+w+2]
    crop_img = (255 - crop_img)
    name = str(ind)+".png"
    cv2.imwrite(name, crop_img)
bk3 = nodes.copy()


cv2.imshow("Final", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

#code for node detection


#direction 0 - u, 1 - right, 2 - down, 3 left

print(src[303, 373])


def move (position, diri):
    comp = 0
    nod = 0
    if diri == 0:
        position[1] = position[1] - 1
    if diri == 1:
        position[0] = position[0] + 1
    if diri == 2:
        position[1] = position[1] + 1
    if diri == 3:
        position[0] = position[0] - 1

    if np.array_equal(src[position[1], position[0]], [0, 255, 0]):
        comp = 1
    if np.array_equal(src1[position[1], position[0]], [0, 0, 255]):
        nod = 1
    
    a = 10
    #cv2.rectangle(src, (x, y), (x+w, y+h), (0,255,255), 2)
    coods = position
    #cv2.rectangle(src, (coods[0]-a, coods[1]-a), (coods[0]+a, coods[1]+a), (0,255,255), 2)
    crop_img = bw[coods[1]-a:coods[1]+a, coods[0]-a:coods[0]+a]
    #crop_img = cv2.ximgproc.thinning(crop_img, crop_img, 0)
    
    #up, right, down, left
    up = crop_img[0:5, 0:19]    
    right = crop_img[0:19, 15:19]
    down = crop_img[15:19, 0:19]
    left = crop_img[0:19, 0:5]
    direct = []
    direct.append(up)
    direct.append(right)
    direct.append(down)
    direct.append(left)
    
    if np.count_nonzero(direct[diri]) == 0:
        if np.count_nonzero(direct[((diri+1)%4)]) != 0:
            diri = (diri+1)%4
        else: 
            diri = (diri-1)%4
    return diri, comp, nod, position


comp_center = np.asarray(comp_center)
comp_dict = {}

for i in range(0, len(comp_center)):
    k = str(i)
    comp_dict[k] = comp_center[i]

node = node_infos
m = 0

cnts = cv2.findContours(bk3, cv2.RETR_EXTERNAL, 
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


for i in node:
    m = m + 1
    src1 = src.copy()
    m1 = 0
    for c in cnts:
        m1 = m1 + 1
        # compute the bounding box of the contour
        if m1 != m:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(src1, (x, y), (x+w, y+h), (0,0,255), 2)
    temp = i[1]
    #print "i=", i
    for j in range(0,4):
        cur_pos = i[0]
        cur_pos = np.asarray(cur_pos)
        #print "j=", j
        nod = 0
        comp = 0
        if temp[j] == 1:
            diri = j
            
            while(comp ==0 and nod == 0):
                #print "comp = ", comp, "nod =", nod, "curr_pos =", cur_pos
                diri, comp, nod, cur_pos = move(cur_pos, diri)
                #print "direction = ",diri
                #print "position = ", cur_pos
            
            if comp == 1:
                cur_pos = [cur_pos]
                dist = euclidean_distances(comp_center, cur_pos)
                dist = np.absolute(dist)
                #print dist
                pos = np.argmin(dist)
                print "component = ", pos
                print "Node = ", m
                temp1 = comp_dict[str(pos)]
                temp2 = []
                temp2.append(temp1)
                temp2.append(m)
                comp_dict[str(pos)] = temp2
            node[m-1][1][j] = 0
            '''
            if nod == 1:
                print m
                '''

cv2.imwrite("Wires.png", wires)
cv2.imwrite("Nodes.png", nodes)


for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(src, (x, y), (x+w, y+h), (0,0,255), 2)
cv2.imwrite("Final.png", src)
print node[0]
print comp_dict