import glob
import cv2
import os

width = 28
height = 28
dim = (width, height)

for i in range (1, 17):
    a = str(i) + ".png"
    print a
    im = cv2.imread(a, cv2.IMREAD_UNCHANGED)
    #bk = im.copy()
    #bk[mask == 255] = (255, 255, 255)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.resize(bw, dim, interpolation = cv2.INTER_AREA)
    (thresh, bw) = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    a = str(i) + ".png"
    a = 'Ground' + a
    #a = '~/Desktop/Snapcuit/Prj/Dataset/' + a
    print a
    cv2.imwrite(a, bw)
    #cv2.imshow('im', bw)   

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
