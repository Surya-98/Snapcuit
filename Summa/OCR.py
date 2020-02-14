import cv2
import sys
import pytesseract
 

im = cv2.imread('Circuit.jpg')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(~gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im = bw
cv2.imshow('BW', im)   

cv2.waitKey(0)
cv2.destroyAllWindows()
im = cv2.ximgproc.thinning(im, im, 0)

config = ('-psm 11')
text = pytesseract.image_to_string(im, config=config)

# Print recognized text
print(text)