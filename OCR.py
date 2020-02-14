import cv2
import sys
import pytesseract
 

im = cv2.imread('Circuit.jpg')

cv2.imshow('BW', im)   

cv2.waitKey(0)
cv2.destroyAllWindows()
im = cv2.ximgproc.thinning(im, im, 0)

config = ('-psm 11')
text = pytesseract.image_to_string(im, config=config)

# Print recognized text
print(text)