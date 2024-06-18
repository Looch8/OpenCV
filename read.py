import cv2 as cv

# Read an image
img = cv.imread('Photos/PICT0010.jpeg')

cv.imshow('PICT0010', img)

cv.waitKey(0)
