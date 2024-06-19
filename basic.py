import cv2 as cv
img = cv.imread('Photos/PICT0010.jpeg')

cv.imshow('PICT0010', img)

# Converting to grayscale

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

inverted = cv.bitwise_not(img)
cv.imshow('Inverted', inverted)

cv.waitKey(0)
cv.destroyAllWindows()
