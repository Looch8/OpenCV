import cv2 as cv

# Read an image
img = cv.imread('Photos/PICT0010.jpeg')

cv.imshow('PICT0010', img)

# Rescale image size


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = (frame.shape[0] * scale)
    dimensions = (width, height)

    # Resize image by particular dimension
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


cv.waitKey(0)
