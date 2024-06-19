import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Photos/PICT0010.jpeg')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply G blur to reduce noise and improve blob detection
blurred = cv.GaussianBlur(gray, (7, 7), 0)

# Apply Otsu's thresholding to obtain a binary image
_, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Invert the binary image
thresh = 255 - thresh

# Apply morphological operations to enhance blob detection
kernel = np.ones((5, 5), np.uint8)
cleaned = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# Set up the SimpleBlobDetector parameters
params = cv.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 100  # Adjust minArea as needed
params.maxArea = 10000  # Adjust maxArea as needed

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(cleaned)

# Draw blobs on the original image
blobs = img.copy()
for keypoint in keypoints:
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    diameter = int(keypoint.size)
    radius = diameter // 2
    cv.circle(blobs, (x, y), radius, (0, 0, 255), 2)

# Display the number of blobs detected
num_blobs = len(keypoints)
print(f'Number of blobs detected: {num_blobs}')

# Write the number of blobs on the image
text = f'Blobs detected: {num_blobs}'
cv.putText(blobs, text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the original image with blobs detected and the number of blobs
cv.imshow('Blobs', blobs)


cv.waitKey(0)
cv.destroyAllWindows()
