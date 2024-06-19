import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Photos/PICT0001.jpeg')

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply blur to reduce noise and improve blob detection
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Apply binary thresholding to obtain a binary image
_, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY)

# Apply morphological operations to enhance blob detection
kernel = np.ones((3, 3), np.uint8)
cleaned = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# Set up the SimpleBlobDetector parameters
params = cv.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 100  # Adjust minArea as needed
params.maxArea = 3000  # Adjust maxArea as needed

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5  # Adjust minCircularity as needed

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8  # Adjust minConvexity as needed

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1  # Adjust minInertiaRatio as needed

# Create a detector with the parameters
detector = cv.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(cleaned)

# Create a mask image where blobs are white on a black background
mask = np.zeros_like(gray)
for keypoint in keypoints:
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    diameter = int(keypoint.size)
    radius = diameter // 2
    # Draw filled white circle on mask
    cv.circle(mask, (x, y), radius, 255, -1)

# Invert the mask to use it as a background mask
background_mask = 255 - mask

# Apply the background mask to set gray areas to black in the original image
# Set pixels to black where background_mask is white
img[background_mask == 255] = 0

# Apply the mask to the original image to show blobs as white
result = np.zeros_like(img)
result[mask == 255] = 255  # Set pixels to white where mask is white
# Keep original image where mask is black
result[result == 0] = img[result == 0]

# Draw circles around each detected blob in red
for keypoint in keypoints:
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    diameter = int(keypoint.size)
    radius = diameter // 2
    cv.circle(result, (x, y), radius, (0, 0, 255), 2)  # Draw red circle

# Display the number of blobs detected
num_blobs = len(keypoints)
print(f'Number of blobs detected: {num_blobs}')

# Write the number of blobs on the image with red text at the top middle
text = f'Cells detected: {num_blobs}'
text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
text_x = (result.shape[1] - text_size[0]) // 2
cv.putText(result, text, (text_x, 50),
           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the image with blobs detected and the number of blobs
cv.imshow('Blobs', result)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()
