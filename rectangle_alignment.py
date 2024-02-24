import cv2
import numpy as np

# Load the image
image = cv2.imread("Screenshot from 2024-02-22 21-22-06.png", cv2.IMREAD_COLOR)
image = image[10:-10, 10:-10]

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
ret, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around contours and rectify them
for contour in contours:
    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(contour)

    # Extract the rectangle from the original image
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation to rectify the rectangle
    warped = cv2.warpPerspective(image, M, (width, height))

    # Display the rectified rectangle
    cv2.imshow("Rectified Rectangles", warped)
    cv2.waitKey(0)

cv2.destroyAllWindows()
