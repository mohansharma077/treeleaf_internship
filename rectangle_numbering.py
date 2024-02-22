from PIL import Image
from IPython.display import display
import cv2

def imshow(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image[:, :, ::-1]
    return display(Image.fromarray(image))

import cv2
import numpy as np

mask = cv2.imread("Screenshot from 2024-02-22 21-22-06.png", 0)

output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Inverse threshold
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find contours (with hierarchy)
contours, [hist] = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

rectangles = []

for rect, (nxt, prv, first_child, parent) in zip(contours, hist):
    # If it's top-level contour
    if parent == 0:
        # Find index of the line contour
        _, _, line_index, _ = hist[first_child]

        # Get the line
        line = contours[line_index]


        _, (width, height), _ = cv2.minAreaRect(line)

        line_length = max(width, height)


        rectangle = cv2.minAreaRect(rect)
        box = cv2.boxPoints(rectangle)
        box = np.intp(box)

        box_center = rectangle[0]

        rectangles.append((line_length, box, box_center))

# Sort by line length
rectangles.sort()

# Draw the minAreaRects of the rectangles and the numbering
for index, (_, box, (x, y)) in enumerate(rectangles):
    cv2.drawContours(output, [box], 0, (0, 0, 255), 3)
    cv2.putText(output, f"{index +1}", (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 9)
    cv2.putText(output, f"{index + 1}", (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()