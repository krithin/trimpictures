import random
import sys
import cv2 as cv
import numpy as np

input_filename = sys.argv[1]
output_filename = sys.argv[2]

img = cv.imread(input_filename)
if img is None:
    sys.exit("Could not read the image.")

resized = cv.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)), interpolation = cv.INTER_AREA)
cv.imshow("Display window", resized)
k = cv.waitKey(0)

canny_output = cv.Canny(resized, 100, 200)
cv.imshow("Edges", canny_output)
k = cv.waitKey(0)

# Draw contours
contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv.drawContours(drawing, contours, i, color)
cv.imshow('Contours', drawing)
k = cv.waitKey(0)

# Draw convex hull
hull = cv.convexHull(np.array([point for contour in contours if cv.arcLength(contour, False) > 40 for point in contour]))
color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
cv.drawContours(drawing, [hull], -1, color)
cv.imshow('Contours', drawing)
k = cv.waitKey(0)

rect = cv.minAreaRect(hull)
print(input_filename, rect)
box = np.int0(cv.boxPoints(rect))
cv.drawContours(drawing, [box], -1, (255,255,255))
cv.imshow('Contours', drawing)
k = cv.waitKey(0)

# get width and height of the detected rectangle
width = int(rect[1][0])
height = int(rect[1][1])

src_pts = box.astype("float32") * 4
# coordinate of the points in box points after the rectangle has been
# straightened
dst_pts = np.array([[0, height * 4 -1],
                    [0, 0],
                    [width * 4 -1, 0],
                    [width * 4 -1, height * 4 -1]], dtype="float32")

# the perspective transformation matrix
M = cv.getPerspectiveTransform(src_pts, dst_pts)

# directly warp the original image to get the straightened rectangle
warped = cv.warpPerspective(img, M, (width * 4, height * 4))

cv.imwrite(output_filename, warped)
#cv.imshow("Warped", warped)
#cv.waitKey(0)
