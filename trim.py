import argparse
import random
import sys
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description="Trim and deskew an image.")
parser.add_argument("--quiet", action="store_true", default=False, help="Don't prompt for a keypress before saving.")
parser.add_argument("--process-scale", type=int, default=1, help="Scale intermediate processing images down by this integer factor for speed.")
parser.add_argument("input_filename")
parser.add_argument("output_filename")
args = parser.parse_args()

img = cv.imread(args.input_filename)
if img is None:
    sys.exit("Could not read the image.")

scale = args.process_scale
if scale != 1:
    resized = cv.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)), interpolation = cv.INTER_AREA)
else:
    resized = img
if not args.quiet:
    cv.imshow("Display window", resized)
    k = cv.waitKey(0)

canny_output = cv.Canny(resized, 100, 200)
if not args.quiet:
    cv.imshow("Edges", canny_output)
    k = cv.waitKey(0)

# Draw contours
contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv.drawContours(drawing, contours, i, color)
if not args.quiet:
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# Reject contours of length < 200 because they're probably bits of dirt
real_contours = [c for c in contours if cv.arcLength(c, False) > 200.0 / scale]
# Draw convex hull
hull = cv.convexHull(
    np.array([point for contour in real_contours for point in contour])
)
color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
cv.drawContours(drawing, [hull], -1, color)
if not args.quiet:
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

rect = cv.minAreaRect(hull)
print(args.input_filename, rect)
box = np.int0(cv.boxPoints(rect))
cv.drawContours(drawing, [box], -1, (255,255,255))
if not args.quiet:
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# get width and height of the detected rectangle
width = int(rect[1][0])
height = int(rect[1][1])

src_pts = box.astype("float32") * scale
# coordinate of the points in box points after the rectangle has been
# straightened
dst_pts = np.array([[0, height * scale -1],
                    [0, 0],
                    [width * scale -1, 0],
                    [width * scale -1, height * scale -1]], dtype="float32")

# the perspective transformation matrix
M = cv.getPerspectiveTransform(src_pts, dst_pts)

# directly warp the original image to get the straightened rectangle
warped = cv.warpPerspective(img, M, (width * scale, height * scale))

cv.imwrite(args.output_filename, warped)
#cv.imshow("Warped", warped)
#cv.waitKey(0)
