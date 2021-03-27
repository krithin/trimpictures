import argparse
import random
import sys
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description="Trim and deskew an image.")
parser.add_argument("--quiet", action="store_true", default=False, help="Don't prompt for a keypress before saving.")
parser.add_argument("--process-scale", type=int, default=1, help="Scale intermediate processing images down by this integer factor to help previews fit on screen.")
parser.add_argument("input_filename")
parser.add_argument("output_filename")
args = parser.parse_args()

img = cv.imread(args.input_filename)
if img is None:
    sys.exit("Could not read the image.")

scale = args.process_scale
if scale != 1:
    resized = cv.resize(
        img,
        (int(img.shape[1] / scale), int(img.shape[0] / scale)),
        interpolation = cv.INTER_AREA
    )
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
if not args.quiet:
    for i in range(len(contours)):
        color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# Reject contours of length < 200 because they're probably bits of dirt
real_contours = [c for c in contours if cv.arcLength(c, False) > 200.0 / scale]
# Draw convex hull
hull = cv.convexHull(
    np.array([point for contour in real_contours for point in contour])
)
if not args.quiet:
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv.drawContours(drawing, [hull], -1, color)
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# Find the smallest (rotated) rectangle that bounds that convex hull
rect = cv.minAreaRect(hull)
print(args.input_filename, rect)
corners = cv.boxPoints(rect)
if not args.quiet:
    cv.drawContours(drawing, [np.int0(corners)], -1, (255,255,255))
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# Identify points in the same order in the minimum bounding rectangle (src) and
# the output image boundary (dst) to build an affine transform between them.
corners = sorted(corners, key=lambda p: p[0] + p[1])
top_left = corners[0]
top_right, bottom_left = sorted([corners[1], corners[2]], key=lambda p: p[1])
src_points = np.array([
    top_left, top_right, bottom_left
]).astype(np.float32) * scale

width, height = np.int0(rect[1]) * scale
rotation_angle = rect[2]
if rotation_angle > 80:
    width, height = height, width
dst_points = np.array([
    [0, 0], [width-1, 0], [0, height-1],
]).astype(np.float32)

M = cv.getAffineTransform(src_points, dst_points)

# Run the transform, rotating and cropping the input image.
warped = cv.warpAffine(img, M, (width, height))
cv.imwrite(args.output_filename, warped)
