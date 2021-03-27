import argparse
import random
import sys
import cv2 as cv
import numpy as np

import transforms

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

# Find the smallest (rotated) rectangle that bounds all the real contours
points = np.array([point for contour in real_contours for point in contour])
rect = cv.minAreaRect(points)
print(args.input_filename, rect)
if not args.quiet:
    corners = cv.boxPoints(rect)
    cv.drawContours(drawing, [np.int0(corners)], -1, (255,255,255))
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

M, output_dimensions = transforms.rotate_crop_to_rect(rect, scale)

# Run the transform, rotating and cropping the input image.
warped = cv.warpAffine(img, M, output_dimensions)
cv.imwrite(args.output_filename, warped)
