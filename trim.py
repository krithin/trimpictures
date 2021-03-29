import argparse
import random
import sys
import cv2 as cv
import numpy as np

import transforms

parser = argparse.ArgumentParser(description="Trim and deskew an image.")
parser.add_argument("--quiet", action="store_true", default=False, help="Don't prompt for a keypress before saving.")
parser.add_argument("--process-scale", type=int, default=1, help="Scale intermediate processing images down by this integer factor to help previews fit on screen.")
parser.add_argument("--num-splits", type=int, default=1, help="Split the raw scanned image into this many sections before trimming and rotating each one.")
parser.add_argument("input_filename")
parser.add_argument("output_filename")
args = parser.parse_args()

def index_output_filename(index: int) -> str:
    """Generates a specifc output filename for each picture when splitting a scanned image."""
    base = args.output_filename
    split = base.rindex('.')
    return f"{base[:split]}_{index}{base[split:]}"

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
if not args.quiet:
    drawing = np.zeros((*canny_output.shape, 3), dtype=np.uint8)
    for i in range(len(contours)):
        colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        cv.drawContours(drawing, contours, i, colour)
    cv.imshow("Contours", drawing)
    k = cv.waitKey(0)

# Reject contours of length < 200 because they're probably bits of dirt
real_contours = [c for c in contours if cv.arcLength(c, False) > 200.0 / scale]
if not args.quiet:
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(real_contours)):
        colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        cv.drawContours(drawing, real_contours, i, colour)
    cv.imshow("Cleaned Contours", drawing)
    k = cv.waitKey(0)

point_sets = transforms.partition_contours(real_contours, args.num_splits)
if not args.quiet:
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for point_set in point_sets:
        cluster_colour = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        for point in point_set:
            drawing[point[1]][point[0]] = cluster_colour
    cv.imshow("Clusters", drawing)
    k = cv.waitKey(0)

for i, point_set in enumerate(point_sets):
    # Find the smallest (rotated) rectangle that bounds all the real contours
    rect = cv.minAreaRect(point_set)
    print(args.input_filename, rect)
    if not args.quiet:
        corners = cv.boxPoints(rect)
        cv.drawContours(drawing, [np.int0(corners)], -1, (255,255,255))
        cv.imshow("Clusters", drawing)
        k = cv.waitKey(0)

    # Compute and run a transform that rotates and crops the input image.
    M, output_dimensions = transforms.rotate_crop_to_rect(rect, scale)
    warped = cv.warpAffine(img, M, output_dimensions)

    output_filename = args.output_filename if args.num_splits == 1 else index_output_filename(i)
    print(output_filename)
    cv.imwrite(output_filename, warped)
