"""Useful transforms for trimming scanned images."""
import cv2 as cv
import numpy as np

def rotate_crop_to_rect(rect: any, scale: int = 1):
    """
    Computes the transform needed to extract just rect from the larger image it's part of,
    with appropriate rotation and cropping.

    Will scale the given rect by _scale_ if provided.

    Returns an affine transform as well as the intended output (width, height)

    Usage:
      rect = cv.minAreaRect(points)
      M, output_dimensions = transforms.rotate_crop_to_rect(rect)  # This function
      warped = cv.warpAffine(img, M, output_dimensions)
    """
    # Identify points in the same order in the minimum bounding rectangle (src) and
    # the output image boundary (dst) to build an affine transform between them.
    corners = sorted(cv.boxPoints(rect), key=lambda p: p[0] + p[1])
    top_left = corners[0]
    top_right, bottom_left = sorted([corners[1], corners[2]], key=lambda p: p[1])
    src_points = np.array([
        top_left, top_right, bottom_left
    ]).astype(np.float32) * scale

    # rect is a RotatedRect, but we don't have good Python types for it.
    # It's made up of ((lowest vertext), (dimensions), angle).
    width = int(rect[1][0] * scale)
    height = int(rect[1][1] * scale)
    rotation_angle = rect[2]
    if rotation_angle > 80:
        width, height = height, width
    dst_points = np.array([
        [0, 0], [width-1, 0], [0, height-1],
    ]).astype(np.float32)

    return cv.getAffineTransform(src_points, dst_points), (width, height)
