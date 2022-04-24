import numpy as np
import cv2
import imutils


def resize_with_aspect_ratio(image, longest_edge=1280):

    """
    Resize image while preserving aspect ratio

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    longest_edge (int): Number of pixels on the longest edge

    Returns
    -------
    image (numpy.ndarray of shape (resized_height, resized_width, 3)): Resized image array
    """

    height, width = image.shape[:2]

    if np.maximum(height, width) <= longest_edge:
        return image

    if height >= width:
        image = imutils.resize(image, height=longest_edge, inter=cv2.INTER_LANCZOS4)
    else:
        image = imutils.resize(image, width=longest_edge, inter=cv2.INTER_LANCZOS4)

    return image
