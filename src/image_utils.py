import numpy as np
import cv2


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
    scale = longest_edge / max(height, width)
    image = cv2.resize(image, dsize=(int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)

    return image
