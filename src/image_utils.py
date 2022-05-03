import pathlib
import numpy as np
import cv2
import kornia

import settings


def resize_with_aspect_ratio(image, longest_edge=840):

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


def get_image_tensor(image, device, longest_edge=840, normalize='max_pixel_value', grayscale=False):

    """
    Load image and move it to specified device

    Parameters
    ----------
    image (str or numpy.ndarray of shape (height, width, 3)): Image path relative to data directory or image array
    device (torch.device): Location of image tensor
    longest_edge (int): Number of pixels on the longest edge
    normalize (str): Normalization method (max_pixel_value or dataset_statistics)
    grayscale (bool): Whether to convert RGB image to grayscale

    Returns
    -------
    image (torch.Tensor of shape (1, 1 or 3, height, width)): Image tensor
    """

    if isinstance(image, pathlib.Path) or isinstance(image, str):
        # Read image from the given path
        image_path = image
        image = cv2.imread(str(settings.DATA / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if longest_edge is not None:
        image = resize_with_aspect_ratio(image=image, longest_edge=longest_edge)

    if normalize == 'max_pixel_value':
        image = image / 255.
    elif normalize == 'dataset_statistics':
        image = (image - np.array([128.01498015, 128.7181146, 126.2084097])) / np.array([68.81605695, 70.68666045, 76.6746954])

    image = kornia.image_to_tensor(image, False).float()

    if grayscale:
        image = kornia.color.rgb_to_grayscale(image)

    image = image.to(device)

    return image
