import pathlib
import cv2
import torch
import kornia
from kornia.feature import laf_from_center_scale_ori
from kornia_moons.feature import draw_LAF_matches
import matplotlib.pyplot as plt

import settings
import image_utils


def get_image_tensor(image, longest_edge, device):

    """
    Load image and move it to specified device

    Parameters
    ----------
    image (str or numpy.ndarray of shape (height, width, 3)): Image path relative to data directory or image array
    longest_edge (int): Number of pixels on the longest edge
    device (torch.device): Location of image tensor

    Returns
    -------
    image (torch.Tensor of shape (1, 3, height, width)): Image tensor
    """

    if isinstance(image, pathlib.Path) or isinstance(image, str):
        # Read image from the given path
        image_path = image
        image = cv2.imread(str(settings.DATA / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image_utils.resize_with_aspect_ratio(image=image, longest_edge=longest_edge)
    image = kornia.image_to_tensor(image, False).float() / 255.
    image = image.to(device)

    return image


def match(image1, image2, matcher, confidence_threshold=0.1):

    """
    Match given two with images with each other using matcher

    Parameters
    ----------
    image1 (torch.Tensor of shape (1, 3, height, width)): First image tensor
    image2 (torch.Tensor of shape (1, 3, height, width)): Second image tensor
    matcher (torch.nn.Module): LoFTR Model
    confidence_threshold (float): Threshold to filter out keypoints with low confidence

    Returns
    -------
    output (dictionary of
            keypoints0 (numpy.ndarray of shape (n_keypoints, 2)),
            keypoints1 (numpy.ndarray of shape (n_keypoints, 2)),
            confidence (numpy.ndarray of shape (n_keypoints))
    ): Keypoints from first and second image and their confidences
    """

    input_dict = {
        'image0': kornia.color.rgb_to_grayscale(image1),
        'image1': kornia.color.rgb_to_grayscale(image2)
    }

    with torch.no_grad():
        correspondences = matcher(input_dict)

    output = {
        'keypoints0': correspondences['keypoints0'].cpu().numpy(),
        'keypoints1': correspondences['keypoints1'].cpu().numpy(),
        'confidence': correspondences['confidence'].cpu().numpy(),
    }

    keypoint_mask = output['confidence'] >= confidence_threshold
    output = {
        'keypoints0': output['keypoints0'][keypoint_mask],
        'keypoints1': output['keypoints1'][keypoint_mask],
        'confidence': output['confidence'][keypoint_mask],
    }

    return output


def visualize_matches(image1, image2, keypoints1, keypoints2, inliers, path=None):

    """
    Visualize matched keypoints of an image pair

    Parameters
    ----------
    image1 (torch.Tensor of shape (1, 3, height, width)): First image tensor
    image2 (torch.Tensor of shape (1, 3, height, width)): Second image tensor
    keypoints1 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from first image
    keypoints2 (numpy.ndarray of shape (n_keypoints, 2)): Keypoints from second image
    inliers (numpy.ndarray of shape (n_keypoints)): Inlier mask
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    draw_LAF_matches(
        lafs1=laf_from_center_scale_ori(
            torch.from_numpy(keypoints1).view(1, -1, 2),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1)
        ),
        lafs2=laf_from_center_scale_ori(
            torch.from_numpy(keypoints2).view(1, -1, 2),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1)
         ),
        tent_idxs=torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        img1=kornia.tensor_to_image(image1),
        img2=kornia.tensor_to_image(image2),
        inlier_mask=inliers,
        draw_dict={
            'inlier_color': (0.2, 1, 0.2),
            'tentative_color': None,
            'feature_color': (0.2, 0.5, 1),
            'vertical': False
        }
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
