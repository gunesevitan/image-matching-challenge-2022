import torch
import kornia
from kornia.feature import laf_from_center_scale_ori
from kornia_moons.feature import draw_LAF_matches
import matplotlib.pyplot as plt


def match(image1, image2, loftr_model, confidence_threshold=0.1):

    """
    Match given two images with each other using LoFTR model

    Parameters
    ----------
    image1 (torch.Tensor of shape (1, 1, height, width)): First image tensor
    image2 (torch.Tensor of shape (1, 1, height, width)): Second image tensor
    loftr_model (torch.nn.Module): LoFTR Model
    confidence_threshold (float): Threshold to filter out keypoints with low confidence

    Returns
    -------
    output (dictionary of
            keypoints1 (numpy.ndarray of shape (n_keypoints, 2)),
            keypoints2 (numpy.ndarray of shape (n_keypoints, 2)),
            confidences (numpy.ndarray of shape (n_keypoints))
    ): Matched keypoints from first and second image and their confidences
    """

    input_dict = {
        'image0': image1,
        'image1': image2
    }

    with torch.no_grad():
        correspondences = loftr_model(input_dict)

    output = {
        'keypoints1': correspondences['keypoints0'].cpu().numpy(),
        'keypoints2': correspondences['keypoints1'].cpu().numpy(),
        'confidences': correspondences['confidence'].cpu().numpy(),
    }

    keypoint_mask = output['confidences'] >= confidence_threshold
    output = {
        'keypoints1': output['keypoints1'][keypoint_mask],
        'keypoints2': output['keypoints2'][keypoint_mask],
        'confidences': output['confidences'][keypoint_mask],
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
