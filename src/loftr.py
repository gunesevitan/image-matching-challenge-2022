import torch


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
