import torch
from kornia.feature import \
    LocalFeature, PassLAF, LAFOrienter, PatchDominantGradientOrientation, OriNet, LAFAffNetShapeEstimator,\
    KeyNetDetector, LAFDescriptor, HardNet8, HyNet, TFeat, SOSNet, get_laf_center

import settings


class LocalFeatureDetectorDescriptor(LocalFeature):

    def __init__(self,
                 orientation_module_name, orientation_module_weights,
                 affine_module_name, affine_module_weights,
                 detector_module_name, detector_module_weights,
                 descriptor_module_name, descriptor_module_weights,
                 device):

        """
        Module that combines local feature detector and descriptor

        Parameters
        ----------
        orientation_module_name (str): Name of the orientation module
        orientation_module_weights (str): Path of the orientation module weights relative to models/lfdd
        affine_module_name (str): Name of the affine module
        affine_module_weights (str): Path of the affine module weights relative to models/lfdd
        detector_module_name (str): Name of the detector module
        detector_module_weights (str): Path of the detector module weights relative to models/lfdd
        descriptor_module_name (str): Name of the descriptor module
        descriptor_module_weights (str): Path of the descriptor module weights relative to models/lfdd
        device (torch.device): Location of model

        Returns
        -------
        lafs (torch.Tensor of shape (1, n_detections, 2, 3)): Detected local affine frames
        responses (torch.Tensor of shape (1, n_detections)): Response function values for corresponding lafs
        descriptors (torch.Tensor of shape (1, n_detections, n_dimensions)): Local descriptors
        """

        device = torch.device(device)

        # Instantiate specified orientation module
        if orientation_module_name == 'PassLAF':
            orientation_module = PassLAF()
        elif orientation_module_name == 'OriNet':
            orientation_module = LAFOrienter(angle_detector=OriNet(pretrained=(orientation_module_weights is None)))
        elif orientation_module_name == 'PatchDominantGradientOrientation':
            orientation_module = LAFOrienter(angle_detector=PatchDominantGradientOrientation())
        else:
            orientation_module = None

        # Instantiate specified affine module
        if affine_module_name == 'LAFAffNetShapeEstimator':
            affine_module = LAFAffNetShapeEstimator(pretrained=(affine_module_weights is None))
        else:
            affine_module = None

        # Instantiate specified detector module
        if detector_module_name == 'KeyNetDetector':
            detector_module = KeyNetDetector(pretrained=(detector_module_weights is None), ori_module=orientation_module, aff_module=affine_module).to(device)
        else:
            detector_module = None

        # Load pretrained weights for the detector module
        if orientation_module_weights is not None:
            detector_module.ori.angle_detector.load_state_dict(torch.load(settings.MODELS / 'lfdd' / orientation_module_weights)['state_dict'])
        if affine_module_weights is not None:
            detector_module.aff.load_state_dict(torch.load(settings.MODELS / 'lfdd' / affine_module_weights)['state_dict'])
        if detector_module_weights is not None:
            detector_module.model.load_state_dict(torch.load(settings.MODELS / 'lfdd' / detector_module_weights)['state_dict'])

        # Instantiate specified descriptor module
        if descriptor_module_name == 'HardNet8':
            descriptor_module = LAFDescriptor(HardNet8(pretrained=(descriptor_module_weights is None))).to(device)
        elif descriptor_module_name == 'HyNet':
            descriptor_module = LAFDescriptor(HyNet(pretrained=(descriptor_module_weights is None))).to(device)
        elif descriptor_module_name == 'TFeat':
            descriptor_module = LAFDescriptor(TFeat(pretrained=(descriptor_module_weights is None))).to(device)
        elif descriptor_module_name == 'SOSNet':
            descriptor_module = LAFDescriptor(SOSNet(pretrained=(descriptor_module_weights is None))).to(device)
        else:
            descriptor_module = None

        # Load pretrained weights for the descriptor module
        if descriptor_module_weights is not None:
            descriptor_module.descriptor.load_state_dict(torch.load(settings.MODELS / 'lfdd' / descriptor_module_weights))

        super().__init__(detector_module, descriptor_module)


def _extract_features(image, feature_extractor):

    """
    Extract local feature descriptors on given image with given model

    Parameters
    ----------
    image (torch.Tensor of shape (1, 1, height, width)): Image tensor
    feature_extractor (torch.nn.Module): Local feature detector and descriptor model

    Returns
    -------
    lafs (torch.Tensor of shape (1, n_detections, 2, 3)): Detected local affine frames
    responses (torch.Tensor of shape (n_detections)): Response function values for corresponding lafs
    descriptors (torch.Tensor of shape (n_detections, n_dimensions)): Local descriptors
    keypoints (numpy.ndarray of shape (n_detections, 2)): Keypoints
    """

    with torch.no_grad():
        lafs, responses, descriptors = feature_extractor(image)

    responses = torch.squeeze(responses, dim=0).detach().cpu().numpy()
    descriptors = torch.squeeze(descriptors, dim=0)
    keypoints = get_laf_center(lafs)
    keypoints = keypoints.detach().cpu().numpy().reshape(-1, 2)
    lafs = lafs.detach().cpu().numpy()

    return lafs, responses, descriptors, keypoints


def _match_descriptors(descriptors1, descriptors2, descriptor_matcher):

    """
    Match descriptors with nearest neighbor algorithm

    Parameters
    ----------
    descriptors1 (torch.Tensor of shape (n_detections, n_dimensions)): Descriptors from first image
    descriptors2 (torch.Tensor of shape (n_detections, n_dimensions)): Descriptors from second image
    descriptor_matcher (torch.nn.Module): Descriptor matcher model

    Returns
    -------
    distances (numpy.ndarray of shape (n_matches)): Distances of matching descriptors
    indexes (numpy.ndarray of shape (n_matches, 2)): Indexes of matching descriptors
    """

    with torch.no_grad():
        distances, indexes = descriptor_matcher(descriptors1, descriptors2)

    distances = distances.detach().cpu().numpy().reshape(-1)
    indexes = indexes.detach().cpu().numpy()

    return distances, indexes


def match(image1, image2, feature_extractor, descriptor_matcher, distance_threshold=0.1):

    """
    Match given two images with each other using given feature extractor and descriptor matcher

    Parameters
    ----------
    image1 (torch.Tensor of shape (1, 1, height, width)): First image tensor
    image2 (torch.Tensor of shape (1, 1, height, width)): Second image tensor
    feature_extractor (torch.nn.Module): Local feature detector and descriptor model
    descriptor_matcher (torch.nn.Module): Descriptor matcher model
    distance_threshold (float): Threshold to filter out keypoints with low distance

    Returns
    -------
    output (dictionary of
            keypoints1 (numpy.ndarray of shape (n_keypoints, 2)),
            keypoints2 (numpy.ndarray of shape (n_keypoints, 2)),
            distances (numpy.ndarray of shape (n_keypoints))
    ): Matched keypoints from first and second image and their distances
    """

    _, _, descriptors1, keypoints1 = _extract_features(image=image1, feature_extractor=feature_extractor)
    _, _, descriptors2, keypoints2 = _extract_features(image=image2, feature_extractor=feature_extractor)
    distances, indexes = _match_descriptors(descriptors1=descriptors1, descriptors2=descriptors2, descriptor_matcher=descriptor_matcher)

    output = {
        'keypoints1': keypoints1[indexes[:, 0]],
        'keypoints2': keypoints2[indexes[:, 1]],
        'distances': distances
    }

    keypoint_mask = output['distances'] >= distance_threshold
    output = {
        'keypoints1': output['keypoints1'][keypoint_mask],
        'keypoints2': output['keypoints2'][keypoint_mask],
        'distances': output['distances'][keypoint_mask],
    }

    return output
