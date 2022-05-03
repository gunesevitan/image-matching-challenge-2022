import torch
from kornia.feature import \
    LocalFeature, PassLAF, LAFOrienter, PatchDominantGradientOrientation, OriNet,\
    LAFAffNetShapeEstimator, KeyNetDetector, LAFDescriptor, HardNet8, HyNet, TFeat, SOSNet

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
        orientation_module_weights (str): Path of the orientation module weights relative to models/deep_local_feature_detector_descriptors
        affine_module_name (str): Name of the affine module
        affine_module_weights (str): Path of the affine module weights relative to models/deep_local_feature_detector_descriptors
        detector_module_name (str): Name of the detector module
        detector_module_weights (str): Path of the detector module weights relative to models/deep_local_feature_detector_descriptors
        descriptor_module_name (str): Name of the descriptor module
        descriptor_module_weights (str): Path of the descriptor module weights relative to models/deep_local_feature_detector_descriptors
        """

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
            affine_module = LAFAffNetShapeEstimator(pretrained=(affine_module_weights is None)).eval()
        else:
            affine_module = None

        # Instantiate specified detector module
        if detector_module_name == 'KeyNetDetector':
            detector_module = KeyNetDetector(pretrained=(detector_module_weights is None), ori_module=orientation_module, aff_module=affine_module).to(device)
        else:
            detector_module = None

        # Load pretrained weights for the detector module
        detector_module.ori.angle_detector.load_state_dict(torch.load(settings.MODELS / 'deep_local_feature_detector_descriptors' / orientation_module_weights)['state_dict'])
        detector_module.aff.load_state_dict(torch.load(settings.MODELS / 'deep_local_feature_detector_descriptors' / affine_module_weights)['state_dict'])
        detector_module.model.load_state_dict(torch.load(settings.MODELS / 'deep_local_feature_detector_descriptors' / detector_module_weights)['state_dict'])

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
        descriptor_module.descriptor.load_state_dict(torch.load(settings.MODELS / 'deep_local_feature_detector_descriptors' / detector_module_weights))

        super().__init__(detector_module, descriptor_module)
