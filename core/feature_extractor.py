# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:40:01 2019

@author: shirhe-lyh


Extracts features for different models.
"""

import torch

from core import xception


# A map from network name to network function
networks_map = {
    'xception_41': xception.xception_41,
    'xception_65': xception.xception_65,
    'xception_71': xception.xception_71,
}

# A map from network name to out_channels
out_channels_map = {
    'xception_41': 2048,
    'xception_65': 2048,
    'xception_71': 2048,
}

# Names for end point features
DECODER_END_POINTS = 'decoder_end_points'

# A dictionary from network name to a map of end point features
networks_to_feature_maps = {
    'xception_41': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
    'xception_65': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block2/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
    'xception_71': {
        DECODER_END_POINTS: {
            4: ['entry_flow/block3/unit_1/xception_module/'
                'separable_conv2_pointwise'],
        },
    },
}

feature_out_channels_map = {
    'xception_41': 256,
    'xception_65': 256,
    'xception_71': 256,
}

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.

# Mean pixel vale.
_MEAN_RGB = [123.15, 115.90, 103.06]


def feature_extractor(model_variant, output_stride=8, pretrained=True):
    """Extracts features by the particular model_variant.
    
    Args:
        model_variant: The name of feature extractor.
        output_stride: The ratio of input to output spatial resolution.
        pretrained: Whether or not to load pretrained parameters.
        
    Returns:
        extractor: The particular network to extract features.
        out_channels: The out channels of extractor.
        
    Raises:
        If model_variant not in networks_map.
    """
    if model_variant not in networks_map:
        raise ValueError('Unsupported network %s.' % model_variant)

    extractor = networks_map[model_variant](num_classes=None,
                                            global_pool=False,
                                            output_stride=output_stride,
                                            pretrained=pretrained)
    out_channels = out_channels_map[model_variant]
    extractor.out_channels = out_channels
    return extractor