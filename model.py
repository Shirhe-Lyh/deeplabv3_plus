# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:24:03 2019

@author: shirhe-lyh


Implementation of DeepLabV3+:
    Encoder-Decoder with atrous separable convolution for semantic image
    segmentation, Liang-Chieh Chen, YuKun Zhu, George Papandreou, Florian
    Schroff, Hartwig Adam, arxiv:1802.02611 (https://arxiv.org/abs/1802.02611).
    
Official implementation:
    https://github.com/tensorflow/models/tree/master/research/deeplab
"""

import os
import torch

import common
import core.feature_extractor as extractor


_BATCH_NORM_PARAMS = {
    'momentum': 0.9997,
    'eps': 1e-5,
    'affine': True,
}


class DeepLab(torch.nn.Module):
    """Implementation of DeepLab V3+."""
    
    def __init__(self, feature_extractor, model_options, 
                 fine_tune_batch_norm=False):
        """Constructor.
        
        Args:
            feature_extractor: The backbone of DeepLab model.
            model_options: A ModelOptions instance to configure models.
            fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
        """
        super(DeepLab, self).__init__()
        self._model_options = model_options
        
        # Feature extractor
        self._feature_extractor = feature_extractor

        # Atrous spatial pyramid pooling
        self._aspp = AtrousSpatialPyramidPooling(
            in_channels=feature_extractor.out_channels,
            crop_size=model_options.crop_size,
            output_stride=model_options.output_stride,
            atrous_rates=model_options.atrous_rates,
            use_bounded_activation=model_options.use_bounded_activation,
            add_image_level_feature=model_options.add_image_level_feature,
            image_pooling_stride=model_options.image_pooling_stride,
            image_pooling_crop_size=model_options.image_pooling_crop_size,
            aspp_with_separable_conv=model_options.aspp_with_separable_conv)
        
        # Refine by decoder
        self._refine_decoder = None
        if model_options.decoder_output_stride:
            self._refine_decoder = RefineDecoder(
                crop_size=model_options.crop_size,
                decoder_output_stride=model_options.decoder_output_stride[0],
                decoder_use_separable_conv=model_options.decoder_use_separable_conv,
                model_variant=model_options.model_variant,
                use_bounded_activation=model_options.use_bounded_activation)
            
        # Branch logits
        num_classes = model_options.outputs_to_num_classes[common.OUTPUT_TYPE]
        self._logits_layer = torch.nn.Conv2d(
            in_channels=256, out_channels=num_classes,
            kernel_size=model_options.logits_kernel_size)
        
        # Intermediate feature
        model_variant = model_options.model_variant
        decoder_output_stride = model_options.decoder_output_stride[0]
        feature_names = extractor.networks_to_feature_maps[
            model_variant][extractor.DECODER_END_POINTS][decoder_output_stride]
        self._feature_name = '{}/{}'.format(model_variant, feature_names[0])
            
        if not fine_tune_batch_norm:
            self._freeze_batch_norm_params()
                    
    def forward(self, x):
        features = self._feature_extractor(x)
        features = self._aspp(features)
        inter_features = self._feature_extractor.end_points()[self._feature_name]
        if self._refine_decoder is not None:
            features = self._refine_decoder(features, inter_features)
        logits = self._logits_layer(features)
        _, _, height, width = x.shape
        if self._model_options.prediction_with_upsampled_logits:
            logits = resize_bilinear(logits, (height, width))
        return logits
    
    def _freeze_batch_norm_params(self):
        """Freeze the batch norm parameters."""
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        
        
class AtrousSpatialPyramidPooling(torch.nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    
    def __init__(self, 
                 in_channels,
                 out_channels=256,
                 output_stride=16,
                 crop_size=[513, 513],
                 atrous_rates=[6, 12, 18], 
                 use_bounded_activation=False,
                 add_image_level_feature=True, 
                 image_pooling_stride=[1, 1],
                 image_pooling_crop_size=None,
                 aspp_with_separable_conv=True):
        """Constructor.
        
        Args:
            in_channels: Number of input filters.
            out_channels: Number of filters in the 1x1 pointwise convolution.
            atrous_rates: A list of atrous convolution rates for ASPP.
            use_bounded_activation: Whether or not to use bounded activations.
            crop_size: A tuple [crop_height, crop_width].
            image_pooling_crop_size: Image pooling crop size [height, width] 
                used in the ASPP module.
        """
        super(AtrousSpatialPyramidPooling, self).__init__()
        activation_fn = (
            torch.nn.ReLU6(inplace=False) if use_bounded_activation else 
            torch.nn.ReLU(inplace=False))
        
        depth = out_channels
        branches = []
        if add_image_level_feature:
            layers = []
            if crop_size is not None:
                # If image_pooling_crop_size is not specified, use crop_size
                if image_pooling_crop_size is None:
                    image_pooling_crop_size = crop_size
                pool_height = scale_dimension(image_pooling_crop_size[0],
                                              1. / output_stride)
                pool_width = scale_dimension(image_pooling_crop_size[1],
                                             1. / output_stride)
                layers += [torch.nn.AvgPool2d(
                    (pool_height, pool_width), image_pooling_stride)]
                resize_height = scale_dimension(
                    crop_size[0], 1. / output_stride)
                resize_width = scale_dimension(
                    crop_size[1], 1. / output_stride)
            else:
                # If crop_size is None, we simply do global pooling
                layers += [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))]
                resize_height, resize_width = None, None
            self._resize_height = resize_height
            self._resize_width = resize_width
            layers += [torch.nn.Conv2d(in_channels, depth, 1, bias=False),
                       torch.nn.BatchNorm2d(depth, **_BATCH_NORM_PARAMS),
                       activation_fn]
            branches.append(torch.nn.Sequential(*layers))
        
        # Employ a 1x1 convolution.
        branches.append(torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, depth, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(depth, **_BATCH_NORM_PARAMS),
            activation_fn))
        
        if atrous_rates:
            # Employ 3x3 convolutions with different atrous rates.
            for i, rate in enumerate(atrous_rates):
                layers = []
                if aspp_with_separable_conv:
                    layers += [
                        SplitSeparableConv2d(in_channels, depth, rate=rate,
                                             activation_fn=activation_fn)]
                else:
                    layers += [
                        torch.nn.Conv2d(in_channels, depth, kernel_size=3,
                                        rate=rate, padding=1, bias=False),
                        torch.nn.BatchNorm2d(depth, **_BATCH_NORM_PARAMS),
                        activation_fn]
                branches.append(torch.nn.Sequential(*layers))
        self._branches = torch.nn.Sequential(*branches) 
        
        # Merge branch logits
        self._conv_concat = torch.nn.Sequential(
            torch.nn.Conv2d(len(self._branches) * depth,
                            depth, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(depth, **_BATCH_NORM_PARAMS),
            activation_fn)
        self._dropout = torch.nn.Dropout2d(p=0.9, inplace=True)
    
    def forward(self, x):
        branch_logits = []
        conv_branches = self._branches
        if len(self._branches) > 4:
            conv_branches = self._branches[1:]
            image_feature = self._branches[0](x)
            if self._resize_height is None:
                _, _, height, width = x.size()
                self._resize_height, self._resize_width = height, width
            image_feature = resize_bilinear(
                image_feature, (self._resize_height, self._resize_width))
            branch_logits.append(image_feature)
        branch_logits += [branch(x) for branch in conv_branches]
        cancat_logits = torch.cat(branch_logits, dim=1)
        cancat_logits = self._conv_concat(cancat_logits)
        cancat_logits = self._dropout(cancat_logits)
        return cancat_logits
    
    
class SplitSeparableConv2d(torch.nn.Module):
    """Splits a seperable conv2d into depthwise and pointwise conv2d."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, rate=1,
                 use_batch_norm=True, activation_fn=None):
        """Constructor.
        
        Args:
            in_channels: Number of input filters.
            out_channels: Number of filters in the 1x1 pointwise convolution.
            kernel_size: A list of length 2: [kernel_height, kernel_width]
                of the filters. Can be an int if both values are the same.
            rate: Atrous convolution rate for the depthwise convolution.
            with_batch_norm: Whether or not to use batch normalization.
            activation_fn: The activation function to be applied.
        """
        super(SplitSeparableConv2d, self).__init__()
        # For the shape of output of Conv2d, see details at:
        # https://pytorch.org/docs/stable/nn.html#convolution-layers
        # Here, we assume that floor(padding) = padding
        padding = (kernel_size - 1) * rate // 2
        self._conv_depthwise = torch.nn.Conv2d(in_channels,
                                               in_channels,
                                               kernel_size=kernel_size,
                                               dilation=rate,
                                               padding=padding,
                                               groups=in_channels,
                                               bias=False)
        self._conv_pointwise = torch.nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=1,
                                               bias=False)
        self._batch_norm_depthwise = None
        self._batch_norm_pointwise = None
        if use_batch_norm:
            self._batch_norm_depthwise = torch.nn.BatchNorm2d(
                num_features=in_channels, **_BATCH_NORM_PARAMS)
            self._batch_norm_pointwise = torch.nn.BatchNorm2d(
                num_features=out_channels, **_BATCH_NORM_PARAMS)
        self._activation_fn = activation_fn
        
    def forward(self, x):
        x = self._conv_depthwise(x)
        if self._batch_norm_depthwise is not None:
            x = self._batch_norm_depthwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        x = self._conv_pointwise(x)
        if self._batch_norm_pointwise is not None:
            x = self._batch_norm_pointwise(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
    
    
class RefineDecoder(torch.nn.Module):
    """Adds the decoder to obtain sharper segmentation results."""
    
    def __init__(self, aspp_channels=256, crop_size=None,
                 decoder_output_stride=None, decoder_use_separable_conv=False,
                 model_variant=None, use_bounded_activation=False):
        """Constructor.
        
        Args:
            aspp_channels: The out channels of ASPP.
            crop_size: A tuple [crop_height, crop_width] specifying whole
                patch crop size.
            decoder_output_stride: An integer specifying the output stride of
                low-level features used in the decoder module.
            decoder_use_separable_conv: Employ separable convolution for 
                decoder or not.
            model_variant: Model variant for feature extractor.
            use_bounded_activation: Whether or not to use bounded activations.
                Bounded activations better lend themselves to quantized
                inference.
            
        Raises:
            ValueError: If crop_size is None.
        """
        super(RefineDecoder, self).__init__()
        
        if crop_size is None:
            raise ValueError('crop_size must be provided when using decoder.')
            
        self._crop_size = crop_size
        self._output_stride = decoder_output_stride
        activation_fn = (
            torch.nn.ReLU6(inplace=False) if use_bounded_activation else 
            torch.nn.ReLU(inplace=False))
        self._decoder = torch.nn.Sequential(
            torch.nn.Conv2d(extractor.feature_out_channels_map[model_variant], 
                            out_channels=48, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(num_features=48, **_BATCH_NORM_PARAMS),
            activation_fn)
        concat_layers = []
        decoder_depth = 256
        if decoder_use_separable_conv:
            concat_layers += [
                SplitSeparableConv2d(in_channels=aspp_channels + 48,
                                     out_channels=decoder_depth, rate=1,
                                     activation_fn=activation_fn),
                SplitSeparableConv2d(in_channels=decoder_depth, 
                                     out_channels=decoder_depth,
                                     rate=1, activation_fn=activation_fn)]
        else:
            concat_layers += [
                torch.nn.Conv2d(in_channels=aspp_channels + 48,
                                out_channels=decoder_depth, kernel_size=3,
                                padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=decoder_depth,
                                     **_BATCH_NORM_PARAMS),
                activation_fn,
                torch.nn.Conv2d(in_channels=decoder_depth,
                                out_channels=decoder_depth,
                                kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(num_features=decoder_depth,
                                     **_BATCH_NORM_PARAMS),
                activation_fn]
        self._concat_layers = torch.nn.Sequential(*concat_layers)
        
    def forward(self, x, features):
        decoder_features_list = [x]
        decoder_features_list.append(self._decoder(features))
        # Determine the output size
        decoder_height = scale_dimension(self._crop_size[0],
                                         1.0 / self._output_stride)
        decoder_width = scale_dimension(self._crop_size[1],
                                        1.0 / self._output_stride)
        # Resize to decoder_height/decoder_width
        for j, feature in enumerate(decoder_features_list):
            decoder_features_list[j] = resize_bilinear(
                feature, (decoder_height, decoder_width))
        x = self._concat_layers(torch.cat(decoder_features_list, dim=1))
        return x
    

def scale_dimension(dim, scale):
    """Scales the input dimension.
    
    Args:
        dim: Input dimension (a scalar).
        scale: The amount of scaling applied to the input.
        
    Returns:
        scaled dimension.
    """
    return int((float(dim) - 1.0) * scale + 1.0)


def resize_bilinear(images, size):
    """Returns resized images.
    
    Args:
        images: A tensor of size [batch, height_in, width_in, channels].
        size: A tuple (height, width).
        
    Returns:
        A tensor of shape [batch, height_out, height_width, channels].
    """
    return torch.nn.functional.interpolate(
        images, size, mode='bilinear', align_corners=True)
    
    
def deeplab(num_classes, crop_size=[513, 513], atrous_rates=[6, 12, 18],
            output_stride=16, fine_tune_batch_norm=False,
            pretrained=True, pretained_num_classes=21,
            checkpoint_path='./pretrained_models/deeplabv3_pascal_trainval.pth'):
    """DeepLab v3+ for semantic segmentation."""
    outputs_to_num_classes = {'semantic': num_classes}
    model_options = common.ModelOptions(outputs_to_num_classes,
                                        crop_size=crop_size,
                                        atrous_rates=atrous_rates,
                                        output_stride=output_stride)
    feature_extractor = extractor.feature_extractor(
        model_options.model_variant, pretrained=False,
        output_stride=model_options.output_stride)
    model = DeepLab(feature_extractor, model_options, fine_tune_batch_norm)
    
    if pretrained:
        _load_state_dict(model, num_classes, pretained_num_classes,
                         checkpoint_path)
    return model


def _load_state_dict(model, num_classes, pretained_num_classes, 
                     checkpoint_path):
    """Load pretrained weights."""
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        if num_classes is None or num_classes != pretained_num_classes:
            state_dict.pop('_logits_layer.weight')
            state_dict.pop('_logits_layer.bias')
        model.load_state_dict(state_dict, strict=False)
        print('Load pretrained weights successfully.')
    else:
        raise ValueError('`checkpoint_path` does not exist.')
