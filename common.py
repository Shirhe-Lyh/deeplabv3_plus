# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:23:03 2019

@author: shirhe-lyh


Provides flags that are common to scripts.

Common flags from tf_weights_to_pth/model_test.py/train.py are collected in 
this script.
"""

import argparse
import collections
import copy
import json
import torch

parser = argparse.ArgumentParser(
    description='Set default values to common arguments.',
    add_help=False)

# Model dependent flags
parser.add_argument('--logits_kernel_size', default=1, type=int, 
                    help='The kernel size for the convolutional kernel that '
                    'generates logits.')

# When using 'mobilenet_v2', we set atrous_rates=decoder_output_stride=None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates=[6, 12, 18] (output_stride 16) and decoder_output_stride=4.
# See core/feature_extractor.py for supported model variants.
parser.add_argument('--model_variant', default='xception_65', type=str,
                    help='DeepLab model variant.')

parser.add_argument('--add_image_level_feature', default=True, type=bool,
                    help='Add image level feature.')

parser.add_argument('--image_pooling_crop_size', default=None, type=list,
                    help='Image pooling crop size [height, width] used in '
                    'the ASPP module. When value is None, the model performs '
                    'image pooling with `crop_size`. This flag is useful '
                    'when one likes to use different image pooling sizes.')

parser.add_argument('--image_pooling_stride', default=[1, 1], type=list,
                    help='Image pooling stride [height, width] used in the '
                    'ASPP image pooling.')

parser.add_argument('--aspp_with_batch_norm', default=True, type=bool,
                    help='Use batch norm parameters for ASPP or not.')

parser.add_argument('--aspp_with_separable_conv', default=True, type=bool,
                    help='Use separable convolution for ASPP or not.')

# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50, 101}_beta' checkpoints.
parser.add_argument('--multi_grid', default=None, type=list,
                    help='Employ a hierarchy of atrous rates for ResNet.')

parser.add_argument('--depth_multiplier', default=1.0, type=float,
                    help='Multiplier for the depth (number of channels) for '
                    'all convolution ops used in MobileNet.')

parser.add_argument('--divisible_by', default=None, type=int,
                    help='An integer that ensures the layer # channels are '
                    'divisible by this value. used in MobileNet.')

# For 'xception_65', use decoder_output_stride=4. For 'mobilenet_v2', use
# decoder_output_stride=None.
parser.add_argument('--decoder_output_stride', default=[4], type=list,
                    help='Comma-separated list of integers with the number '
                    'specifying output stride of low-level features at each '
                    'network level. Current semantic segmentation '
                    'implementation assumes at most one output stride (i.e., '
                    'either None or a list with only one element)')

parser.add_argument('--decoder_use_separable_conv', default=True, type=bool,
                    help='Employ separable convolution for decoder or not.')

parser.add_argument('--merge_method', default='max', type=str,
                    choices=['max', 'avg'],
                    help='Scheme to merge multi scale features.')

parser.add_argument('--prediction_with_upsampled_logits', 
                    default=True, type=bool,
                    help='When performing prediction, there are two options: '
                    '(1) bilinear upsampling the logits followed by argmax, '
                    'or (2) argmax followed by nearest upsampling the '
                    'predicted labels. The second option may introduce some '
                    ' "blocking effect", but it is more computationaly '
                    'efficient. Currently, prediction_with_upsampled_logits'
                    '=False is only supported for single-scale inference.')

parser.add_argument('--dense_prediction_cell_json', default='', type=str,
                    help='A JSON file that specifies the dense prediction '
                    'cell.')

parser.add_argument('--nas_stem_output_num_conv_filters', default=20, type=int,
                    help='Number of filters of the stem output tensor in NAS '
                    'models.')

parser.add_argument('--use_bounded_activation', default=False, type=bool,
                    help='Whether or not to use bounded activations. Bounded '
                    'activations better lend themselves to quantized '
                    'inferece.')


#----------------------------------------------
# Arguments for tf_weights_to_pth.py
#----------------------------------------------
parser.add_argument('--tf_checkpoint_path', type=str, default=None,
                    help='Path to checkpoint file.')

parser.add_argument('--output_dir', type=str, default='./pretrained_models',
                    help='Where the output pytorch model file is stored.')

parser.add_argument('--output_name', type=str, 
                    default='deeplabv3_pascal_trainval.pth',
                    help='Name of the stored pytorch model file.')

parser.add_argument('--pretained_num_classes', type=int, default=21,
                    help='Number of classes of the pretrained models.')


#----------------------------------------------
# Arguments for model_test.py
#----------------------------------------------
parser.add_argument('--checkpoint_path', type=str, 
                    default='./pretrained_models/deeplabv3_pascal_trainval.pth',
                    help='Path to checkpoint file.')

parser.add_argument('--image_path', type=str, 
                    default='./test/COCO_val2014_000000000294.jpg',
                    help='Path to a test image.')

parser.add_argument('--colormap_type', type=str, default='pascal',
                    choices=['pascal', 'cityscapes'],
                    help='Visualization colormap type.')

parser.add_argument('--val_output_stride', type=int, default=8,
                    help='The ratio of input to output spatial resolution.')

parser.add_argument('--val_atrous_rates', type=int, nargs='+',
                    default=[12, 24, 36],
                    help='A list of atrous convolution rates for ASPP.')


#----------------------------------------------
# Arguments for train.py
#----------------------------------------------
# Train hyperparameters
parser.add_argument('--gpu_indices', type=int, nargs='+', default=[0, 1, 2, 3],
                    help='The gpu indices to be used.')
parser.add_argument('--num_epochs', type=int, default=3, 
                    help='Number of epochs.')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of classes.')
parser.add_argument('--batch_size', type=int, default=3,
                    help='Batch size per gpu.')
parser.add_argument('--lr_decay', type=float, default=0.8,
                    help='The decay rate of learning rate.')
parser.add_argument('--decay_step', type=int, default=1000,
                    help='Decay learning rate every decay_step.')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Initial learning rate.')
parser.add_argument('--model_dir', type=str, default='./models',
                    help='Where the trained model file is stored.')

# The feature extractor, see common.py
parser.add_argument('--crop_size', type=int, nargs='+', default=[560, 560],
                    help='The input size')
parser.add_argument('--output_stride', type=int, default=16,
                    help='The ratio of input to output spatial resolution.')
parser.add_argument('--atrous_rates', type=int, nargs='+',
                    default=[6, 12, 18],
                    help='A list of atrous convolution rates for ASPP.')

# Data augmentation parameters
parser.add_argument('--min_scale_factor', type=float, default=0.5,
                    help='Minimum scale factor.')
parser.add_argument('--max_scale_factor', type=float, default=2.0,
                    help='Minimum scale factor.')
parser.add_argument('--scale_factor_step_size', type=float, default=0.25,
                    help='Minimum scale factor.')

# Image correspondence: image_paths, matting_paths, alpha_paths, mask_paths
parser.add_argument('--annotation_path', type=str,
                    default='./data/train.txt',
                    help='Path to images corresponding file.')
parser.add_argument('--root_dir', type=str, default=None,
                    help='The root dir: xxx/Matting_Human_Half.')

FLAGS = parser.parse_args()

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_stem_output_num_conv_filters',
        'use_bounded_activation'
    ])):
    """Immutable class to hold model options."""
    
    __slots__ = ()
    
    def __new__(cls,
                outputs_to_num_classes,
                crop_size=None,
                atrous_rates=None,
                output_stride=8,
                preprocessed_images_dtype=torch.float32):
        """Constructor to set default values.
        
        Args:
            outputs_to_num_classes: A dictionary from output type to the 
                number of classes. For example, for the task of semantic
                segmentation with 21 semantic classes, we would have 
                outputs_to_num_classes['semantic']=21.
                crop_size: A tuple [crop_height, crop_width].
                atrous_rates: A list of atrous convolution rates for ASPP.
                output_stride: The ratio of input to output spatial resolution.
                preprocessed_images_dtype: The type after the preprocessing 
                    function.
        
        Returns:
            A new ModelOptions instance.
        """
        dense_prediction_cell_config = None
        if FLAGS.dense_prediction_cell_json:
            with open(FLAGS.dense_prediction_cell_json, 'r') as f:
                dense_prediction_cell_config = json.load(f)
        decoder_output_stride = None
        if FLAGS.decoder_output_stride:
            decoder_output_stride = FLAGS.decoder_output_stride
            if (sorted(decoder_output_stride, reverse=True) != 
                decoder_output_stride):
                raise ValueError('Decoder output stride need to be sorted in '
                                 'the descending order.')
        image_pooling_crop_size = None
        if FLAGS.image_pooling_crop_size:
            image_pooling_crop_size = FLAGS.image_pooling_crop_size
        image_pooling_stride = [1, 1]
        if FLAGS.image_pooling_stride:
            image_pooling_stride = FLAGS.image_pooling_stride
        return super(ModelOptions, cls).__new__(
            cls, outputs_to_num_classes, crop_size, atrous_rates, 
            output_stride, preprocessed_images_dtype, FLAGS.merge_method,
            FLAGS.add_image_level_feature,
            image_pooling_crop_size,
            image_pooling_stride,
            FLAGS.aspp_with_batch_norm,
            FLAGS.aspp_with_separable_conv, FLAGS.multi_grid, 
            decoder_output_stride,
            FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size,
            FLAGS.model_variant, FLAGS.depth_multiplier, FLAGS.divisible_by,
            FLAGS.prediction_with_upsampled_logits, 
            dense_prediction_cell_config,
            FLAGS.nas_stem_output_num_conv_filters,
            FLAGS.use_bounded_activation)
        
    def __deepcopy__(self, memo):
        return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                            self.crop_size,
                            self.atrous_rates,
                            self.output_stride,
                            self.preprocessed_images_dtype)