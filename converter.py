# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:46:13 2019

@author: shirhe-lyh


Convert tensorflow weights to pytorch weights for DeepLab V3+ models.

Reference:
    https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/
        tf_to_pytorch/convert_tf_to_pt/load_tf_weights.py
"""

import numpy as np
import tensorflow as tf
import torch


_BLOCK_UNIT_COUNT_MAP = {
    'xception_41': [[3, 1], [1, 8], [2, 1]],
    'xception_65': [[3, 1], [1, 16], [2, 1]],
    'xception_71': [[5, 1], [1, 16], [2, 1]],
}


def load_param(checkpoint_path, conversion_map, model_name):
    """Load parameters according to conversion_map.
    
    Args:
        checkpoint_path: Path to tensorflow's checkpoint file.
        conversion_map: A dictionary with format 
            {pytorch tensor in a model: checkpoint variable name}
        model_name: The name of Xception model, only supports 'xception_41',
            'xception_65', or 'xception_71'.
    """
    for pth_param, tf_param_name in conversion_map.items():
        param_name_strs =  tf_param_name.split('_')
        if len(param_name_strs) > 1 and param_name_strs[1].startswith('flow'):
            tf_param_name = str(model_name) + '/' + tf_param_name
        tf_param = tf.train.load_variable(checkpoint_path, tf_param_name)
        if 'conv' in tf_param_name and 'weights' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            if 'depthwise' in tf_param_name:
                tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif 'depthwise_weights' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif tf_param_name.endswith('weights'):
            tf_param = np.transpose(tf_param)
        assert pth_param.size() == tf_param.shape, ('Dimension mismatch: ' + 
            '{} vs {}; {}'.format(pth_param.size(), tf_param.shape, 
                 tf_param_name))
        pth_param.data = torch.from_numpy(tf_param)


def convert(model, checkpoint_path):
    """Load Pytorch Xception from TensorFlow checkpoint file.
    
    Args:
        model: The pytorch Xception model, only supports 'xception_41',
            'xception_65', or 'xception_71'.
        checkpoint_path: Path to tensorflow's checkpoint file.
        
    Returns:
        The pytorch Xception model with pretrained parameters.
    """
    block_unit_counts = _BLOCK_UNIT_COUNT_MAP.get(
        model._feature_extractor.scope, None)
    if block_unit_counts is None:
        raise ValueError('Unsupported Xception model name.')
    flow_names = []
    block_indices = []
    unit_indices = []
    flow_names_unique = ['entry_flow', 'middle_flow', 'exit_flow']
    for i, [block_count, unit_count] in enumerate(block_unit_counts):
        flow_names += [flow_names_unique[i]] * (block_count * unit_count)
        for i in range(block_count):
            block_indices += [i + 1] * unit_count
            unit_indices += [j + 1 for j in range(unit_count)]
    
    conversion_map = {}
    # Feature extractor: Root block
    conversion_map_for_root_block = {
        model._feature_extractor._layers[0]._conv.weight: 
            'entry_flow/conv1_1/weights',
        model._feature_extractor._layers[0]._batch_norm.bias: 
            'entry_flow/conv1_1/BatchNorm/beta',
        model._feature_extractor._layers[0]._batch_norm.weight: 
            'entry_flow/conv1_1/BatchNorm/gamma',
        model._feature_extractor._layers[0]._batch_norm.running_mean: 
            'entry_flow/conv1_1/BatchNorm/moving_mean',
        model._feature_extractor._layers[0]._batch_norm.running_var: 
            'entry_flow/conv1_1/BatchNorm/moving_variance',
        model._feature_extractor._layers[1]._conv.weight: 
            'entry_flow/conv1_2/weights',
        model._feature_extractor._layers[1]._batch_norm.bias: 
            'entry_flow/conv1_2/BatchNorm/beta',
        model._feature_extractor._layers[1]._batch_norm.weight: 
            'entry_flow/conv1_2/BatchNorm/gamma',
        model._feature_extractor._layers[1]._batch_norm.running_mean: 
            'entry_flow/conv1_2/BatchNorm/moving_mean',
        model._feature_extractor._layers[1]._batch_norm.running_var: 
            'entry_flow/conv1_2/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_root_block)
    
    # Feature extractor: Xception block
    for i in range(len(model._feature_extractor._layers[2]._blocks)):
        block = model._feature_extractor._layers[2]._blocks[i]
        ind = [1, 3, 5]
        if len(block._separable_conv_block) < 6:
            ind = [0, 1, 2]
        for j in range(3):
            conversion_map_for_separable_block = {
                block._separable_conv_block[ind[j]]._conv_depthwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/depthwise_weights').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._conv_pointwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/weights').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.bias:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/beta').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/gamma').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.running_mean:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/moving_mean').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_depthwise.running_var:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_depthwise/BatchNorm/moving_variance').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.bias:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/beta').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.weight:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/gamma').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.running_mean:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/moving_mean').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
                block._separable_conv_block[ind[j]]._batch_norm_pointwise.running_var:
                    ('{}/block{}/unit_{}/xception_module/' +
                     'separable_conv{}_pointwise/BatchNorm/moving_variance').format(
                        flow_names[i], block_indices[i], unit_indices[i], j+1),
            }
            conversion_map.update(conversion_map_for_separable_block)
            
            if getattr(block, '_conv_skip_connection', None) is not None:
                conversion_map_for_shortcut = {
                    block._conv_skip_connection.weight:
                       ('{}/block{}/unit_{}/xception_module/shortcut/' +
                        'weights').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.bias:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/beta').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.weight:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/gamma').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.running_mean:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/moving_mean').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                    block._batch_norm_shortcut.running_var:
                        ('{}/block{}/unit_{}/xception_module/shortcut/' +
                         'BatchNorm/moving_variance').format(
                            flow_names[i], block_indices[i], unit_indices[i]),
                }
                conversion_map.update(conversion_map_for_shortcut)
        
    # Atrous Spatial Pyramid Pooling: Image feature
    branches = model._aspp._branches
    conversion_map_for_aspp_image_feature = {
        branches[0][1].weight: 'image_pooling/weights',
        branches[0][2].bias: 'image_pooling/BatchNorm/beta',
        branches[0][2].weight: 'image_pooling/BatchNorm/gamma',
        branches[0][2].running_mean: 'image_pooling/BatchNorm/moving_mean',
        branches[0][2].running_var: 'image_pooling/BatchNorm/moving_variance',
        branches[1][0].weight: 'aspp0/weights',
        branches[1][1].bias: 'aspp0/BatchNorm/beta',
        branches[1][1].weight: 'aspp0/BatchNorm/gamma',
        branches[1][1].running_mean: 'aspp0/BatchNorm/moving_mean',
        branches[1][1].running_var: 'aspp0/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_aspp_image_feature)
    
    # Atrous Spatial Pyramid Pooling: Atrous convolution
    for i in range(3):
        branch = branches[i+2][0]
        conversion_map_for_atrous_conv = {
            branch._conv_depthwise.weight:
                'aspp{}_depthwise/depthwise_weights'.format(i+1),
            branch._conv_pointwise.weight:
                'aspp{}_pointwise/weights'.format(i+1),
            branch._batch_norm_depthwise.bias: 
                'aspp{}_depthwise/BatchNorm/beta'.format(i+1),
            branch._batch_norm_depthwise.weight: 
                'aspp{}_depthwise/BatchNorm/gamma'.format(i+1),
            branch._batch_norm_depthwise.running_mean: 
                'aspp{}_depthwise/BatchNorm/moving_mean'.format(i+1),
            branch._batch_norm_depthwise.running_var: 
                'aspp{}_depthwise/BatchNorm/moving_variance'.format(i+1),
            branch._batch_norm_pointwise.bias: 
                'aspp{}_pointwise/BatchNorm/beta'.format(i+1),
            branch._batch_norm_pointwise.weight: 
                'aspp{}_pointwise/BatchNorm/gamma'.format(i+1),
            branch._batch_norm_pointwise.running_mean: 
                'aspp{}_pointwise/BatchNorm/moving_mean'.format(i+1),
            branch._batch_norm_pointwise.running_var: 
                'aspp{}_pointwise/BatchNorm/moving_variance'.format(i+1),
        }
        conversion_map.update(conversion_map_for_atrous_conv)
        
    # Atrous Spatial Pyramid Pooling: Concat projection
    conversion_map_for_concat_projection = {
        model._aspp._conv_concat[0].weight:
            'concat_projection/weights',
        model._aspp._conv_concat[1].bias:
            'concat_projection/BatchNorm/beta',
        model._aspp._conv_concat[1].weight:
            'concat_projection/BatchNorm/gamma',
        model._aspp._conv_concat[1].running_mean:
            'concat_projection/BatchNorm/moving_mean',
        model._aspp._conv_concat[1].running_var:
            'concat_projection/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_concat_projection)
    
    # Refine decoder: Feature projection
    conversion_map_for_decoder = {
        model._refine_decoder._decoder[0].weight:
            'decoder/feature_projection0/weights',
        model._refine_decoder._decoder[1].bias:
            'decoder/feature_projection0/BatchNorm/beta',
        model._refine_decoder._decoder[1].weight:
            'decoder/feature_projection0/BatchNorm/gamma',
        model._refine_decoder._decoder[1].running_mean:
            'decoder/feature_projection0/BatchNorm/moving_mean',
        model._refine_decoder._decoder[1].running_var:
            'decoder/feature_projection0/BatchNorm/moving_variance',
    }
    conversion_map.update(conversion_map_for_decoder)
    
    # Refine decoder: Concat
    layers = model._refine_decoder._concat_layers
    for i in range(2):
        layer = layers[i]
        conversion_map_decoder = {
            layer._conv_depthwise.weight:
                'decoder/decoder_conv{}_depthwise/depthwise_weights'.format(i),
            layer._conv_pointwise.weight:
                'decoder/decoder_conv{}_pointwise/weights'.format(i),
            layer._batch_norm_depthwise.bias: 
                'decoder/decoder_conv{}_depthwise/BatchNorm/beta'.format(i),
            layer._batch_norm_depthwise.weight: 
                'decoder/decoder_conv{}_depthwise/BatchNorm/gamma'.format(i),
            layer._batch_norm_depthwise.running_mean: 
                'decoder/decoder_conv{}_depthwise/BatchNorm/moving_mean'.format(i),
            layer._batch_norm_depthwise.running_var: 
                'decoder/decoder_conv{}_depthwise/BatchNorm/moving_variance'.format(i),
            layer._batch_norm_pointwise.bias: 
                'decoder/decoder_conv{}_pointwise/BatchNorm/beta'.format(i),
            layer._batch_norm_pointwise.weight: 
                'decoder/decoder_conv{}_pointwise/BatchNorm/gamma'.format(i),
            layer._batch_norm_pointwise.running_mean: 
                'decoder/decoder_conv{}_pointwise/BatchNorm/moving_mean'.format(i),
            layer._batch_norm_pointwise.running_var: 
                'decoder/decoder_conv{}_pointwise/BatchNorm/moving_variance'.format(i),
        }
        conversion_map.update(conversion_map_decoder)
        
    # Prediction logits
    conversion_map_for_logits = {
        model._logits_layer.weight: 'logits/semantic/weights',
        model._logits_layer.bias: 'logits/semantic/biases',
    }
    conversion_map.update(conversion_map_for_logits)
        
    # Load TensorFlow parameters into PyTorch model
    load_param(checkpoint_path, conversion_map, model._feature_extractor.scope)