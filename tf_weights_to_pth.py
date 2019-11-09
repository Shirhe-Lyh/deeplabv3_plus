# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:05:17 2019

@author: shirhe-lyh
"""

import argparse
import os
import torch

import common
import converter
import model

parser = argparse.ArgumentParser(
    description='Convert tensorflow weights to pytorch.',
    parents=[common.parser])

FLAGS = parser.parse_args()


if __name__ == '__main__':
    checkpoint_path = FLAGS.tf_checkpoint_path
    output_dir = FLAGS.output_dir
    output_name = FLAGS.output_name
    pretained_num_classes = FLAGS.pretained_num_classes
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)
    
    deeplab = model.deeplab(num_classes=pretained_num_classes, pretrained=False)
    converter.convert(deeplab, checkpoint_path)
        
    # Save pytorch file
    torch.save(deeplab.state_dict(), output_path)
    print('Save model to: ', output_path)
