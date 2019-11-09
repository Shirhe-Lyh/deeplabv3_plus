# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:41:10 2019

@author: shirhe-lyh
"""

import argparse
import cv2
import numpy as np
import os
import torch

import common
import model
from utils import save_annotation

parser = argparse.ArgumentParser(
    description='Convert tensorflow weights to pytorch.',
    parents=[common.parser])

FLAGS = parser.parse_args()


def resize_and_pad(image, output_size=513, value=0):
    "Preserving aspect ratio resize, then pad to desired size."""
    height, width, _ = image.shape
    if height > width and height > output_size:
        width = int(width * output_size / height)
        height = output_size
    if width >= height and width > output_size:
        height = int(height * output_size / width)
        width = output_size
    image_ = cv2.resize(image, (width, height))
    if height < output_size or width < output_size:
        image_padded = value * np.ones((output_size, output_size, 3))
        h_start = (output_size - height) // 2
        w_start = (output_size - width) // 2
        image_padded[h_start:h_start+height, w_start:w_start+width] = image_
        image_ = image_padded.astype(np.uint8)
    return image_


def unpad_and_resize(gray, original_size):
    """Unpad."""
    height, width = gray.shape
    output_size = height
    ori_height, ori_width = original_size
    new_height, new_width = original_size
    if ori_height > ori_width and ori_height > output_size:
        new_width = int(ori_width * output_size / ori_height)
        new_height = output_size
    if ori_width >= height and ori_width > output_size:
        new_height = int(ori_height * output_size / ori_width)
        new_width = output_size
    h_start = (height - new_height) // 2
    w_start = (width - new_width) // 2
    gray_unpadded = gray[h_start:h_start+new_height, w_start:w_start+new_width]
    return cv2.resize(gray_unpadded, (ori_width, ori_height),
                      interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    image_path = FLAGS.image_path
    checkpoint_path = FLAGS.checkpoint_path
    colormap_type = FLAGS.colormap_type
    val_output_stride = FLAGS.val_output_stride
    val_atrous_rates = FLAGS.val_atrous_rates
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(image_path):
        raise ValueError('`image_path` does not exist.')
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (513, 513))
    #image_rgb = resize_and_pad(image_rgb, output_size=513)
    image_center = (2.0 / 255) * image_rgb - 1.0
    image_center = image_center.astype(np.float32)
    images = np.expand_dims(image_center, axis=0)
    images_pth = np.expand_dims(np.transpose(image_center, axes=(2, 0, 1)),
                               axis=0)
    images_pth = torch.from_numpy(images_pth).to(device)
    
    deeplab = model.deeplab(num_classes=21, pretrained=True,
                            atrous_rates=val_atrous_rates,
                            output_stride=val_output_stride,
                            checkpoint_path=checkpoint_path).to(device)
    deeplab.eval()
    with torch.no_grad():
        logits = deeplab(images_pth)
        logits = torch.nn.functional.softmax(logits, dim=1)
        #logits = model.resize_bilinear(logits, original_size)
        labels = torch.argmax(logits, dim=1)
        label_np = np.squeeze(labels.cpu().numpy()).astype(np.uint8)
        label_np = unpad_and_resize(label_np, original_size)
        
        output_path = image_path.replace('.jpg', '_seg')
        file_name = output_path.split('/')[-1]
        save_dir = output_path.replace(file_name, '')
        save_annotation.save_annotation(label_np, save_dir, file_name,
                                        add_colormap=True, 
                                        colormap_type=colormap_type)
        print('The result is saved to: ', save_dir)
        
        