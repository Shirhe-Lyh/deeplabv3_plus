# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:50:00 2019

@author: shirhe-lyh
"""

import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torchvision as tv

import common
import model

# See details at common.py
parser = argparse.ArgumentParser(
    description='Train DeepLab v3+ model.',
    parents=[common.parser])

FLAGS = parser.parse_args()


if __name__ == '__main__':
    ckpt_path = './models/model.pth'
    test_images_dir = './data/test'
    output_dir = './data/test'
    test_images_paths = glob.glob(os.path.join(test_images_dir, '*.jpg'))
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    deeplab = model.deeplab(num_classes=FLAGS.num_classes,
                            crop_size=FLAGS.crop_size,
                            atrous_rates=FLAGS.atrous_rates,
                            output_stride=FLAGS.output_stride,
                            pretrained=False).to(device)
#    deeplab.load_state_dict(torch.load(ckpt_path))
    deeplab_pretrained_params = torch.load(ckpt_path).items()
    deeplab_state_dict = {k.replace('module.', ''): v for k, v in
                          deeplab_pretrained_params}
    deeplab.load_state_dict(deeplab_state_dict)
    
    # Transform
    channel_means = [0.5, 0.5, 0.5]
    channel_std = [0.5, 0.5, 0.5]
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=channel_means, std=channel_std)])
    
    deeplab.eval()
    with torch.no_grad():
        for image_path in test_images_paths:
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image = cv2.resize(image, tuple(FLAGS.crop_size))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_processed = transforms(image_rgb).to(device)
            images = torch.unsqueeze(image_processed, dim=0)
            
            outputs = deeplab(images)
            masks = torch.nn.functional.softmax(outputs, dim=1)
            mask_person = masks.data.cpu().numpy()[0][1]
            mask_person = 255 * mask_person
            mask_person = mask_person.astype(np.uint8)
            mask_person = cv2.resize(mask_person, (width, height))
            
            image_name = image_path.replace('\\', '/').split('/')[-1]
            output_path = os.path.join(output_dir, 
                                       image_name.replace('.jpg', '_pred.png'))
            cv2.imwrite(output_path, mask_person)
        