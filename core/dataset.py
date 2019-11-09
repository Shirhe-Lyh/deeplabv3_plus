# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:43:49 2019

@author: shirhe-lyh
"""
import numpy as np
import os
import PIL
import torch
import torchvision as tv

from abc import ABCMeta
from abc import abstractmethod

from core import preprocess_utils


class SegDataset(torch.utils.data.Dataset):
    """Read dataset for Matting."""
    __metaclass__ = ABCMeta
    
    def __init__(self, annotation_path, root_dir=None, transforms=None, 
                 output_height=513, output_width=513, 
                 min_scale_factor=0.5, max_scale_factor=2.0,
                 scale_factor_step_size=0.25):
        """Constructor.
        
        Args:
            annotation_path: A file contains the paths of images and masks.
            transforms: torchvision.tansforms.Compose([...])
            output_height: The height of the preprocessed images.
            output_width: The width of the preprocessed images.
            min_scael_factor: Minimum scale factor value.
            max_scale_factor: Maximum scale factor value.
            scale_factor_step_size: The step size from min scale factor to
                max scale factor. The input is randomly scaled based on the 
                value of 
                (min_scale_factor, max_scale_factor, scale_factor_step_size).
        """
        self._transforms = transforms
        self._output_height = output_height
        self._output_width = output_width
        self._min_scale_factor = min_scale_factor
        self._max_scale_factor = max_scale_factor
        self._scale_factor_step_size = scale_factor_step_size
        
        # Transform
        if transforms is None:
            channel_means = [0.5, 0.5, 0.5]
            channel_std = [0.5, 0.5, 0.5]
            self._transforms = tv.transforms.Compose([
                tv.transforms.ColorJitter(brightness=32/255., contrast=0.5, 
                                      saturation=0.5, hue=0.2),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=channel_means, std=channel_std)])
        
        # A list [[image_path, mask_path], [image_path, mask_path], ...]
        self._image_mask_paths = self.get_image_mask_paths(annotation_path,
                                                           root_dir=root_dir)
        self.remove_invalid_data()
        
    def __getitem__(self, index):
        image_path, mask_path = self._image_mask_paths[index]
        image = PIL.Image.open(image_path)
        mask = PIL.Image.open(mask_path)
        
        # Data augmentation
        # Rotate
#        image, mask = preprocess_utils.random_rotate(image, mask)
        # Resize
        image, mask = preprocess_utils.random_resize(
            image, mask, self._min_scale_factor, self._max_scale_factor,
            self._scale_factor_step_size)
        # Pad
        image = preprocess_utils.pad(image, 0, 0, self._output_height,
                                     self._output_width, 
                                     pad_value=(127, 127, 127))
        mask = preprocess_utils.pad(mask, 0, 0, self._output_height,
                                    self._output_width, pad_value=0)
        # Crop
        image, mask = preprocess_utils.random_crop(image, mask,
                                                   self._output_height,
                                                   self._output_width)
        # Flip
        image, mask = preprocess_utils.random_flip(image, mask)
        
        image = PIL.Image.fromarray(np.array(image))
        mask = torch.LongTensor(np.array(mask))
        image_preprocessed = self._transforms(image)
        return image_preprocessed, mask
    
    def __len__(self):
        return len(self._image_mask_paths)
            
    @abstractmethod
    def get_image_mask_paths(self, annotation_path, **kwargs):
        """Get the paths of images and masks.
        
        Args:
            annotation_path: A file contains the paths of images and masks.
            kwargs: Additional key word arguments.
            
        Returns:
            A list [[image_path, mask_path], [image_path, mask_path], ...].
        """
        pass
    
    def remove_invalid_data(self):
        valid_data = []
        for image_path, mask_path in self._image_mask_paths:
            if not os.path.exists(image_path):
                continue
            if not os.path.exists(mask_path):
                continue
            valid_data.append([image_path, mask_path])
        self._image_mask_paths = valid_data
        
        if not self._image_mask_paths:
            raise ValueError('Error: No images found.')