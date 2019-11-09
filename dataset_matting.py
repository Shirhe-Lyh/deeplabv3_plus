# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:43:49 2019

@author: shirhe-lyh
"""

import numpy as np
import os
import PIL
import torch

from core import dataset
from core import preprocess_utils
from data import utils


class MattingDataset(dataset.SegDataset):
    """Read dataset for Matting_Human_Half datas."""
    
    def __init__(self, annotation_path, root_dir=None, transforms=None, 
                 output_height=513, output_width=513):
        """Constructor."""
        super(MattingDataset, self).__init__(
            annotation_path=annotation_path,
            root_dir=root_dir,
            transforms=transforms,
            output_height=output_height,
            output_width=output_width)
        
    def __getitem__(self, index):
        """For Matting_Human_Half datasets, just resize to desired size"""
        image_path, mask_path = self._image_mask_paths[index]
        image = PIL.Image.open(image_path)
        mask = PIL.Image.open(mask_path)
        
        # Resize
        image = image.resize((self._output_width, self._output_height), 
                             PIL.Image.ANTIALIAS)
        mask = mask.resize((self._output_width, self._output_height), 
                            PIL.Image.NEAREST)
        # Flip
        image, mask = preprocess_utils.random_flip(image, mask)
        
        image = PIL.Image.fromarray(np.array(image))
        mask = torch.LongTensor(np.array(mask))
        image_preprocessed = self._transforms(image)
        return image_preprocessed, mask
        
    def get_image_mask_paths(self, annotation_path, root_dir=None):
        """Get the paths of images and masks.
        
        Args:
            annotation_path: A file contains the paths of images and masks.
            
        Returns:
            A list [[image_path, mask_path], [image_path, mask_path], ...].
            
        Raises:
            ValueError: If annotation_file does not exist.
        """
        # Format: [[image_path, matting_path, alpha_path, mask_path], ...]
        image_matting_alpha_mask_paths = utils.provide(annotation_path)
        # Remove matting_paths, alpha_paths
        image_mask_paths = []
        for image_path, _, _, mask_path in image_matting_alpha_mask_paths:
            if root_dir is not None:
                if not image_path.startswith(root_dir):
                    image_path = os.path.join(root_dir, image_path)
                    mask_path = os.path.join(root_dir, mask_path)
                    image_path = image_path.replace('\\', '/')
                    mask_path = mask_path.replace('\\', '/')
            image_mask_paths.append([image_path, mask_path])
        return image_mask_paths
        