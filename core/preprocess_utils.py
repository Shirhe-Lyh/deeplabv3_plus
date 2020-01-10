# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:45:59 2019

@author: shirhe-lyh
"""

import numpy as np
import PIL


def random_rotate(image, label=None, max_angle=30):
    """Randomly rotate."""
    degree = np.random.randint(low=-max_angle, high=max_angle)
    image = image.rotate(degree)
    if label is not None:
        label = label.rotate(degree)
    return image, label


def get_random_scale(min_scale_factor=0.5, max_scale_factor=2.0,
                     step_size=0.25):
    """Gets a random scale value."""
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')
        
    if min_scale_factor == max_scale_factor:
        return min_scale_factor
    
    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return np.random.uniform(low=min_scale_factor, high=max_scale_factor)
    
    # When step_size !=0, we randomly select one discrete value from [min, max].
    scale_factors = np.arange(min_scale_factor, max_scale_factor, step_size)
    return np.random.choice(scale_factors)


def resize(image, label=None, scale=1.0):
    """Scales image and label."""
    if scale == 1.0:
        return image, label
    
    width, height = image.size
    width_new = int(width * scale)
    height_new = int(height * scale)
    image = image.resize((width_new, height_new), PIL.Image.ANTIALIAS)
    if label is not None:
        label = label.resize((width_new, height_new), PIL.Image.NEAREST)
    return image, label


def random_resize(image, label=None, min_scale_factor=0.5, 
                  max_scale_factor=2.0, step_size=0.25):
    """Randomly resize."""
    scale = get_random_scale(min_scale_factor, max_scale_factor, step_size)
    return resize(image, label, scale)


def pad(image, offset_height, offset_width, target_height, target_width,
        pad_value):
    """Pad the given image with the given pad_value."""
    width, height = image.size
    if height >= target_height and width >= target_width:
        return image
    
    target_height = np.max((target_height, height))
    target_width = np.max((target_width, width))
    image_pad = PIL.Image.new(image.mode, size=(target_width, target_height), 
                              color=pad_value)
    offset_height = np.min((offset_height, target_height - height))
    offset_width = np.min((offset_width, target_width - width))
    image_pad.paste(image, (offset_width, offset_height))
    return image_pad


def random_crop(image, label=None, crop_height=513, crop_width=513):
    """Crops the given image."""
    width, height = image.size
    if height < crop_height or width < crop_width:
        raise ValueError('`crop_height` must be not greater than height '
                         'and `crop_width` must be not greater than width.')
        
    offset_height = np.random.randint(low=0, high=height - crop_height + 1)
    offset_width = np.random.randint(low=0, high=width - crop_width + 1)
    box = (offset_width, offset_height, offset_width + crop_width, 
           offset_height + crop_height)
    image = image.crop(box=box)
    if label is not None:
        label = label.crop(box=box)
    return image, label


def random_flip(image, label=None, prob=0.5):
    """Flips horizontally or not."""
    chance = np.random.uniform()
    if chance > prob:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if label is not None:
            label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return image, label
