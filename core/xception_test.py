# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:28:42 2019

@author: shirhe-lyh
"""

import torch

import xception


if __name__ == '__main__':
    fe = xception.xception_65(global_pool=False, pretrained=False)
    fe.eval()
    
    inputs = torch.rand((1, 3, 224, 224))
    
    with torch.no_grad():
        net = fe(inputs)
        print(fe.end_points().keys())

