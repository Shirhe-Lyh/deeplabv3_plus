# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:19:14 2019

@author: shirhe-lyh
"""

import argparse
import json
import numpy as np
import os
import torch

from torch.utils.tensorboard import SummaryWriter

import common
import dataset_matting
import model

# See details at common.py
parser = argparse.ArgumentParser(
    description='Train DeepLab v3+ model.',
    parents=[common.parser])

FLAGS = parser.parse_args()


def config_learning_rate(optimizer, step):
    lr = FLAGS.learning_rate * FLAGS.lr_decay ** (step / FLAGS.decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    annotation_path = FLAGS.annotation_path
    crop_size = FLAGS.crop_size
    gpu_indices = FLAGS.gpu_indices
    learning_rate = FLAGS.learning_rate
    model_dir = FLAGS.model_dir
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size * len(gpu_indices)
    
    gpu_ids_str = ','.join([str(index) for index in gpu_indices])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    
    # Device configuration
    cuda_ = 'cuda:{}'.format(gpu_indices[0])
    device = torch.device(cuda_ if torch.cuda.is_available() else 'cpu')
    
    matting_dataset = dataset_matting.MattingDataset(
        annotation_path=annotation_path,
        root_dir=FLAGS.root_dir,
        output_height=crop_size[0],
        output_width=crop_size[1])
    train_loader = torch.utils.data.DataLoader(matting_dataset, 
                                               batch_size=batch_size,
                                               shuffle=True, 
                                               num_workers=32,
                                               drop_last=True)
    
    deeplab = model.deeplab(num_classes=FLAGS.num_classes,
                            crop_size=crop_size,
                            atrous_rates=FLAGS.atrous_rates,
                            output_stride=FLAGS.output_stride,
                            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                            pretrained=True,
                            pretained_num_classes=FLAGS.pretained_num_classes,
                            checkpoint_path=FLAGS.checkpoint_path).to(device)
    
    # Load last trained parameters
    start_epoch, start_step = 0, 0
    last_checkpoint_path = None
    json_path = os.path.join(model_dir, 'checkpoint.json')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        if os.path.exists(json_path):
            with open(json_path, 'r') as reader:
                ckpt_dict = json.load(reader)
                start_epoch = ckpt_dict.get('epoch', 0) + 1
                start_step = ckpt_dict.get('step', 0) + 1
            ckpt_name = 'model-{}-{}.pth'.format(start_epoch, start_step)
            if os.path.exists(os.path.join(model_dir, ckpt_name)):
                last_checkpoint_path = os.path.join(model_dir, ckpt_name)
            if os.path.exists(os.path.join(model_dir, 'model.pth')):
                last_checkpoint_path = os.path.join(model_dir, 'model.pth')
    if last_checkpoint_path and os.path.exists(last_checkpoint_path):
        # deeplab.load_state_dict(torch.load(last_checkpoint_path))
        pretrained_params = torch.load(last_checkpoint_path).items()
        state_dict = {k.replace('module.', ''): v for k, v in
                      pretrained_params}
        deeplab.load_state_dict(state_dict)
        print('Load pretrained parameters, Done')
    
    # Multi GPUs
    deeplab = torch.nn.DataParallel(deeplab, device_ids=gpu_indices)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(deeplab.parameters(), lr=learning_rate)
    
    # Tensorboard
    log_dir = os.path.join(model_dir, 'logs')
    log = SummaryWriter(log_dir=log_dir)
    
    total_step = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = deeplab(images)
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            step = i + epoch * total_step
            if (i+1) % 50 == 0:
#                print('Epoch {}/{}, Step: {}/{}, Loss: {:.4f}'.format(
#                    epoch+1, num_epochs, i+1, total_step, loss.item()))
                
                # Log scalar values
                log.add_scalar('loss', loss.item(), step+1)
                
                # Log training images
                gt_masks = np.expand_dims(masks.cpu().numpy()[:2], axis=1)
                pred_masks = torch.nn.functional.softmax(
                    outputs, dim=1).data.cpu().numpy()[:2][:, 1:, :, :]
                pred_masks = np.sum(pred_masks, axis=1, keepdims=True)
                log.add_images('gt_masks', gt_masks, step+1, dataformats='NCHW')
                log.add_images('pred_masks', pred_masks, step+1,
                               dataformats='NCHW')
                
            # Decay learning rate
            if step % FLAGS.decay_step == 0:
                lr = config_learning_rate(optimizer, step)
                log.add_scalar('learning_rate', lr, step+1)
                
        # Save model
#            print('Save Model: Epoch {}/{}, Step: {}/{}'.format(
#                epoch+1, num_epochs, i+1, total_step))
        model_name = 'model-{}-{}.pth'.format(epoch+1, i+1)
        model_path = os.path.join(model_dir, model_name)
        torch.save(deeplab.state_dict(), model_path)
        ckpt_dict = {'epoch': epoch, 'step': i, 'global_step': step}
        with open (json_path, 'w') as writer:
            json.dump(ckpt_dict, writer)
    log.close()

    # Final save          
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(deeplab.state_dict(), model_path)
    ckpt_dict = {'epoch': num_epochs-1, 'step': total_step-1, 
                 'global_step': num_epochs * total_step - 1}
    with open (json_path, 'w') as writer:
        json.dump(ckpt_dict, writer)
    
    
if __name__ == '__main__':
    train()