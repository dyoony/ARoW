'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'RecorderMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch   = total_epoch
        self.current_epoch = 0
        self.epoch_losses  = np.zeros((self.total_epoch, 3), dtype=np.float32) # [epoch, train/val/att_val]
        self.epoch_losses  = self.epoch_losses - 1

        self.epoch_accuracy= np.zeros((self.total_epoch, 3), dtype=np.float32) # [epoch, train/val/att_val]
        self.epoch_accuracy= self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc, attack_val_loss, attack_val_acc):
        assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
        self.epoch_losses  [idx, 0] = train_loss
        self.epoch_losses  [idx, 1] = val_loss
        self.epoch_losses  [idx, 2] = attack_val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.epoch_accuracy[idx, 2] = attack_val_acc
        self.current_epoch = idx + 1
        
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80  
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 20
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)
  
        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='c', linestyle='-', label='test-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)
        
        y_axis[:] = self.epoch_accuracy[:, 2]
        plt.plot(x_axis, y_axis, color='r', linestyle='-', label='attack-test-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

    
        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis*50, color='c', linestyle=':', label='test-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)
        
        y_axis[:] = self.epoch_losses[:, 2]
        plt.plot(x_axis, y_axis*50, color='r', linestyle=':', label='test-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print ('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)
    