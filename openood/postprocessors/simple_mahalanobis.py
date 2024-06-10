from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

def update_statistics(old_mean, old_cov, old_count, new_data):
    new_mean = torch.mean(new_data, dim=0)
    new_count = new_data.size(0)
    mean_diff = new_mean - old_mean
    demeaned_new_data = new_data - new_mean
    new_cov = torch.mm(demeaned_new_data.t(), demeaned_new_data) / (new_count - 1)
    total_count = old_count + new_count
    new_weighted_mean = old_mean + mean_diff * (new_count / total_count)
    new_weighted_cov = (old_cov * (old_count - 1) + new_cov * (new_count - 1) +
                        torch.outer(mean_diff, mean_diff) * (old_count * new_count / total_count)) / (total_count - 1)
    return new_weighted_mean, new_weighted_cov, total_count

def extract_patches(feature_maps, kernel_size=3):
    """ Extracts sliding window patches from the feature maps. """
    batch_size, channels, height, width = feature_maps.shape
    patches = feature_maps.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    patches = patches.contiguous().view(batch_size, channels, -1, kernel_size * kernel_size)
    patches = patches.permute(0, 2, 1, 3).contiguous().view(-1, channels * kernel_size * kernel_size)
    return patches

def mahalanobis_distance(x, mean, cov_inv):
    delta = x - mean
    m_distance = torch.sqrt(torch.sum(delta @ cov_inv * delta, dim=1))
    return m_distance

class SimpleMahalanobis(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.mean = None
        self.precision = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            total, corrects = 0, 0
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label'].cuda()
                    logits, features = net(data, return_feature=True)
                    cor = labels.eq(logits.argmax(1))
                    total += len(labels)
                    if self.mean is None:
                        self.mean = torch.zeros(features.shape[-1], device=features.device)
                        cov = torch.zeros((features.shape[-1], features.shape[-1]), device=features.device)
                    self.mean, cov, corrects = update_statistics(self.mean, cov, corrects, features[cor])

            # sanity check on train acc
            train_acc = corrects / total
            print(f' Train acc: {train_acc:.2%}')

            self.precision = torch.linalg.inv(cov)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)

        delta = features - self.mean.view(1, -1)
        conf = -((delta @ self.precision) * delta).sum(1)

        return pred, conf
