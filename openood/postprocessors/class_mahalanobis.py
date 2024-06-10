from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
from .simple_mahalanobis2 import update_statistics


# class ClassMahalanobis(BasePostprocessor):
#     def __init__(self, config):
#         self.config = config
#         self.num_classes = num_classes_dict[self.config.dataset.name]
#         self.setup_flag = False
#         self.means = None
#         self.precision = None

#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         if not self.setup_flag:
#             # estimate mean and variance from training set
#             print('\n Estimating mean and variance from training set...')
#             total, total_corrects = 0, 0
#             with torch.no_grad():
#                 for batch in tqdm(id_loader_dict['train'],
#                                   desc='Setup: ',
#                                   position=0,
#                                   leave=True):
#                     data, labels = batch['data'].cuda(), batch['label'].cuda()
#                     logits, features = net(data, return_feature=True)
#                     cor = labels.eq(logits.argmax(1))
#                     total += len(labels)
#                     total_corrects += cor.sum()
#                     if self.means is None:
#                         self.means = [torch.zeros(features.shape[-1], device=features.device) for _ in labels.shape[1]]
#                         covs = [torch.zeros((features.shape[-1], features.shape[-1]), device=features.device) for _ in labels.shape[1]]
#                         counts = [0 for _ in labels.shape[1]]
#                         waiting = [None for _ in labels.shape[1]]
                    
#                     self.mean, cov, corrects = update_statistics(self.mean, cov, corrects, features[cor])

#             # sanity check on train acc
#             train_acc = corrects / total
#             print(f' Train acc: {train_acc:.2%}')

#             self.precision = [torch.linalg.inv(cov+torch.eye(cov.shape[0], device=cov.device)*cov.trace()/cov.shape[0]/100) for cov in covs]
#             self.setup_flag = True
#         else:
#             pass

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         logits, features = net(data, return_feature=True)
#         pred = logits.argmax(1)

#         tensor = features - self.mean.view(1, -1)
#         conf = -torch.matmul(
#                 torch.matmul(tensor, self.precision), tensor.t()).diag()

#         return pred, conf

class ClassMahalanobis(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.class_mean = []
        self.precision = []

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
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
                    corrects += cor.sum()
                    all_feats.append(features[cor].cpu())
                    all_labels.append(labels[cor].cpu())

                all_feats = torch.cat(all_feats)
                all_labels = torch.cat(all_labels)
                # sanity check on train acc
                train_acc = corrects / total
                print(f' Train acc: {train_acc:.2%}')

                # compute class-conditional statistics
                self.class_mean = []
                centered_data = []
                for c in range(self.num_classes):
                    class_samples = all_feats[all_labels.eq(c)].data
                    self.class_mean.append(class_samples.mean(0))
                    centered_data.append(class_samples -
                                        self.class_mean[c].view(1, -1))

                self.class_mean = torch.stack(
                    self.class_mean).cuda()  # shape [#classes, feature dim]
                del all_feats
                del all_labels

                for data in centered_data:
                    cov = torch.mm(data.t(), data) / (data.shape[0] - 1)
                    self.precision.append(torch.linalg.inv(cov+torch.eye(cov.shape[0], device=cov.device)*cov.trace()/cov.shape[0]/100))
                self.precision = torch.stack(self.precision).cuda()
                self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)
        demeaned = features - self.class_mean[pred]
        conf = -torch.sum((demeaned.unsqueeze(1) @ self.precision[pred]).squeeze(1) * demeaned, dim=1)

        return pred, conf

