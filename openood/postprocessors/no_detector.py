from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class NoDetector(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    logits, features = net(data, return_feature=True)
            #         all_feats.append(features.cpu())
            #         all_labels.append(deepcopy(labels))
            #         all_preds.append(logits.argmax(1).cpu())

            # all_feats = torch.cat(all_feats)
            # all_labels = torch.cat(all_labels)
            # all_preds = torch.cat(all_preds)
            # correct = all_preds.eq(all_labels)
            # # sanity check on train acc
            # train_acc = correct.float().mean()
            # print(f' Train acc: {train_acc:.2%}')

            # # compute class-conditional statistics
            # self.mean = all_feats[correct].mean(0)
            # centered_data = all_feats[correct] - self.mean.view(1, -1)

            # group_lasso = sklearn.covariance.EmpiricalCovariance(
            #     assume_centered=False)
            # group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            # # inverse of covariance
            # self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        conf, pred = logits.max(1)

        # tensor = features.cpu() - self.mean.view(1, -1)
        # conf = -torch.matmul(
        #         torch.matmul(tensor, self.precision), tensor.t()).diag()

        return pred, conf
