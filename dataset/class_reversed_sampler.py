import random

import numpy as np
from torch.utils.data import Sampler
import torch


class ClassReversedSampler(Sampler):

    def __init__(self, data_source, total_samples=None, shuffle=True):
        try:
            labels = data_source.dataset["labels"] 
        except:
            labels = data_source.targets 

        num_classes = len(np.unique(labels))
        label_to_count = [0] * num_classes

        for idx, label in enumerate(labels):
            label_to_count[label] += 1

        
        n_max=max(label_to_count)
        per_cls_weights = n_max/np.array(label_to_count)
        per_cls_weights = per_cls_weights/per_cls_weights.sum()

        weights = torch.DoubleTensor([per_cls_weights[label] for label in labels])

        # total train epochs
        num_epochs = int(total_samples / len(labels)) + 1
        total_inds = []
        for epoch in range(num_epochs): # 提前把所有epoch的样本index计算好
            inds_list = torch.multinomial(weights, len(labels), replacement=True).tolist()
            if shuffle:
                random.shuffle(inds_list)
            total_inds.extend(inds_list)
        total_inds = total_inds[:total_samples]

        self.per_cls_prob = per_cls_weights / np.sum(per_cls_weights)

        self._indices = total_inds 

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

class DistClassReversedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, data_source, total_samples=None, shuffle=True,num_replicas=0,
            rank=0):
        super().__init__(data_source,num_replicas,rank)
        try:
            labels = data_source.dataset["labels"] 
        except:
            labels = data_source.targets 

        num_classes = len(np.unique(labels))
        label_to_count = [0] * num_classes

        for idx, label in enumerate(labels):
            label_to_count[label] += 1

        
        n_max=max(label_to_count)
        per_cls_weights = n_max/np.array(label_to_count)
        per_cls_weights = per_cls_weights/per_cls_weights.sum()

        weights = torch.DoubleTensor([per_cls_weights[label] for label in labels])

        # total train epochs
        num_epochs = int(total_samples / len(labels)) + 1
        total_inds = []
        for epoch in range(num_epochs): # 提前把所有epoch的样本index计算好
            inds_list = torch.multinomial(weights, len(labels), replacement=True).tolist()
            if shuffle:
                random.shuffle(inds_list)
            total_inds.extend(inds_list)
        total_inds = total_inds[:total_samples]

        self.per_cls_prob = per_cls_weights / np.sum(per_cls_weights)

        self._indices = total_inds 

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

 