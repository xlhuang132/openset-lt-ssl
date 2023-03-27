import torch
import torch.nn as nn
from torch import Tensor 
from typing import Tuple, Union
from collections import defaultdict

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def accuracy(output, label):
#     cnt = label.shape[0]
#     true_count = (output == label).sum() # TP/TOTAL
#     now_accuracy = true_count / cnt
#     return now_accuracy, cnt 
    
