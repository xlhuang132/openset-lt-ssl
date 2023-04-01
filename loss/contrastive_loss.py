import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  
# def cosine_similarity(x1, x2, eps=1e-12):
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = x2.norm(p=2, dim=1, keepdim=True)
#     return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def cosine_pairwise(x):
    x = x.permute((1, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-1)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise

def pairwise_similarity(outputs_1, outputs_2,temperature=0.5):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    '''  
    # outputs=torch.cat((outputs_1,outputs_2),dim=0)
    # B   = outputs.shape[0]
    # outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    # similarity_matrix = (1./temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))
    # return similarity_matrix
    outputs=torch.cat((outputs_1,outputs_2),dim=0)
    similarity_matrix = (1./temperature) * torch.mm(outputs,outputs.transpose(0,1))
    return similarity_matrix

 

def masked_NT_xent(similarity_matrix,mask):
    '''
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    ''' 

    N2  = len(similarity_matrix)
    N   = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (mask - torch.eye(N2,N2).cuda()).cuda()

    NT_xent_loss        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    NT_xent_loss_total  = (1./float(N2)) * torch.sum(torch.diag(NT_xent_loss[0:N,N:]) + torch.diag(NT_xent_loss[N:,0:N]))

    return NT_xent_loss_total

def weighted_masked_NT_xent(similarity_matrix,weight,mask):
    '''
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    ''' 

    N2  = len(similarity_matrix)
    N   = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(weight*similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (mask - torch.eye(N2,N2).cuda()).cuda()

    NT_xent_loss        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    NT_xent_loss_total  = (1./float(N2)) * torch.sum(torch.diag(NT_xent_loss[0:N,N:]) + torch.diag(NT_xent_loss[N:,0:N]))

    return NT_xent_loss_total


def NT_xent(similarity_matrix):
    '''
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    ''' 

    N2  = len(similarity_matrix)
    N   = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2,N2)).cuda()

    NT_xent_loss        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    NT_xent_loss_total  = (1./float(N2)) * torch.sum(torch.diag(NT_xent_loss[0:N,N:]) + torch.diag(NT_xent_loss[N:,0:N]))

    return NT_xent_loss_total

def pairwise_partial_similarity(outputs_1, outputs_2,temperature=0.5):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    '''  
    outputs=torch.cat((outputs_1,outputs_2),dim=0)
    B   = outputs.shape[0]
    cover=get_random_cover(outputs.shape[0],outputs.shape[1])
    outputs= outputs*(cover)
    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    similarity_matrix = (1./temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))
    return similarity_matrix

def get_random_cover(bs,d):
    rcover= np.random.randint(0,2,d)
    cover=torch.tensor(rcover).cuda() 
    # cover= cover.repeat(bs,1).cuda()
    return cover

def SCL(similarity_matrix,mask):
    '''
        Compute class-wise contrastive loss
        input: pairwise-similarity matrix
        return: SCL loss
    ''' 

    N2  = len(similarity_matrix) 

    # Removing diagonal 
    similarity_matrix_exp = torch.exp(similarity_matrix)
    mask=mask.repeat(2, 2)
    similarity_matrix_exp = similarity_matrix_exp * (mask - torch.eye(N2,N2).cuda())

    scl        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    scl_total  = (1./float(N2)) * torch.sum(torch.diag(scl))

    return scl_total

class DebiasSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, biases=None, mask=None):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        if mask is None:
            assert biases is not None
            biases = biases.contiguous().view(-1, 1)
            label_mask = torch.eq(labels, labels.T)
            bias_mask = torch.ne(biases, biases.T)
            mask = label_mask & bias_mask
            mask = mask.float().cuda()
        else:
            # label_mask 同类的mask mask:cosine相似度
            label_mask = torch.eq(labels, labels.T).float().cuda()
            mask = label_mask * mask # 

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #torch.Size([20480])= torch.Size([320, 64])
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases 1-torch.eye
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask # mask self-self

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

        # loss
        loss = -mean_log_prob_pos #torch.Size([384])
        loss = loss.view(anchor_count, batch_size) #torch.Size([2, 128])
        # loss = loss.mean()
        return loss

class UnsupBiasContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.con_loss = DebiasSupConLoss(temperature=temperature)
        print(f'UnsupBiasContrastiveLoss - T: {self.temperature}')
 

    def forward(self, cont_features, cont_labels, cont_bias_feats):
        # cont_bias_feats = F.normalize(cont_bias_feats, dim=1)
        mask = 1 - cosine_similarity(cont_bias_feats.cpu().numpy())
        mask = torch.from_numpy(mask).cuda()
        # mask : 相似度矩阵
        con_loss = self.con_loss(cont_features, cont_labels, mask=mask)
        return con_loss
    