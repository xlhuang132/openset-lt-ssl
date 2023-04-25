""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity

# class DebiasedSoftSupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(DebiasedSoftSupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, 
#                 features, 
#                 max_probs, 
#                 labels=None,  
#                 pos_mask=None, 
#                 neg_mask=None, 
#                 reduction="mean", 
#                 select_matrix=None):
#         """Compute loss for model. If both `labels` and `pos_mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             pos_mask: contrastive pos_mask of shape [bsz, bsz], pos_mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
        
#         if labels is not None:       
#             labels = labels.contiguous().view(-1, 1)
#             labels_mask= torch.eq(labels, labels.T).float().cuda()
            
#             if pos_mask is not None:
#                 pos_mask=pos_mask*labels_mask
#             else:
#                 pos_mask=labels_mask
                
#             if neg_mask is not None:
#                 neg_mask=neg_mask*(1-labels)
#             else:
#                 pos_mask=1-labels_mask
#         elif labels is None and pos_mask is None:
#             pos_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            
#         if labels is not None and pos_mask is not None:
#             # label_pos_mask 同类的pos_mask pos_mask:cosine相似度        
#             labels = labels.contiguous().view(-1, 1)
#             label_pos_mask = torch.eq(labels, labels.T).float().cuda()
#             pos_mask = label_pos_mask * pos_mask # 
#         #     raise ValueError('Cannot define both `labels` and `pos_mask`')
#         elif labels is None and pos_mask is None:
#         # if labels is None and pos_mask is None:
#             pos_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None and select_matrix is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             pos_mask = torch.eq(labels, labels.T).float().to(device)
#             max_probs = max_probs.contiguous().view(-1, 1)
#             score_pos_mask = torch.matmul(max_probs, max_probs.T)
#             # Some may find that the line 59 is different with eq(6)
#             # Acutuall the final pos_mask will set weight=0 when i=j, following Eq(8) in paper
#             # For more details, please see issue 9
#             # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
#             pos_mask = pos_mask.mul(score_pos_mask) * select_matrix

#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             pos_mask = torch.eq(labels, labels.T).float().to(device)
#             #max_probs = max_probs.reshape((batch_size,1))
#             max_probs = max_probs.contiguous().view(-1, 1)
#             score_pos_mask = torch.matmul(max_probs,max_probs.T) # pi*pj
#             pos_mask = pos_mask.mul(score_pos_mask) # pi*pj*mij
#         else:
#             pos_mask = pos_mask.float().to(device)

#         contrast_count = features.shape[1] # 对比的view数量
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 将两个view的连接来
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

#         # compute logits torch.Size([256, 256]) similarity
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#         # for numerical stability， 找出每个样本的最大的
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach() # 减去最大的
#         logits=anchor_dot_contrast
#         # tile pos_mask contrast_count:2
#         pos_mask = pos_mask.repeat(anchor_count, contrast_count) # torch.Size([256, 256])
#         # pos_mask-out self-contrast cases
#         logits_pos_mask = torch.scatter(
#             torch.ones_like(pos_mask), # torch.Size([256, 256])
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         ) # torch.Size([256, 256]): 1-torch.eye(2N)
#         pos_mask = pos_mask * logits_pos_mask # pos_mask self
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_pos_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1) 
#         sum_pos_mask = pos_mask.sum(1)
#         sum_pos_mask[sum_pos_mask == 0] = 1
#         mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_pos_mask
        
#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos #torch.Size([256])
#         loss = loss.view(anchor_count, batch_size) #torch.Size([2, 128])

#         if reduction == "mean":
#             loss = loss.mean()

#         return loss


class DebiasSoftConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    # def forward(self, features,
    #             max_probs,
    #             labels,
    #             biased_cos_sim=None,
    #             contrastive_with_hp=False,
    #             contrastive_with_hn=False,
    #             select_matrix=None,
    #             reduction='mean'):
            
    #     batch_size = features.shape[0]
    #     labels = labels.contiguous().view(-1, 1)
    #     if labels.shape[0] != batch_size:
    #         raise ValueError('Num of labels does not match num of features')
        
    #     contrast_count = features.shape[1]
    #     contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #torch.Size([20480])= torch.Size([320, 64])
    #     if self.contrast_mode == 'one':
    #         anchor_feature = features[:, 0]
    #         anchor_count = 1
    #     elif self.contrast_mode == 'all':
    #         anchor_feature = contrast_feature
    #         anchor_count = contrast_count
    #     else:
    #         raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    #     # compute logits sim
    #     anchor_dot_contrast = torch.div(
    #         torch.matmul(anchor_feature, contrast_feature.T),
    #         self.temperature)
    #     # for numerical stability
    #     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - logits_max.detach()

    #     # tile mask 
    #     label_mask = torch.eq(labels, labels.T).float().cuda() 
        
    #     max_probs = max_probs.contiguous().view(-1, 1)
    #     score_mask = torch.matmul(max_probs, max_probs.T)
    #     label_mask = label_mask.mul(score_mask) 
        
    #     label_mask = label_mask.repeat(anchor_count, contrast_count)
         
        
        
    #     if contrastive_with_hp:
    #         pos_weight = 1 - biased_cos_sim 
    #     else:
    #         pos_weight=torch.ones_like(z).cuda()
    #     if select_matrix is not None:
    #         pos_weight*=select_matrix
    #     pos_weight=pos_weight.repeat(anchor_count, contrast_count)
    #     pos_mask=(1+pos_weight)*label_mask
        
    #     if contrastive_with_hn:
    #         neg_weight = biased_cos_sim 
    #     else:
    #         neg_weight=torch.ones_like(pos_weight).cuda() 
    #     neg_weight=neg_weight.repeat(anchor_count, contrast_count)
    #     neg_mask=(1+neg_weight)*(1-label_mask)
        
    #     # mask-out self-contrast cases m_ii
    #     logits_mask = torch.scatter(
    #         torch.ones_like(label_mask),
    #         1,
    #         torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
    #         0
    #     )
    #     pos_mask = pos_mask * logits_mask # mask self-self

    #     # compute log_prob
    #     exp_logits = torch.exp(logits) * logits_mask
    #     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    #     # compute mean of log-likelihood over positive
    #     sum_mask = pos_mask.sum(dim=1)+neg_mask.sum(dim=1)
    #     sum_mask[sum_mask == 0] = 1
    #     mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask
        
    #     # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

    #     # loss
    #     loss = -mean_log_prob_pos #torch.Size([384]) 
    #     loss = loss.view(anchor_count, batch_size)
    #     if reduction=='mean':
    #         return loss.mean()
    #     else:
    #         return loss

    def forward(self, features, 
            # max_probs=None, 
            labels=None, 
            # mask=None,  
            pos_mask=None,
            neg_mask=None,
            reduction="mean", 
            select_matrix=None):
        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().cuda() 
            if neg_mask is None:
                neg_mask=1-mask
            if pos_mask is None:
                pos_mask=mask
        else: 
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        # if max_probs is None and labels is None:
        #     mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        # elif labels is None and max_probs is None:  
        #     mask=mask
        # # mask: 偏置分数 x
        # # 仅仅使用 pi*pj
        # else:
        #     labels = labels.contiguous().view(-1, 1)
        #     if labels.shape[0] != batch_size:
        #         raise ValueError('Num of labels does not match num of features')
        #     label_mask = torch.eq(labels, labels.T).float().cuda() 
            
            # pi*pj 
            # if max_probs is not None and labels is not None: 
            #     label_mask = torch.eq(labels, labels.T).float().cuda()
            #     max_probs = max_probs.contiguous().view(-1, 1)
            #     score_mask = torch.matmul(max_probs, max_probs.T)
            #     label_mask = label_mask.float().cuda()
            #     label_mask = label_mask.mul(score_mask) 
            #     # pi*pj*bias_score
            #     if mask is not None:
            #         mask=label_mask*mask
            #     else:
            #         mask=label_mask
            # # bias_score + labels
            # elif max_probs is None and mask is not None and labels is not None:
            #     # label_mask 同类的mask mask:cosine相似度
            #     label_mask = torch.eq(labels, labels.T).float().cuda() 
            #     # mask = label_mask * mask   # version 13
            # # labels 
            # else:
            #     label_mask = torch.eq(labels, labels.T).float().cuda()
            #     mask = label_mask

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
        pos_mask= pos_mask.repeat(anchor_count, contrast_count)
        neg_mask= neg_mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases 1-torch.eye
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        
        mask = mask * logits_mask # mask self-self        
        pos_mask=pos_mask*logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        # loss version 13
        log_prob = (pos_mask+neg_mask)*logits - torch.log(((pos_mask+neg_mask)*exp_logits).sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        
        sum_mask = (pos_mask+neg_mask).sum(1)
        sum_mask[sum_mask == 0] = 1
        if select_matrix is not None:
            select_matrix = select_matrix.repeat(anchor_count, contrast_count)
            mean_log_prob_pos = (mask*select_matrix * log_prob).sum(1) / sum_mask
        else:
            mean_log_prob_pos = ()
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask # version 13
            # mean_log_prob_pos = ((pos_mask+neg_mask) * log_prob).sum(1) / sum_mask # version 9\10\11\12
        
        # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

        # loss
        loss = -mean_log_prob_pos #torch.Size([384])
        loss = loss.view(anchor_count, batch_size) #torch.Size([2, 128])
        # loss = loss.mean()
        if reduction == "mean":
            loss = loss.mean()

        return loss