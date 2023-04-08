
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
import copy
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
from utils.build_optimizer import get_optimizer
import models 
import time 
import torch.optim as optim 
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer
import models
from loss.contrastive_loss import *
from loss.soft_supconloss import SoftSupConLoss
from loss.gce import GeneralizedCELoss
from utils import FusionMatrix
from models.projector import  Projector 
from dataset.base import BaseNumpyDataset
from dataset.build_dataloader import _build_loader
from utils.misc import AverageMeter  
from loss.debiased_soft_contra_loss import *


class DCSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        # 需要先捕获偏差特征  
        
        self.lambda_d=cfg.ALGORITHM.DCSSL.LAMBDA_D 
        self.m=cfg.ALGORITHM.DCSSL.M
        self.debiased_contra_temperture=cfg.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE
        self.biased_fusion_matrix=FusionMatrix(self.num_classes)
        
        self.loss_contrast= DebiasSoftConLoss(temperature=self.debiased_contra_temperture)
        # SoftSupConLoss(temperature=self.debiased_contra_temperture) #DebiasSoftConLoss(temperature=self.debiased_contra_temperture)
        # self.logger.info("SimCLR contrastive loss")
        self.contrast_with_thresh=cfg.ALGORITHM.DCSSL.CONTRAST_THRESH
        self.contrast_with_hp=cfg.ALGORITHM.DCSSL.CONTRAST_WITH_HP
        self.contrast_wwith_hn=cfg.ALGORITHM.DCSSL.CONTRAST_WITH_HN
        self.losses_d_ctr=AverageMeter()
        self.losses_bx = AverageMeter()
        self.losses_bu = AverageMeter() 
        # self.biased_contrastive_loss=UnsupBiasContrastiveLoss()  
        self.biased_model=self.build_model(cfg).cuda()
        self.biased_optimizer=self.build_optimizer(cfg, self.biased_model) 
        # self.ema_model=self.ema_model.cuda()
        self.mixup_alpha=0.5
        self.sharpen_temp=0.5
        self.gce_loss=GeneralizedCELoss()
        self.fp_k=cfg.ALGORITHM.DCSSL.FP_K
        self.loss_version=self.cfg.ALGORITHM.DCSSL.LOSS_VERSION
        self.logger.info('contrastive loss version {}'.format(self.loss_version))
        
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_d_ctr=AverageMeter()
        self.losses_bx = AverageMeter()
        self.losses_bu = AverageMeter() 
        
    def train_step(self,pretraining=False): 
        try:        
            data_x = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            data_x = self.labeled_train_iter.next() 
        try:       
            data_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data_u = self.unlabeled_train_iter.next()  
        # if self.epoch<100:
        # self.train_biased_step(data_x,data_u) 
        return_data=self.train_debiased_step(data_x,data_u)
        return return_data
    
    def train_biased_step(self,data_x,data_u):
        self.biased_model.train()
        loss =0       
        # dataset
        inputs_x=data_x[0]
        targets_x=data_x[1]        
        inputs_u=data_u[0][0]
        inputs=torch.cat([inputs_x,inputs_u],dim=0)
        inputs=inputs.cuda()
        targets_x=targets_x.long().cuda() 
        # logits
        logits=self.biased_model(inputs) 
        logits_x=logits[:inputs_x.size(0)] 
        # inputs_x=inputs_x.cuda()
        # logits_x=self.biased_model(inputs_x)     
        
        # 1. ce loss
        # loss_cls=self.l_criterion(logits_x,targets_x)
        loss_cls = self.gce_loss(logits_x, targets_x).mean()
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        # # 2. cons loss
        u_weak=logits[inputs_x.size(0):]
        with torch.no_grad(): 
            p = u_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(self.conf_thres).float()  
        loss_cons = self.ul_criterion(
            u_weak, pred_class, weight=loss_weight, avg_factor=u_weak.size(0)
        )
        loss=loss_cls+loss_cons
        # loss=loss_cls
        self.biased_optimizer.zero_grad()
        loss.backward()
        self.biased_optimizer.step()  
        self.losses_bx.update(loss_cls.item(), inputs_x.size(0))
        self.losses_bu.update(loss_cons.item(), inputs_u.size(0))  
        
        if self.iter % self.cfg.SHOW_STEP==0:
        # if self.iter % 2==0:
            self.logger.info('== Biased Epoch:{} Step:[{}|{}]  Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,self.losses_bx.avg,self.losses_bu.avg))
            
            # self.logger.info('== Biased Epoch:{} Step:[{}|{}]  Avg_Loss_x:{:>5.4f} =='\
                # .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        # self.train_per_step,self.losses_bx.avg))
        return now_result.cpu().numpy(), targets_x.cpu().numpy()   
       
    def train_debiased_step(self,data_x,data_u):
        self.model.train()
        loss =0 
        
        inputs_x=data_x[0] 
        targets_x=data_x[1]
        
        # DU   
        inputs_u_w=data_u[0][0]
        inputs_u_s=data_u[0][1]
        inputs_u_s1=data_u[0][2]
        
        inputs = torch.cat(
                [inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1],
                dim=0).cuda()
        
        targets_x=targets_x.long().cuda()
        
        encoding = self.model(inputs,return_encoding=True)
        features=self.model(encoding,return_projected_feature=True)
        logits=self.model(encoding,classifier=True)
        batch_size=inputs_x.size(0)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
        _, f_u_s1, f_u_s2 = features[batch_size:].chunk(3)
        del logits
        del features
        del _ 
        
        # 1. ce loss          
        loss_cls = self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
         
        # 2. cons loss 
        # filter out low confidence pseudo label by self.cfg.threshold 
        with torch.no_grad(): 
            probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
            # pseudo label and scores for u_w
            max_probs, pred_class = torch.max(probs_u_w, dim=-1)             
        loss_weight = max_probs.ge(self.conf_thres).float()  
        loss_cons = self.ul_criterion(
            logits_u_s, pred_class, weight=loss_weight, avg_factor=logits_u_s.size(0)
        )
        
        # 3. ctr loss
        # === biased_contrastive_loss
        # for supervised contrastive
        
        labels = pred_class 
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1) #torch.Size([128, 2, 64])
        # In case of early training stage, pseudo labels have low scores 
        
        # 0. 实例对比学习
        if self.loss_version==0:
            loss_d_ctr = self.loss_contrast(features)   
        elif self.loss_version==1:
        # # 1. 高置信度正样本+pi*pj ccssl
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            loss_d_ctr = self.loss_contrast(features, # projected_feature
                                            max_probs=max_probs, # confidence
                                            labels=labels, # pred_class
                                            reduction=None) #torch.Size([2, 128])
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        elif self.loss_version==2:
        # 2. 选择高置信度的样本对作为正对 + pi*pj
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            with torch.no_grad():
                select_matrix = self.contrast_left_out_p(max_probs)
            loss_d_ctr = self.loss_contrast(features,
                                            max_probs=max_probs,
                                            labels=labels,
                                            reduction=None,
                                            select_matrix=select_matrix)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        elif self.loss_version==3:
        # 3. pos weight + neg weight 
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            with torch.no_grad():
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                pos_mask = torch.from_numpy(cos_sim).cuda()
                mask= torch.eq(labels, labels.T).float().cuda()
                mask=pos_mask*mask+(1-pos_mask)*(1-mask)
            loss_d_ctr = self.loss_contrast(features,
                                            # max_probs=max_probs,
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        # 4. 样本软加权 qi*去偏对比损失
        elif self.loss_version==4:
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            contrast_mask*=max_probs
            with torch.no_grad():
                biased_feat=self.biased_model(inputs[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)],return_encoding=True)
                cos_sim= 1 - cosine_similarity(biased_feat.detach().cpu().numpy())
                mask = torch.from_numpy(cos_sim).cuda() 
            loss_d_ctr = self.loss_contrast(features,
                                            # max_probs=max_probs,
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        elif self.loss_version==5:
        # 5. 直接使用高维特征计算相似度
            contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
            contrast_mask*=max_probs
            with torch.no_grad(): 
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                mask = torch.from_numpy(cos_sim).cuda()
            loss_d_ctr = self.loss_contrast(features, 
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        elif self.loss_version==6:
        # # 6. 直接使用高维特征计算相似度 +  使用所有样本加权qi
            contrast_mask=max_probs
            with torch.no_grad(): 
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                mask = torch.from_numpy(cos_sim).cuda()
            loss_d_ctr = self.loss_contrast(features, 
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        elif self.loss_version==7:
        # 7. 直接使用低维特征计算相似度 +  使用所有样本加权qi
            contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
            contrast_mask*=max_probs
            with torch.no_grad(): 
                cos_sim= 1 - cosine_similarity(f_u_s1.detach().cpu().numpy())
                mask = torch.from_numpy(cos_sim).cuda()
            loss_d_ctr = self.loss_contrast(features, 
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        loss=loss_cls+loss_cons+self.lambda_d*loss_d_ctr
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.losses_d_ctr.update(loss_d_ctr.item(),labels.shape[0]) 
        self.losses_x.update(loss_cls.item(), inputs_x.size(0))
        self.losses_u.update(loss_cons.item(), inputs_u_s.size(0)) 
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_c:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,
                        self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_d_ctr.avg))
             
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    # def get_mixed_feat(self,feat,pred_class,extra_feat=None):
    #     l = np.random.beta(self.mixup_alpha, self.mixup_alpha) 
    #     l = max(l, 1-l)
    #     idx = torch.randperm(feat.size(0))

    #     input_a, input_b = feat, feat[idx]
    #     target_a, target_b = pred_class, pred_class[idx]

    #     mixed_input = l * input_a + (1 - l) * input_b
    #     mixed_target = l * target_a + (1 - l) * target_b
    #     if extra_feat is not None:
    #         extra_a, extra_b = extra_feat, extra_feat[idx]
    #         mixed_extra_input = l * extra_a + (1 - l) * extra_b
    #         return mixed_input,mixed_target,mixed_extra_input
    #     return mixed_input,mixed_target
    
    # def train_debiased_step_v8(self,data_x,data_u):
    #     self.model.train()
    #     loss =0 
    
    #     inputs_x_w=data_x[0][0] 
    #     inputs_x_s=data_x[0][1] 
    #     targets_x=data_x[1]
        
    #     inputs_u_w=data_u[0][0]
    #     inputs_u_s=data_u[0][1]
        
    #     inputs = torch.cat(
    #             [inputs_x_w,inputs_x_s, inputs_u_w,inputs_u_s],
    #             dim=0).cuda()
        
    #     targets_x=targets_x.long().cuda()
        
    #     batch_size=targets_x.size(0)
        
    #     encoding = self.model(inputs,return_encoding=True)
    #     features=self.model(encoding,return_projected_feature=True)
    #     logits=self.model(encoding.detach(),classifier=True) 
        
    #     with torch.no_grad():
           
    #         bin_labels = F.one_hot(targets_x, num_classes=self.num_classes).float() 
    #         l_encoding_w,l_encoding_s=encoding[:batch_size*2].detach().chunk(2)
    #         mixed_w_feat,mixed_label,mixed_s_feat=self.get_mixed_feat( l_encoding_w, bin_labels,extra_feat=l_encoding_s)

    #     # 1. mixed ce loss          
        
    #     mixed_w_logits=self.model(mixed_w_feat,classifier=True)         
    #     mixed_s_logits=self.model(mixed_s_feat,classifier=True)         
    #     loss_cls = -torch.mean(torch.sum((F.log_softmax(mixed_w_logits, dim=1)+F.log_softmax(mixed_s_logits, dim=1))*0.5 * mixed_label, dim=1))
         
    #     # 2. cons loss 
    #     logits_u_w,logits_u_s=logits[2*batch_size:].chunk(2)         
    #     with torch.no_grad(): 
    #         probs_w = torch.softmax(logits_u_w.detach(), dim=-1)
    #         max_probs, pred_class = torch.max(probs_w, dim=-1)             
    #     loss_weight = max_probs.ge(self.conf_thres).float()  
    #     loss_cons = self.ul_criterion(logits_u_s, pred_class, weight=loss_weight, avg_factor=logits_u_w.size(0))
        
    #     # 3. ctr loss
    #     # === biased_contrastive_loss
    #     # for supervised contrastive
    #     l_feat_w,l_feat_s=features[:2*batch_size].chunk(2)
    #     u_feat_w,u_feat_s=features[2*batch_size:].chunk(2)
    #     f_u_w=torch.cat([l_feat_w,u_feat_w],dim=0)
    #     f_u_s=torch.cat([l_feat_s,u_feat_s],dim=0)
        
    #     labels = torch.cat([targets_x,pred_class],dim=0)   
    #     features = torch.cat([f_u_w.unsqueeze(1), f_u_s.unsqueeze(1)], dim=1) #torch.Size([128, 2, 64])
    #     contrast_mask = torch.cat([torch.ones(batch_size).cuda(),max_probs.ge(self.contrast_with_thresh).float()],dim=0)  
    #     max_probs=torch.cat([torch.ones(batch_size).cuda(),max_probs],dim=0)
    #     loss_d_ctr = self.loss_contrast(features, # projected_feature
    #                                     max_probs=max_probs, # confidence
    #                                     labels=labels, # pred_class
    #                                     reduction=None) #torch.Size([2, 128])
    #     loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
    #     # 2. 选择高置信度的样本对作为正对
    #     # with torch.no_grad():
    #     #     select_matrix = self.contrast_left_out_p(max_probs)
    #     # loss_d_ctr = self.loss_contrast(features,
    #     #                                 labels=labels,
    #     #                                 select_matrix=select_matrix)
    #     # 3. debiased pos weight
    #     # contrast_mask = max_probs.ge(
    #     #             self.contrast_with_thresh).float()
    #     # with torch.no_grad():
    #     #     biased_feat=self.biased_model(inputs[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)],return_encoding=True)
    #     #     cos_sim= 1 - cosine_similarity(biased_feat.detach().cpu().numpy())
    #     #     mask = torch.from_numpy(cos_sim).cuda()
    #     # loss_d_ctr = self.loss_contrast(features,
    #     #                                 # max_probs=max_probs,
    #     #                                 labels=labels,
    #     #                                 mask=mask,
    #     #                                 reduction=None)
        
    #     # 4. 样本软加权 qi*去偏对比损失
    #     # contrast_mask = max_probs.ge(
    #     #             self.contrast_with_thresh).float()
    #     # contrast_mask*=max_probs
    #     # with torch.no_grad():
    #     #     biased_feat=self.biased_model(inputs[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)],return_encoding=True)
    #     #     cos_sim= 1 - cosine_similarity(biased_feat.detach().cpu().numpy())
    #     #     mask = torch.from_numpy(cos_sim).cuda() 
    #     # loss_d_ctr = self.loss_contrast(features,
    #     #                                 # max_probs=max_probs,
    #     #                                 labels=labels,
    #     #                                 mask=mask,
    #     #                                 reduction=None)
    #     # loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
    #     # # 5. 直接使用高维特征计算相似度
    #     # contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
    #     # contrast_mask*=max_probs
    #     # with torch.no_grad(): 
    #     #     cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
    #     #     mask = torch.from_numpy(cos_sim).cuda()
    #     # loss_d_ctr = self.loss_contrast(features, 
    #     #                                 labels=labels,
    #     #                                 mask=mask,
    #     #                                 reduction=None)
        
    #     # loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
    #     # # 6. 直接使用高维特征计算相似度 +  使用所有样本加权qi
    #     # contrast_mask=max_probs
    #     # with torch.no_grad(): 
    #     #     cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
    #     #     mask = torch.from_numpy(cos_sim).cuda()
    #     # loss_d_ctr = self.loss_contrast(features, 
    #     #                                 labels=labels,
    #     #                                 mask=mask,
    #     #                                 reduction=None)
    #     # loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
    #     # 7. 直接使用低维特征计算相似度 +  使用所有样本加权qi
    #     # contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
    #     # contrast_mask=max_probs
    #     # with torch.no_grad(): 
    #     #     cos_sim= 1 - cosine_similarity(f_u_s1.detach().cpu().numpy())
    #     #     mask = torch.from_numpy(cos_sim).cuda()
    #     # loss_d_ctr = self.loss_contrast(features, 
    #     #                                 labels=labels,
    #     #                                 mask=mask,
    #     #                                 reduction=None)
    #     # loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
    #     loss=loss_cls+loss_cons+self.lambda_d*loss_d_ctr
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step() 
    #     self.losses_d_ctr.update(loss_d_ctr.item(),labels.shape[0]) 
    #     self.losses_x.update(loss_cls.item(), mixed_label.size(0))
    #     self.losses_u.update(loss_cons.item(), logits_u_w.size(0)) 
    #     self.losses.update(loss.item(),inputs_x_w.size(0))
        
    #     if self.iter % self.cfg.SHOW_STEP==0:
    #         self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_c:{:>5.4f} =='\
    #             .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
    #                     self.train_per_step,
    #                     self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_d_ctr.avg))
             
    #     return 
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def contrast_left_out_p(self, max_probs):
        """contrast_left_out
        If contrast_left_out, will select positive pairs based on
            max_probs > contrast_with_thresh, others will set to 0
            later max_probs will be used to re-weight the contrastive loss
        Args:
            max_probs (torch Tensor): prediction probabilities
        Returns:
            select_matrix: select_matrix with probs < contrast_with_thresh set
                to 0
            将高置信度的正对挑选出来
        """
        contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
        contrast_mask2 = torch.clone(contrast_mask)
        contrast_mask2[contrast_mask == 0] = -1
        select_elements = torch.eq(contrast_mask2.reshape([-1, 1]),
                                   contrast_mask.reshape([-1, 1]).T).float()
        select_elements += torch.eye(contrast_mask.shape[0]).cuda()
        select_elements[select_elements > 1] = 1
        select_matrix = torch.ones(contrast_mask.shape[0]).cuda() * select_elements
        return select_matrix
    
    def dcssl_contra_loss(self,feat=None,targets_x=None,topk_pred_label=None,biased_feat=None,sample_mask=None,temperature=0.07):
        # out_1=torch.cat([feat[:targets_x.shape[0]],feat[2*targets_x.shape[0]:2*targets_x.shape[0]+topk_pred_label[0].shape[0]]],dim=0) 
        # out_2=torch.cat([feat[targets_x.shape[0]:2*targets_x.shape[0]],feat[-topk_pred_label[0].shape[0]:]],dim=0)
        
        # similarity  = pairwise_similarity(out_1,out_2,temperature=temperature) 
        # loss        = NT_xent(similarity) 
        
        
        topk_pred_label=topk_pred_label.T
        y=torch.cat([targets_x,targets_x,topk_pred_label[0],topk_pred_label[0]],dim=0)
      
        similarity_matrix = (1./temperature) * torch.mm(feat,feat.transpose(0,1)) 
        # get weight by biased feature
        cos_sim=torch.from_numpy(cosine_similarity(biased_feat.cpu().numpy())).cuda()
        pos_weight = 1 - cos_sim
        neg_weight= cos_sim        
        # yy=torch.cat([targets_x,pred_class,pred_class],dim=0)
        pos_mask=torch.eq(y.contiguous().view(-1, 1),y.contiguous().view(-1, 1).T) 
        neg_mask=~pos_mask
        # pos_weight 需要乘以一个mask，mask掉其他的负类的 通过top-k
        # fp_mask=copy.deepcopy(top1_mask)
        for i in range(1,topk_pred_label.size(0)):
            tmp_y=torch.cat([targets_x,targets_x,topk_pred_label[i],topk_pred_label[i]],dim=0)
            tmp_mask=torch.eq(tmp_y.contiguous().view(-1, 1),tmp_y.contiguous().view(-1, 1).T) 
            pos_mask |= tmp_mask   
            neg_mask &= ~tmp_mask   
                  
         
        pos_mask=pos_mask.float() 
        neg_mask=neg_mask.float()
        
        # f1, f2 = feat.chunk(2)
        # cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
         
        # 去掉top-k个可能假阴，但也会因为可能混淆的样本也被抹去了
        pos_weight=pos_weight*pos_mask+ neg_mask*neg_weight
        pos_weight*=(1-torch.eye(pos_weight.shape[0],pos_weight.shape[0]).cuda())
        
        # for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # exp 
        exp_logits = torch.exp(logits)*pos_weight
        
        # log exp 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # weight
        sum_mask = pos_mask.sum(1) 
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask
        
        
        loss =(-mean_log_prob_pos* sample_mask).mean()
        # con_loss =(con_loss * sample_mask).mean()
        return loss
        # pos_weight=pos_weight*mask
        # 
        # logits_mask=(1-torch.eye(pos_weight.shape[0],pos_weight.shape[0]).cuda())
        # pos_weight = pos_weight * logits_mask # mask self-self

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # sum_mask = pos_mask.sum(1)+neg_mask.sum(1)
        # sum_mask[sum_mask == 0] = 1
        # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask
        
        # # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

        # # loss
        # loss = -mean_log_prob_pos #torch.Size([384])
        # loss = loss.mean()
        # return loss
        
       
        
        
        # N2  = len(similarity_matrix)
        # # N   = int(len(similarity_matrix) / 2)
        
        # mask=(pos_pair - torch.eye(N2,N2).cuda())
        # # Removing diagonal 
        # # 相似度
        # similarity_matrix_exp = torch.exp(similarity_matrix)
        
        # similarity_matrix_exp= similarity_matrix_exp.mul(pos_weight)+similarity_matrix_exp.mul(neg_weight)
        # similarity_matrix_exp = similarity_matrix_exp * mask
        # log_prob = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
        # # log_prob = logits - torch.log(similarity_matrix_exp.sum(1, keepdim=True) + 1e-9)
        
        # # compute mean of log-likelihood over positive
        # sum_mask= mask.sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(1) /(sum_mask+ 1e-9)
          
        # NT_xent_loss_total  = (1./float(final_index.shape[0])) * torch.sum(mean_log_prob_pos[final_index])

        # return NT_xent_loss_total
    
    # def get_val_model(self):
    #     return self.ema_model  
    
    # def evaluate(self,return_group_acc=False,return_class_acc=False):  
    #     # b_model=self.biased_model
        
    #     # # b_model=self.model
    #     # test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(b_model,self.test_loader, self.val_criterion)
        
    #     # if self.valset_enable:
    #     #     val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(b_model,self.val_loader, self.val_criterion)
    #     # else:
    #     #     val_loss, val_acc,val_group_acc,val_class_acc = test_loss, test_acc ,test_group_acc,test_class_acc
    #     # self.logger.info('== Online Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(val_acc*100,test_acc*100))
                        
    #     # d_model=self.get_val_model() 
    #     d_model=self.model
    #     test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(d_model,self.test_loader, self.val_criterion)
        
    #     if self.valset_enable:
    #         val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(d_model,self.val_loader, self.val_criterion)
    #     else:
    #         val_loss, val_acc,val_group_acc,val_class_acc = test_loss, test_acc ,test_group_acc,test_class_acc
        
            
    #     self.val_losses.append(val_loss)
    #     self.val_accs.append(val_acc)
    #     self.val_group_accs.append(val_group_acc)
    #     self.test_losses.append(test_loss)
    #     self.test_accs.append(test_acc)
    #     self.test_group_accs.append(test_group_acc)
        
    #     if return_group_acc:
    #         if return_class_acc:
    #             return val_acc,test_acc,test_group_acc,test_class_acc
    #         else:
    #             return val_acc,test_acc,test_group_acc
    #     if return_class_acc:
    #         return val_acc,test_acc,test_class_acc
    #     return [val_acc,test_acc]
    
    def get_pos_neg_weight(self,biased_encodings,target_x,pred):
        # 求个cosine相似度
        yy=torch.cat([target_x,pred,pred],dim=0)
        pos_mask=torch.eq(yy.contiguous().view(-1, 1),yy.contiguous().view(-1, 1).T).float()
        neg_mask=1-pos_mask
        # 每个正样本的加权
        # 越相似，权重越小
        cos_sim=torch.from_numpy(cosine_similarity(biased_encodings.cpu().numpy())).cuda()
        pos_weight = 1 - cos_sim
        pos_weight*=pos_mask
        neg_weight=neg_mask*cos_sim
        
        return pos_weight,neg_weight
     
    def get_u_loss_weight(self,confidence):
        # 根据logits阈值
        loss_weight=confidence.ge(self.conf_thres).float()  
        # 根据能量分数
        
        return loss_weight
    
    # def save_checkpoint(self,file_name=""):
    #     if file_name=="":
    #         file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
    #     torch.save({
    #                 'model': self.model.state_dict(), 
    #                 'biased_model': self.biased_model.state_dict(),
    #                 'iter': self.iter, 
    #                 'best_val': self.best_val, 
    #                 'best_val_iter':self.best_val_iter, 
    #                 'best_val_test': self.best_val_test,
    #                 'optimizer': self.optimizer.state_dict(), 
    #                 'biased_optimizer': self.biased_optimizer.state_dict(),  
    #             },  os.path.join(self.model_dir, file_name))
    #     return    
    
    # def load_checkpoint(self, resume) :
    #     self.logger.info(f"resume checkpoint from: {resume}")
    #     if not os.path.exists(resume):
    #         self.logger.info(f"Can\'t resume form {resume}")
            
    #     state_dict = torch.load(resume) 
    #     self.model.load_state_dict(state_dict['model'])
    #     self.biased_model.load_state_dict(state_dict["biased_model"]) 

    #     # load optimizer and scheduler 
    #     self.optimizer.load_state_dict(state_dict["optimizer"])  
    #     self.biased_optimizer.load_state_dict(state_dict["biased_optimizer"])   
    #     self.start_iter=state_dict["iter"]+1
    #     self.best_val=state_dict['best_val']
    #     self.best_val_iter=state_dict['best_val_iter']
    #     self.best_val_test=state_dict['best_val_test']  
    #     self.epoch= (self.start_iter // self.train_per_step)+1 
    #     self.logger.info(
    #         "Successfully loaded the checkpoint. "
    #         f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
    #     )