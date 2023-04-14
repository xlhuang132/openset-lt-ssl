
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

from models.feature_queue import FeatureQueue

class DCSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        
        self.lambda_d=cfg.ALGORITHM.DCSSL.LAMBDA_D 
        self.m=cfg.ALGORITHM.DCSSL.M
        # self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True) 
        self.debiased_contra_temperture=cfg.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE
        # self.biased_fusion_matrix=FusionMatrix(self.num_classes)        
        self.loss_contrast= DebiasSoftConLoss(temperature=self.debiased_contra_temperture)
        self.contrast_with_thresh=cfg.ALGORITHM.DCSSL.CONTRAST_THRESH
        self.contrast_with_hp=cfg.ALGORITHM.DCSSL.CONTRAST_WITH_HP
        self.contrast_wwith_hn=cfg.ALGORITHM.DCSSL.CONTRAST_WITH_HN
        self.losses_d_ctr=AverageMeter()
        self.losses_bx = AverageMeter()
        self.losses_bu = AverageMeter()  
        # self.biased_model=self.build_model(cfg).cuda()
        # self.biased_optimizer=self.build_optimizer(cfg, self.biased_model)  
        self.mixup_alpha=0.5
        self.sharpen_temp=0.5
        # self.gce_loss=GeneralizedCELoss()
        self.dynamic_thresh=torch.tensor([0.5]*self.num_classes).cuda()
        self.means=torch.zeros(self.num_classes,64 if self.cfg.MODEL.NAME in ['WRN_28_2','WRN_28_8'] else 128).cuda()
        self.covs=torch.zeros(self.num_classes,64 if self.cfg.MODEL.NAME in ['WRN_28_2','WRN_28_8'] else 128).cuda()
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
        # del logits
        # del features
        # del _ 
        
        # 1. ce loss          
        loss_cls = self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        with torch.no_grad(): 
            self.update_mean_cov(features[:inputs_x.size(0)].clone().detach(), targets_x.clone().detach())
         
        # 2. cons loss 
        # filter out low confidence pseudo label by self.cfg.threshold 
        with torch.no_grad(): 
            probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
            # pseudo label and scores for u_w
            max_probs, pred_class = torch.max(probs_u_w, dim=-1)             
        loss_weight = max_probs.ge(self.conf_thres).float()  
        # loss_weight=self.select_du_samples(max_probs, pred_class, dynamic_thresh=True,update_thresh=[probs_u_w,pred_class])
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
        # # 1. 高置信度正样本+pi*pj ccssl x
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            loss_d_ctr = self.loss_contrast(features, # projected_feature
                                            max_probs=max_probs, # confidence
                                            labels=labels, # pred_class
                                            reduction=None) #torch.Size([2, 128])
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        elif self.loss_version==2:
        # 2. 选择高置信度的样本对作为正对 + pi*pj x
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
        # 3. pos weight + neg weight x
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            with torch.no_grad():
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                pos_mask = torch.from_numpy(cos_sim).cuda()
                mask= torch.eq(labels, labels.T).float().cuda()
                mask=pos_mask*mask+(1-pos_mask)*(1-mask)
            loss_d_ctr = self.loss_contrast(features, 
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        # 4.  pos weight + neg weight + pi*pj x
        elif self.loss_version==4:
            contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
            with torch.no_grad():
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                pos_mask = torch.from_numpy(cos_sim).cuda()
                mask= torch.eq(labels, labels.T).float().cuda()
                mask=(1+pos_mask)*mask+(2-pos_mask)*(1-mask)
            loss_d_ctr = self.loss_contrast(features,
                                            max_probs=max_probs,  
                                            labels=labels,
                                            mask=mask,
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        
        elif self.loss_version==5:
        # 5. 直接使用高维特征计算相似度，不用pi*pj
            contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
            with torch.no_grad(): 
                cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                pos_mask = torch.from_numpy(cos_sim).cuda()
                mask= torch.eq(labels, labels.T).float().cuda()
                mask=(1+pos_mask)*mask+(2-pos_mask)*(1-mask)
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
        elif self.loss_version==8: # 单纯的有监督对比
            contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
            loss_d_ctr = self.loss_contrast(features, 
                                            labels=labels, 
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
        elif self.loss_version>=9:
            # contrast_mask=max_probs.ge(self.contrast_with_thresh).float()
            if self.iter>100:
                # v9: score
                # v10: 2-score
                if self.loss_version==9:
                    contrast_mask=self.get_contrast_weight(f_u_s1, pred_class)
                elif self.loss_version==10:
                    contrast_mask= 1-self.get_contrast_weight(f_u_s1, pred_class)
                elif self.loss_version==11:
                    contrast_mask=2-max_probs
                if contrast_mask.shape[0]>0:
                    with torch.no_grad(): 
                        cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                        pos_mask = torch.from_numpy(cos_sim).cuda()
                        mask= torch.eq(labels, labels.T).float().cuda()
                        mask=(1+pos_mask)*mask+(2-pos_mask)*(1-mask)
                    loss_d_ctr = self.loss_contrast(features, 
                                                    labels=labels, 
                                                    reduction=None)
                    loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
                else:loss_d_ctr=torch.tensor(0.).cuda()
            else:loss_d_ctr=torch.tensor(0.).cuda()
        
            
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
    
    
    def get_contrast_weight(self,features,pred_class):
        # 获得每个样本的价值，如何评估每个样本是否在边界上？
        prototypes = self.means.clone().detach() 
        means=prototypes[pred_class]          
        score=torch.cosine_similarity(features,means,dim=-1)  
        return score
    
    def update_mean_cov(self,features,y): 
        uniq_c = torch.unique(y)
        for c in uniq_c:
            c = int(c)
            if c==-1:continue
            select_index = torch.nonzero(y == c, as_tuple=False).squeeze(1)
            if select_index.shape[0]>0:
                embedding_temp = features[select_index]  
                mean = embedding_temp.mean(dim=0)
                var = embedding_temp.var(dim=0, unbiased=False)
                self.means[c]= self.m*self.means[c]+(1-self.m)*mean 
                self.covs[c]=self.m * self.covs[c]+(1-self.m)* var 
    
    def select_du_samples(self,confidence, pred_class, fixed_thresh=None,update_thresh=None,dynamic_thresh=False):
        assert fixed_thresh is not None or dynamic_thresh and update_thresh is not None
        if not dynamic_thresh:
            return confidence.ge(fixed_thresh).float()
        else:
            # update_thresh[0]: probability update_thresh[1]:target
            assert update_thresh and len(update_thresh)==2
            self.update_dynamic_thresh(update_thresh[0], update_thresh[1])
            cur_thresh=self.dynamic_thresh[pred_class]
            loss_weight=(confidence > cur_thresh).float()
            return loss_weight
    
    @torch.no_grad()
    def update_dynamic_thresh(self,probs,pred_class):
        for c in range(self.num_classes):
            select_idx=torch.nonzero( pred_class==c , as_tuple=False).squeeze(1)
            if select_idx.shape[0]!=0:
                tmp=probs[select_idx][:,c]
                select_idx=torch.nonzero( tmp>=self.dynamic_thresh[c] , as_tuple=False).squeeze(1)
                if select_idx.shape[0]>0:
                    conf=tmp[select_idx].mean()
                    self.dynamic_thresh[c]= self.m*self.dynamic_thresh[c]+ (1-self.m)*conf
      
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
   