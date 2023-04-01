
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
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
from utils import FusionMatrix
from models.projector import  Projector 
from dataset.base import BaseNumpyDataset
from dataset.build_dataloader import _build_loader
from utils.misc import AverageMeter  
from loss.soft_supconloss import SoftSupConLoss
class CCSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)      
        
        self.losses_c_ctr=AverageMeter() 
        self.u_T=cfg.ALGORITHM.CCSSL.U_T
        self.contrast_with_thresh=cfg.ALGORITHM.CCSSL.CONTRAST_THRESH
        self.lambda_c=cfg.ALGORITHM.CCSSL.LAMBDA_C
        self.temperature=cfg.ALGORITHM.CCSSL.TEMPERATURE
        self.loss_contrast=SoftSupConLoss(temperature=self.temperature)
        
        self.contrast_left_out=cfg.ALGORITHM.CCSSL.CONTRAST_LEFT_OUT
        self.contrast_with_softlabel=cfg.ALGORITHM.CCSSL.CONTRAST_WITH_SOFTLABEL 
        self.contrast_with_thresh=cfg.ALGORITHM.CCSSL.CONTRAST_WITH_THRESH
        
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
      
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()  
        self.losses_c_ctr=AverageMeter()
         
    def train_step(self,pretraining=False): 
        self.model.train()
        loss =0 
        try:        
            data_x = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            data_x = self.labeled_train_iter.next() 
        
        
        inputs_x=data_x[0] 
        targets_x=data_x[1]
        
        # DU   
        try:       
            data_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data_u = self.unlabeled_train_iter.next()
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
            probs_u_w = torch.softmax(logits_u_w.detach() / self.u_T, dim=-1)
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
        if labels.shape[0] != 0:
            if self.contrast_with_softlabel:
                select_matrix = None
                if self.contrast_left_out:
                    with torch.no_grad():
                        select_matrix = self.contrast_left_out(max_probs)
                    Lcontrast = self.loss_contrast(features,
                                                   max_probs,
                                                   labels,
                                                   select_matrix=select_matrix)

                elif self.contrast_with_thresh:
                    contrast_mask = max_probs.ge(
                        self.contrast_with_thresh).float()
                    # ================== feature norm =============== 
                    
                    Lcontrast = self.loss_contrast(features, # projected_feature
                                                   max_probs, # confidence
                                                   labels, # pred_class
                                                   reduction=None) #torch.Size([2, 128])
                    Lcontrast = (Lcontrast * contrast_mask).mean()

                else:
                    Lcontrast = self.loss_contrast(features, max_probs, labels)
            else:
                if self.contrast_left_out:
                    with torch.no_grad():
                        select_matrix = self.contrast_left_out_p(max_probs)
                    Lcontrast = self.loss_contrast(features,
                                                   labels,
                                                   select_matrix=select_matrix)
                else:
                    Lcontrast = self.loss_contrast(features, labels)

        else:
            Lcontrast = sum(features.view(-1, 1)) * 0
        loss=loss_cls+loss_cons+self.lambda_c*Lcontrast
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.losses_c_ctr.update(Lcontrast.item(),labels.shape[0]) 
        self.losses_x.update(loss_cls.item(), inputs_x.size(0))
        self.losses_u.update(loss_cons.item(), inputs_u_s.size(0)) 
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_c:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,
                        self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_c_ctr.avg))
             
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
    
    
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
        return 