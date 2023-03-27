
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
from loss.gce import GeneralizedCELoss
from utils import FusionMatrix
from models.projector import  Projector 
from dataset.base import BaseNumpyDataset
from dataset.build_dataloader import _build_loader
from utils.misc import AverageMeter  
from models.attention_head import MultiHeadSelfAttention
class ACSSLTrainer(BaseTrainer): 
    """
    AttentionContrastive SSL
    
    """  
    def __init__(self, cfg):        
        super().__init__(cfg)  
        if cfg.MODEL.NAME=='Resnet50':   
            dim_in=128
            dim_k=128
            dim_v=128
        elif cfg.MODEL.NAME=='WRN_28_2':
            dim_in=64
            dim_k=64
            dim_v=64
        self.multiattention_head=MultiHeadSelfAttention(dim_in=dim_in, dim_k=dim_k, dim_v=dim_v,num_heads=cfg.ALGORITHM.ACSSL.HEAD_NUM).cuda()
        self.losses_a = AverageMeter() 
        self.t1=cfg.ALGORITHM.ACSSL.T1 # 0.04
        self.t2=cfg.ALGORITHM.ACSSL.T2 # 0.1
        self.lambda_a=1.
        
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_a=AverageMeter() 
        
    def train_step(self,pretraining=False): 
        
        self.model.train()
        loss =0
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
        
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        
        # DU   
        try:       
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
         
        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()          
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0) 
        
        # fixmatch pipelines
        feats=self.model(x,return_encoding=True)
        encodings=self.model(feats,return_projected_feature=True)
        logits_concat = self.model(feats,classifier=True)
        
        num_labels=inputs_x.size(0)
        logits_x = logits_concat[:num_labels]

        # loss computation 
        lx=self.l_criterion(logits_x, targets_x.long()) 
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        with torch.no_grad():
            # compute pseudo-label
            p = logits_weak.softmax(dim=1)  # soft pseudo labels
            confidence, pred_class = torch.max(p.detach(), dim=1) 
            loss_weight = confidence.ge(self.conf_thres).float()
         
        lu = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=pred_class.size(0)
        ) 
        
        l_encoding=encodings[:num_labels]
        ul_w_encoding,ul_s_encoding=encodings[num_labels:].chunk(2)
        
        la=self.get_attention_contrastive_loss(l_encoding,ul_w_encoding,ul_s_encoding)
        
        loss+=lx+lu+self.lambda_a*la
        # loss+=lx+self.lambda_a*la
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(lx.item(), inputs_x.size(0))
        self.losses_u.update(lu.item(), inputs_u.size(0)) 
        self.losses_a.update(la.item(),inputs_x.size(0)+inputs_u.size(0))
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_a:{:>5.4f} =='.
                             format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                                    self.train_per_step,self.losses.avg,self.losses_x.avg,
                                    self.losses_u.avg,self.losses_a.avg))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    
    
    def get_attention_contrastive_loss(self,l_encoding,ul_w_encoding,ul_s_encoding,temperature=0.07):
        
        logitsk=self.multiattention_head(torch.cat([l_encoding,ul_w_encoding],dim=0).unsqueeze(1)).squeeze()
        logitsq=self.multiattention_head(torch.cat([l_encoding,ul_s_encoding],dim=0).unsqueeze(1)).squeeze()
        loss = - torch.sum(F.softmax(logitsk.detach() / self.t1, dim=1) * F.log_softmax(logitsq / self.t2, dim=1), dim=1).mean()
    
        return loss
     
     