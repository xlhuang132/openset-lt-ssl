
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
import models 
import time 
import torch.optim as optim 
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer

from utils import FusionMatrix
class SupervisedTrainer(BaseTrainer):        
    
    def train_step(self,pretraining=False):
        self.model.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next()   
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)     
         
        logits_x = self.model(inputs_x) 
        # loss computation 
        loss=self.l_criterion(logits_x, targets_x.long()) 
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)          
         
  
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(loss.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    def get_val_model(self,):
        return self.model
