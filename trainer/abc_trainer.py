
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse

from utils.misc import AverageMeter  
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

from loss.contrastive_loss import *
from utils import FusionMatrix
from models.projector import  Projector 

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class ABCTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)      
        if self.warmup_enable:
            self.rebuild_unlabeled_dataset_enable=True
        self.train_criterion =SemiLoss() 
        
        self.ir2=min(self.labeled_trainloader.dataset.num_per_cls_list)/self.labeled_trainloader.dataset.num_per_cls_list 
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_abc = AverageMeter() 
        
    def train_step(self,pretraining=False):
        if self.pretraining:
            return self.train_warmup_step()
        else:
            return self.train_abc_step()
    
    def train_abc_step(self):
        self.model.train()
        loss =0
        # DL  
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        if  isinstance(inputs_x,list)  :
            inputs_x=inputs_x[0]
        
        # DU   
        try:       
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
        inputs_u3=data[0][2]
        u_index=data[2]
        
        # id_mask_idx=torch.where(self.id_masks[u_index].detach()==1)[0]
        # inputs_u=inputs_u[id_mask_idx]
        # inputs_u2=inputs_u2[id_mask_idx]
        # inputs_u3=inputs_u3[id_mask_idx]
        
        batch_size = inputs_x.size(0) 
        num_class=self.num_classes
        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, self.num_classes).scatter_(1, targets_x.view(-1,1), 1)
        
        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
        
         # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            q1=self.model(inputs_u,return_encoding=True)
            outputs_u= self.model(q1,classifier1=True)
            targets_u2 = torch.softmax(outputs_u, dim=1).detach()

        targets_u = torch.argmax(targets_u2, dim=1)

        q = self.model(inputs_x,return_encoding=True)
        q2 = self.model(inputs_u2,return_encoding=True)
        q3 = self.model(inputs_u3,return_encoding=True)

        max_p, p_hat = torch.max(targets_u2, dim=1)
        p_hat = torch.zeros(p_hat.size(0), num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        select_mask = max_p.ge(0.95)
        select_mask = torch.cat([select_mask, select_mask], 0).float()

        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)

        logits_x=self.model(q,classifier1=True)
        logits_u1=self.model(q2,classifier1=True)
        logits_u2=self.model(q3,classifier1=True)
        logits_u = torch.cat([logits_u1,logits_u2],dim=0)
        # tmp_ir2=torch.tensor(np.array(self.ir2)).cuda()
        # tmp2=targets_x2.detach() * tmp_ir2
        # maskforbalance = torch.bernoulli(torch.sum(tmp2, dim=1).detach())
        
        maskforbalance = torch.bernoulli(torch.sum(targets_x2.cpu() * torch.tensor(self.ir2), dim=1).detach()).cuda()


        logit = self.model(q,classifier2=True)
        logitu1 = self.model(q1,classifier2=True)
        logitu2 = self.model(q2,classifier2=True)
        logitu3 = self.model(q3,classifier2=True)

        logits = F.softmax(logit,1)
        logitsu1 = F.softmax(logitu1,1)
        max_p2, label_u = torch.max(logitsu1, dim=1)
        select_mask2 = max_p2.ge(0.95)
        label_u = torch.zeros(label_u.size(0), num_class).scatter_(1, label_u.cpu().view(-1, 1), 1)
        
         
        ir22 = 1 - (self.epoch / self.max_epoch) * (1 - self.ir2)
        maskforbalanceu = torch.bernoulli(torch.sum(label_u.cuda() * torch.tensor(ir22).cuda(), dim=1).detach())
        logitsu2 = F.softmax(logitu2,1)
        logitsu3 = F.softmax(logitu3,1)

        abcloss = -torch.mean(maskforbalance * torch.sum(torch.log(logits) * targets_x2.cuda(), dim=1))
        abcloss1 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu2) * logitsu1.cuda().detach(), dim=1))

        abcloss2 = -torch.mean(
            select_mask2 * maskforbalanceu * torch.sum(torch.log(logitsu3) * logitsu1.cuda().detach(), dim=1))

        totalabcloss= abcloss + abcloss1 + abcloss2
        Lx, Lu = self.train_criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu+ totalabcloss

        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_u.update(Lu.item(), inputs_x.size(0))
        self.losses_abc.update(abcloss.item(), inputs_x.size(0))  

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_abc:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_abc.avg))
        return 
    
    