
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
class ReSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        # 需要先捕获偏差特征  
        
        self.self.losses_rc=AverageMeter() 
        self.t1=cfg.ALGORITHM.RESSL.T1
        self.t2=cfg.ALGORITHM.RESSL.T2
        self.K = cfg.ALGORITHM.RESSL.K
        self.m = cfg.ALGORITHM.RESSL.M 
        if cfg.MODEL.NAME=='WRN_28_2':
            dim=64
        else:
            dim=128
        self.ema_model=self.build_model(cfg).cuda()
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
         # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_rc=AverageMeter() 
        
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
        
        inputs_u=data_u[0][0]
        inputs_u2=data_u[0][1]
        inputs_x=data_x[0]
        targets_x=data_x[1]
         
        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
          
        inputs_x=inputs_x.cuda()
        targets_x=targets_x.long().cuda()
        
        logits=self.model(inputs_x,classifier=True)
         
        # 1. ce loss 
        loss_cls = self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        # 2. cons loss

        
        # 3. relation ctr loss 
        q = self.model(inputs_u,return_encoding=True)
        q = self.model(q,return_projected_feature=True)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            # shuffle for making use of BN 
            k = self.ema_model(inputs_u2,return_encoding=True)  # keys: NxC
            k = self.ema_model(k,return_projected_feature=True)  # keys: NxC 

        logitsq = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logitsk = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
         
        loss_re_ctr = - torch.sum(F.softmax(logitsk.detach() / self.t1, dim=1) * F.log_softmax(logitsq / self.t2, dim=1), dim=1).mean()
        loss=loss_cls+loss_cons+loss_re_ctr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.losses_x.update(loss_cls.item(), inputs_x.size(0))
        # self.losses_u.update(loss_cons.item(), inputs_u.size(0)) 
        self.losses_rc.update(loss_re_ctr.item(), inputs_u.size(0)) 
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
        # if self.iter % 2==0:
            self.logger.info('== Debiased Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Loss_d_ctr:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,self.losses.avg,
                        self.losses_x.avg,self.losses_u.avg,
                        self.losses_d_ctr.avg))
            self.logger.info("=="*40)
            # self.logger.info('=================  ==============='.format(self.losses_d_ctr.avg,))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'model': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'optimizer': self.optimizer.state_dict(), 
                },  os.path.join(self.model_dir, file_name))
        return    
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")
        assert os.path.exists(resume)
            
        state_dict = torch.load(resume) 
        self.model.load_state_dict(state_dict['model'])
        self.ema_model.load_state_dict(state_dict["ema_model"]) 
        for param_k in self.ema_model.parameters(): 
            param_k.requires_grad = False  # not update by gradient
        # load optimizer and scheduler 
        self.optimizer.load_state_dict(state_dict["optimizer"])   
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.train_per_step)+1 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )