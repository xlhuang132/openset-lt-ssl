
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
from loss.ressl_loss import *
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
        
        self.losses_rc=AverageMeter() 
        self.temperature_q=cfg.ALGORITHM.RESSL.TEMP_Q 
        self.temperature_k=cfg.ALGORITHM.RESSL.TEMP_K
        self.K = cfg.ALGORITHM.RESSL.K
        self.m = cfg.ALGORITHM.RESSL.M 
        self.lambda_r=cfg.ALGORITHM.RESSL.LAMBDA_R
        
        if cfg.MODEL.NAME=='WRN_28_2':
            dim=64
        else:
            dim=128
        self.ema_model=self.ema_model.cuda()
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
         # create the queue 
        # self.register_buffer("queue", torch.randn(dim, self.K)) # nn.module 里的函数，只能手动更新，但保存参数的时候会一起保存
        self.queue=torch.randn(self.K,dim).cuda()
        self.queue = nn.functional.normalize(self.queue, dim=0)  
        self.queue_ptr=torch.zeros(1, dtype=torch.long).cuda()
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
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
        
        logits_x=self.model(inputs_x)
         
        # 1. ce loss 
        loss_cls = self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        # 2. cons loss 
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_feature=self.model(ul_images,return_encoding=True) 
        ul_logits = self.model(ul_feature,classifier=True) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(self.conf_thres).float()
         
        loss_cons = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        ) 

        
        # 3. relation ctr loss 
        q = self.model(inputs_u,return_encoding=True)
        q = self.model(q,return_projected_feature=True)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            # shuffle for making use of BN 
            k = self.ema_model(inputs_u2,return_encoding=True)  # keys: NxC
            k = self.ema_model(k,return_projected_feature=True)  # keys: NxC 

        loss_re_ctr = ressl_loss_func(q, k, self.queue.clone().detach(), self.temperature_q, self.temperature_k)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
         
        loss=loss_cls+loss_cons+loss_re_ctr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.losses_x.update(loss_cls.item(), inputs_x.size(0))
        self.losses_u.update(loss_cons.item(), inputs_u.size(0)) 
        self.losses_rc.update(loss_re_ctr.item(), inputs_u.size(0)) 
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
        # if self.iter % 2==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_rc:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,self.losses.avg,
                        self.losses_x.avg,self.losses_u.avg,
                        self.losses_rc.avg))  
        
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
        # keys = concat_all_gather(keys) # ddp

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'model': self.model.state_dict(),
                    'ema_model': self.ema_model.state_dict(),
                    'queue':self.queue,
                    'queue_ptr':self.queue_ptr,
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'optimizer': self.optimizer.state_dict(), 
                    'id_masks':self.id_masks,
                    'ood_masks':self.ood_masks,                   
                },  os.path.join(self.model_dir, file_name))
        return 
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")

        state_dict = torch.load(resume)
        # load model 
        model_dict=self.model.state_dict() 
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict["model"].items() if k in self.model.state_dict()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        # self.model.load_state_dict(state_dict["model"]) 
        try:
            self.id_masks=state_dict['id_masks']
            self.ood_masks=state_dict['ood_masks']  
        except:
            self.logger.warning('load id_masks or ood_masks wrong!')
        # load ema model 
        try:
            self.ema_model.load_state_dict(state_dict["ema_model"])
        except:
            self.logger.warning('load ema model wrong!')

        # load optimizer and scheduler 
        self.optimizer.load_state_dict(state_dict["optimizer"])  
        self.queue=state_dict['queue']
        self.queue_ptr=state_dict['queue_ptr'] 
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.train_per_step) 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )
        if self.rebuild_unlabeled_dataset_enable :
            id_index=torch.nonzero(self.id_masks == 1, as_tuple=False).squeeze(1)
            id_index=id_index.cpu().numpy()
            self.rebuild_unlabeled_dataset(id_index) 
     
    