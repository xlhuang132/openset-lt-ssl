
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
from loss.build_loss import build_loss 
class MTCFTrainer(BaseTrainer):      
     
    def __init__(self, cfg):        
        super().__init__(cfg)       
        self.domain_train_iter=iter(self.domain_trainloader)        
        self.results = np.zeros((len(self.domain_trainloader.dataset)), dtype=np.float32)
    
    def build_loss(self):
        self.l_criterion,self.val_criterion = build_loss(self.cfg,combine=True)
        return 
    
    def train_step(self,pretraining=False):
        if self.pretraining:
            return self.train_warmup_step()
        else:
            return self.train_mtcf_step() 
        
    def train_warmup_step(self):
        self.model.train()
        loss =0 
        try:
            data= self.domain_train_iter.next()    
        except:            
            self.domain_train_iter=iter(self.domain_trainloader)   
            data = self.domain_train_iter.next() 
        inputs=data[0]
        if len(inputs)==2:
            inputs=inputs[0]
        # targets=data[1]
        domain_labels=data[2] 
        indexs=data[3]
        inputs, domain_labels = inputs.cuda(), domain_labels.cuda(non_blocking=True)
        _,logits = self.model(inputs) 
        probs = torch.sigmoid(logits).view(-1)
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1,1))
        self.results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()
        loss = Ld
        # record loss
        self.losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        return  
    
    def train_mtcf_step(self):
        self.model.train()
        loss =0
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
        # DU with domain label  
        try:
            inputs, _, domain_labels, indexs = self.domain_train_iter.next()
        except:
            train_iter = iter(self.domain_trainloader)
            inputs, _, domain_labels, indexs = train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
         
        
        inputs_x, targets_x_old = inputs_x.cuda(), targets_x.cuda(non_blocking=True)        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()          
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0) 
        
        # Transform label to one-hot 
        original_targets_x=targets_x.long()

        targets_x = torch.zeros(targets_x_old.size(0), self.num_classes).cuda().scatter_(1, targets_x_old.view(-1,1), 1)
 
         
        domain_labels = domain_labels.cuda()
        if len(inputs)==2:
            inputs=inputs[0]
        inputs=inputs.cuda()
        self.model.apply(self.set_bn_eval)
        _,logits = self.model(inputs)
        self.model.apply(self.set_bn_train)

        probs = torch.sigmoid(logits).view(-1)
        Ld = F.binary_cross_entropy_with_logits(logits, domain_labels.view(-1,1))

        self.results[indexs.detach().numpy().tolist()] = probs.cpu().detach().numpy().tolist()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u,_ = self.model(inputs_u)
            outputs_u2,_= self.model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/self.cfg.ALGORITHM.MTCF.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(self.cfg.ALGORITHM.MTCF.MIXUP_ALPHA,self.cfg.ALGORITHM.MTCF.MIXUP_ALPHA)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, targets_x.size(0)))
        mixed_input = self.interleave(mixed_input, targets_x.size(0))

        
        logits = [self.model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(self.model(input)[0])

        # put interleaved samples back
        logits = self.interleave(logits, targets_x.size(0))
        logits_x = logits[0]
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = self.l_criterion(logits_x, mixed_target[:targets_x.size(0)], logits_u, mixed_target[targets_x.size(0):], self.epoch+self.iter/self.val_iter)

        loss = Ld + Lx + w * Lu
        # self.iter
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_u.update(Lu.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        # current_lr = self.optimizer.param_groups[0]["lr"]
        # ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
 
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.val_iter if self.iter%self.val_iter>0 else self.val_iter,self.val_iter,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x_old.cpu().numpy()
    
    def operate_after_epoch(self):         
         
        if self.iter==self.warmup_iter:
            self.save_checkpoint(file_name="warmup_model.pth")
        
        self.domain_trainloader.dataset.soft_label_update(self.results) 
        
        self.results = np.zeros((len(self.domain_trainloader.dataset)), dtype=np.float32)
        self.logger.info('=='*40)  
          
    # def get_val_model(self,):
        # return self.ema_model
        
    
    def linear_rampup(current, rampup_length=16):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)      

    def set_bn_eval(self,module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False

    def set_bn_train(self,module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = True

    def interleave(self,xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
    
    def interleave_offsets(self,batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets
