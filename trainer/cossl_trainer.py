
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse 

import random
from utils.misc import AverageMeter  
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
import models 
from torch.utils.data.sampler import WeightedRandomSampler
import time 
import torch.utils.data as data
import torch.optim as optim 
from dataset.build_transform import get_strong_transform
from dataset.base import BaseNumpyDataset
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer
from utils.ema_model import * 
from utils import FusionMatrix
from models.projector import  Projector 
from utils.build_optimizer import get_param_optimizer
from torch.utils.data.sampler import BatchSampler

def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / abs(gamma), 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    # print(class_num_list)
    return list(class_num_list)

def get_weighted_sampler(target_sample_rate, num_sample_per_class, target):
    assert len(num_sample_per_class) == len(np.unique(target))

    sample_weights = target_sample_rate / num_sample_per_class  # this is the key line!!!

    # assign each sample a weight by sampling rate
    samples_weight = np.array([sample_weights[t] for t in target])

    return WeightedRandomSampler(samples_weight, len(samples_weight), True)
  

class FixMatch_Loss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        return Lx, Lu

class CoSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)      
        self.train_criterion=FixMatch_Loss()
         # Define a new classifier for TFE branch
        self.ema_model=self.ema_model.cuda()
        self.ema_optimizer = WeightEMA(
            self.model, self.ema_model, 
            cfg.MODEL.OPTIMIZER.BASE_LR, 
            alpha=cfg.MODEL.EMA_DECAY)
        self.teacher_head = nn.Linear(self.model.fc.in_features, self.num_classes, bias=True).cuda()
        self.ema_teacher_head = create_ema_model(self.teacher_head)  
        self.wd_tfe=cfg.ALGORITHM.CoSSL.WD_TFE 
        self.lr_tfe=cfg.ALGORITHM.CoSSL.LR_TFE
        self.ema_decay_tfe=cfg.ALGORITHM.CoSSL.EMA_DECAY_TFE
        self.warm_epoch_tfe=cfg.ALGORITHM.CoSSL.WARM_EPOCH_TFE
        self.max_lam=cfg.ALGORITHM.CoSSL.MAX_LAM
        self.num_samples_per_class=self.labeled_trainloader.dataset.num_per_cls_list
        wd_params, non_wd_params = [], []
        for name, param in self.teacher_head.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        param_list = [{'params': wd_params, 'weight_decay': self.wd_tfe}, {'params': non_wd_params, 'weight_decay': 0}]
        self.teacher_head_optimizer = get_param_optimizer(cfg, param_list) 
        self.ema_teacher_head_optimizer = WeightEMA(
            self.teacher_head, self.ema_teacher_head, \
            cfg.MODEL.OPTIMIZER.BASE_LR, alpha=cfg.MODEL.EMA_WEIGHT_DECAY, wd=False)
        
        self.mask_prob = AverageMeter()
        self.total_c = AverageMeter()
        self.used_c = AverageMeter()
        # Different classes have different TFE probability
        self.tfe_prob = [(max(self.num_samples_per_class) - i) / max(self.num_samples_per_class) for i in self.num_samples_per_class]
 
        
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()
        self.losses_teacher = AverageMeter()  
        
    def build_crt_loader(self):
        l_dataset = self.labeled_trainloader.dataset 
        l_data_np = l_dataset.select_dataset()
        strong_transform = get_strong_transform(self.cfg) 
        crt_labeled_dataset = BaseNumpyDataset(l_data_np, transforms=strong_transform,num_classes=self.num_classes) 
        crt_full_set =  self.pre_train_loader.dataset 
        full_data_np = crt_full_set.select_dataset()
        crt_full_dataset = BaseNumpyDataset(full_data_np, transforms=strong_transform,num_classes=self.num_classes) 
        class_balanced_disb = torch.Tensor(make_imb_data(30000, self.num_classes, 1))
        class_balanced_disb = self.num_samples_per_class / class_balanced_disb.sum()
        sampler_x = get_weighted_sampler(class_balanced_disb, torch.Tensor(self.num_samples_per_class), crt_labeled_dataset.dataset['labels'])
        batch_sampler_x = BatchSampler(sampler_x, batch_size=self.batch_size, drop_last=True)
        self.crt_labeled_loader = data.DataLoader(crt_labeled_dataset, batch_sampler=batch_sampler_x) 
        self.crt_labeled_iter = iter(self.crt_labeled_loader)  
        
        self.crt_full_loader=data.DataLoader(crt_full_dataset, batch_size=self.batch_size, shuffle=True,
                                      drop_last=True)
        self.crt_full_iter = iter(self.crt_full_loader) 
        
        return
        
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        # pretrain
        self.logger.info('====================== pretrain phase: ======================')
        for self.iter in range(self.start_iter, self.max_iter): 
            return_data=self.train_pretrain_step( )
            if return_data is not None:
                pred,gt=return_data[0],return_data[1]
                fusion_matrix.update(pred, gt) 
            if self.iter%self.train_per_step==0:  
                end_time = time.time()           
                time_second=(end_time - start_time)
                eta_seconds = time_second * (self.max_epoch - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
                self.train_group_accs.append(group_acc)
                results=self.evaluate()
                
                if self.best_val<results[0]:
                    self.best_val=results[0]
                    self.best_val_test=results[1]
                    self.best_val_iter=self.iter
                    self.save_checkpoint(file_name="best_model.pth")
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint()
                self.train_losses.append(self.losses.avg)
                self.logger.info("== Pretraining phase")
                self.logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f} '.\
                    format(self.losses.avg, self.losses_x.avg, self.losses_u.avg))
                self.logger.info('== val_losss:{:>5.4f}   test_loss:{:>5.4f}   epoch_Time:{:>5.2f}min eta:{}'.\
                        format(self.val_losses[-1], self.test_losses[-1],time_second / 60,eta_string))
                self.logger.info('== Train  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.train_group_accs[-1][0]*100,self.train_group_accs[-1][1]*100,self.train_group_accs[-1][2]*100))
                self.logger.info('==  Val   group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.val_group_accs[-1][0]*100,self.val_group_accs[-1][1]*100,self.val_group_accs[-1][2]*100))
                self.logger.info('==  Test  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.test_group_accs[-1][0]*100,self.test_group_accs[-1][1]*100,self.test_group_accs[-1][2]*100))
                self.logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(results[0]*100,results[1]*100))
                self.logger.info('== Best Results: Epoch:{} Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(self.best_val_iter//self.train_per_step,self.best_val*100,self.best_val_test*100))
              
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.loss_init()             
                start_time = time.time()   
                self.operate_after_epoch()
                self.epoch+=1   
        
        # TFE warmup
        init_teacher, init_ema_teacher = self.classifier_warmup(
            copy.deepcopy(self.ema_model))
        self.teacher_head.weight.data.copy_(init_teacher.fc.fc.weight.data)
        self.teacher_head.bias.data.copy_(init_teacher.fc.fc.bias.data)
        self.ema_teacher_head.weight.data.copy_(init_ema_teacher.fc.fc.weight.data)
        self.ema_teacher_head.bias.data.copy_(init_ema_teacher.fc.fc.bias.data)
        self.build_crt_loader()
        # cossl
        self.logger.info('====================== cossl phase: ======================')
        
        self.iter=0
        self.epoch=0
        for self.iter in range(self.start_iter, self.max_iter):
            return_data=self.train_cossl_step()
            if return_data is not None:
                pred,gt=return_data[0],return_data[1]
                fusion_matrix.update(pred, gt) 
            if self.iter%self.train_per_step==0:  
                end_time = time.time()           
                time_second=(end_time - start_time)
                eta_seconds = time_second * (self.max_epoch - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
                self.train_group_accs.append(group_acc)
                results=self.evaluate()
                
                if self.best_val<results[0]:
                    self.best_val=results[0]
                    self.best_val_test=results[1]
                    self.best_val_iter=self.iter
                    self.save_checkpoint(file_name="best_model.pth")
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint()
                self.train_losses.append(self.losses.avg)
                self.logger.info("== CoSSL traning phase")
                self.logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f} '.\
                    format(self.losses.avg, self.losses_x.avg, self.losses_u.avg))
                self.logger.info('== val_losss:{:>5.4f}   test_loss:{:>5.4f}   epoch_Time:{:>5.2f}min eta:{}'.\
                        format(self.val_losses[-1], self.test_losses[-1],time_second / 60,eta_string))
                self.logger.info('== Train  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.train_group_accs[-1][0]*100,self.train_group_accs[-1][1]*100,self.train_group_accs[-1][2]*100))
                self.logger.info('==  Val   group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.val_group_accs[-1][0]*100,self.val_group_accs[-1][1]*100,self.val_group_accs[-1][2]*100))
                self.logger.info('==  Test  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.test_group_accs[-1][0]*100,self.test_group_accs[-1][1]*100,self.test_group_accs[-1][2]*100))
                self.logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(results[0]*100,results[1]*100))
                self.logger.info('== Best Results: Epoch:{} Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(self.best_val_iter//self.train_per_step,self.best_val*100,self.best_val_test*100))
              
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.loss_init()             
                start_time = time.time()   
                self.operate_after_epoch()
                self.epoch+=1   
        self.plot()       
        return
     
    
    def train_pretrain_step(self):
        self.model.train()
        self.ema_model.eval()
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
        gt_targets_u=data[1]
        
        batch_size = inputs_x.size(0)  
        # Transform label to one-hot
        targets_x2 = torch.zeros(batch_size, self.num_classes).scatter_(1, targets_x.view(-1,1), 1)
        
        inputs_x, targets_x2 = inputs_x.cuda(), targets_x2.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3  = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
        
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u = self.model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            max_p, p_hat = torch.max(targets_u, dim=1)
            select_mask = max_p.ge(self.conf_thres).float()

            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                self.used_c.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            self.mask_prob.update(select_mask.mean().item())
            self.total_c.update(total_acc.mean(0).item())

            p_hat = torch.zeros(p_hat.size(0), self.num_classes).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            select_mask = torch.cat([select_mask, select_mask], 0)

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x2, p_hat, p_hat], dim=0)

        all_outputs = self.model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = self.train_criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu 

        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_u.update(Lu.item(), inputs_x.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.ema_optimizer.step()
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.avg,self.losses_x.avg,self.losses_u.avg))
        return 

    
    def train_cossl_step(self):
        self.model.train()
        self.ema_model.eval()
        loss =0
        
        # Data loading
        try:
            inputs_x, targets_x, _ = self.labeled_train_iter.next()
        except:
            self.labeled_train_iter = iter(self.labeled_trainloader)
            inputs_x, targets_x, _ = self.labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter = iter(self.unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = self.unlabeled_train_iter.next()

        try:
            crt_input_x, crt_targets_x, _ = self.crt_labeled_iter.next()
        except:
            self.crt_labeled_iter = iter(self.crt_labeled_loader)
            crt_input_x, crt_targets_x, _ = self.crt_labeled_iter.next() #

        try:
            crt_input_u, crt_targets_u, _ = self.crt_full_iter.next()
        except:
            self.crt_full_iter = iter(self.crt_full_loader)
            crt_input_u, crt_targets_u, _ = self.crt_full_iter.next()

            
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, self.num_classes).scatter_(1, targets_x.view(-1, 1), 1)
        crt_targets_x = torch.zeros(crt_targets_x.size(0), self.num_classes).scatter_(1, crt_targets_x.view(-1, 1), 1)
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
        crt_input_x, crt_input_u, crt_targets_x = crt_input_x.cuda(), crt_input_u.cuda(), crt_targets_x.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by ema_model and ema_teacher
            feature_u = self.ema_model(inputs_u, return_encoding=True)
            outputs_u = self.teacher_head(feature_u.squeeze())

            targets_u = torch.softmax(outputs_u, dim=1)
            max_p, p_hat = torch.max(targets_u, dim=1)

            select_mask = max_p.ge(self.conf_thres).float()

            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                self.used_c.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            self.mask_prob.update(select_mask.mean().item())
            self.total_c.update(total_acc.mean(0).item())

            p_hat = torch.zeros(p_hat.size(0), self.num_classes).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            select_mask = torch.cat([select_mask, select_mask], 0)

        # Extract the features for classifier learning
        with torch.no_grad():
            crt_feat_x = self.ema_model(crt_input_x, return_encoding=True)
            crt_feat_x = crt_feat_x.squeeze()

            crt_feat_u = self.ema_model(crt_input_u, return_encoding=True)
            crt_feat_u = crt_feat_u.squeeze()

            new_feat_list = []
            new_target_list = []

            for x, label_x, u in zip(crt_feat_x, crt_targets_x, crt_feat_u[:len(crt_targets_x)]):
                if random.random() < self.tfe_prob[label_x.argmax()]:
                    lam = np.random.uniform(self.max_lam, 1., size=1)
                    lam = torch.FloatTensor(lam).cuda()

                    new_feat = lam * x + (1 - lam) * u
                    new_target = label_x
                else:
                    new_feat = x
                    new_target = label_x
                new_feat_list.append(new_feat)
                new_target_list.append(new_target)
            new_feat_tensor = torch.stack(new_feat_list, dim=0)  # [64, 128]
            new_target_tensor = torch.stack(new_target_list, dim=0)  # [64, 10]

        teacher_logits = self.teacher_head(new_feat_tensor)
        teacher_loss = -torch.mean(torch.sum(F.log_softmax(teacher_logits, dim=1) * new_target_tensor, dim=1))
        self.teacher_head_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_head_optimizer.step()
        self.ema_teacher_head_optimizer.step()

        with torch.no_grad():
            acc = (torch.argmax(teacher_logits, dim=1) == torch.argmax(crt_targets_x, dim=1)).float().mean()
            score_result = self.func(teacher_logits)
            now_result = torch.argmax(score_result, 1)
            self.losses_teacher.update(teacher_loss.item(), crt_targets_x.size(0))

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs = self.model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = self.train_criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0))
        self.losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema_optimizer.step()
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.avg,self.losses_x.avg,self.losses_u.avg))
        
        return 
    
    def classifier_warmup(self,model):
        # train_labeled_set, train_unlabeled_set, N_SAMPLES_PER_CLASS

        # Hypers used during warmup
        epochs = self.cfg.ALGORITHM.CoSSL.WARM_EPOCH_TFE  # 10
        lr = self.cfg.ALGORITHM.CoSSL.LR_TFE  # 0.002
        ema_decay = self.cfg.ALGORITHM.CoSSL.EMA_DECAY_TFE  # 0.999
        weight_decay = self.cfg.ALGORITHM.CoSSL.WD_TFE  # 5e-4
        batch_size = self.cfg.DATASET.BATCH_SIZE  # 64
        val_iteration = self.cfg.TRAIN_STEP  # 500

        # Construct dataloaders
    
        tfe_model = self.weight_imprint(copy.deepcopy(model))

        # Fix the feature extractor and reinitialize the classifier
        for param in model.parameters():
            param.requires_grad = False
        model.fc.fc.reset_parameters()
        for param in model.fc.fc.parameters():
            param.requires_grad = True

        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        ema_optimizer = WeightEMA(model, ema_model, lr, alpha=ema_decay, wd=False)

        wd_params, non_wd_params = [], []
        for name, param in model.fc.fc.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias, conv2d.bias
            else:
                wd_params.append(param)
        param_list = [{'params': wd_params, 'weight_decay': weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
        self.logger.info('    Total params: %.2fM' % (sum(p.numel() for p in model.fc.fc.parameters()) / 1000000.0))

        optimizer = get_param_optimizer(self.cfg, param_list)

        # Generate TFE features in advance as the model and the data loaders are fixed anyway
        balanced_feature_set = self.TFE(tfe_model)
        balanced_feature_loader = data.DataLoader(balanced_feature_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0, drop_last=True)

        # Main function
        for epoch in range(epochs):
            self.logger.info('==  cRT: Epoch: [%d | %d] LR: %f' % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))
            loss, train_acc=self.classifier_train(balanced_feature_loader, model, optimizer, None, ema_optimizer)
            self.logger.info('==       loss:{}, train_acc:{}'.format(loss,train_acc))
        return model, ema_model


    def TFE(self,tfe_model):

        tfe_model.eval()
        with torch.no_grad():
            # ****************** extract features  ********************
            # extract features from labeled data
            for batch_idx, (inputs, targets, _) in enumerate(self.labeled_trainloader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                features = tfe_model(inputs, return_encoding=True)
                logits=tfe_model(features,classifier=True)
                cls_probs = torch.softmax(logits, dim=1)
                features = features.squeeze()  # Note: a flatten is needed here
                if batch_idx == 0:
                    labeled_feature_stack = features
                    labeled_target_stack = targets
                    labeled_cls_prob_stack = cls_probs
                else:
                    labeled_feature_stack = torch.cat((labeled_feature_stack, features), 0)
                    labeled_target_stack = torch.cat((labeled_target_stack, targets), 0)
                    labeled_cls_prob_stack = torch.cat((labeled_cls_prob_stack, cls_probs), 0)
            # extract features from unlabeled data
            for batch_idx, (data_batch, _, _) in enumerate(self.unlabeled_trainloader):
                # if hasattr(unlabeled_loader.dataset.transform, 'transform2'):  # FixMatch, ReMixMatch
                inputs_w, inputs_s, _ = data_batch
                inputs_s = inputs_s.cuda()
                inputs_w = inputs_w.cuda()

                features = tfe_model(inputs_s, return_encoding=True)
                logits = tfe_model(inputs_w)
                # else:  # MixMatch
                #     inputs_w, _ = data_batch
                #     inputs_w = inputs_w.cuda()
                #     logits, _, features = tfe_model(inputs_w, return_feature=True)
                cls_probs = torch.softmax(logits, dim=1)
                _, targets = torch.max(cls_probs, dim=1)
                features = features.squeeze()
                if batch_idx == 0:
                    unlabeled_feature_stack = features
                    unlabeled_target_stack = targets
                    unlabeled_cls_prob_stack = cls_probs
                else:
                    unlabeled_feature_stack = torch.cat((unlabeled_feature_stack, features), 0)
                    unlabeled_target_stack = torch.cat((unlabeled_target_stack, targets), 0)
                    unlabeled_cls_prob_stack = torch.cat((unlabeled_cls_prob_stack, cls_probs), 0)

            # ****************** create TFE features for each class  ********************
            # create idx array for each class, per_cls_idx[i] contains all indices of images of class i
            labeled_set_idx = torch.tensor(list(range(len(labeled_feature_stack))))
            labeled_set_per_cls_idx = [labeled_set_idx[labeled_target_stack == i] for i in range(self.num_classes)]

            augment_features = []  # newly generated tfe features will be appended here
            augment_targets = []  # as well as their one-hot targets
            for cls_id in range(self.num_classes):
                if self.num_samples_per_class[cls_id] < max(self.num_samples_per_class):

                    # how many we need for the cls
                    augment_size = max(self.num_samples_per_class) - self.num_samples_per_class[cls_id]

                    # create data belonging to class i
                    current_cls_feats = labeled_feature_stack[labeled_target_stack == cls_id]

                    # create data not belonging to class i
                    other_labeled_data_idx = np.concatenate(labeled_set_per_cls_idx[:cls_id] + labeled_set_per_cls_idx[cls_id + 1:], axis=0)
                    other_cls_feats = torch.cat([labeled_feature_stack[other_labeled_data_idx], unlabeled_feature_stack], dim=0)
                    other_cls_probs = torch.cat([labeled_cls_prob_stack[other_labeled_data_idx], unlabeled_cls_prob_stack], dim=0)

                    assert len(other_cls_feats) == len(other_cls_probs)
                    # the total number of data should be the same for label-unlabel split, and current-the-rest split
                    assert (len(other_cls_feats) + len(current_cls_feats)) == (len(labeled_feature_stack) + len(unlabeled_feature_stack))

                    # sort other_cls_feats according to the probs assigned to class i
                    probs4current_cls = other_cls_probs[:, cls_id]
                    sorted_probs, order = probs4current_cls.sort(descending=True)  # sorted_probs = probs belonging to cls i
                    other_cls_feats = other_cls_feats[order]

                    # select features from the current class
                    input_a_idx = np.random.choice(list(range(len(current_cls_feats))), augment_size, replace=True)
                    # take first n features from all other classes
                    input_b_idx = np.asarray(list(range(augment_size)))
                    lambdas = np.random.beta(0.75, 0.75, size=augment_size)

                    # do TFE
                    for l, a_idx, b_idx in zip(lambdas, input_a_idx, input_b_idx):
                        tfe_input = l * current_cls_feats[a_idx] + (1 - l) * other_cls_feats[b_idx]  # [128]
                        tfe_target = torch.zeros((1, self.num_classes))
                        tfe_target[0, cls_id] = 1  # pseudo_label.tolist()
                        augment_features.append(tfe_input.view(1, -1))
                        augment_targets.append(tfe_target)

            # ****************** merge newly generated data with labeled dataset  ********************
            augment_features = torch.cat(augment_features, dim=0)
            augment_targets = torch.cat(augment_targets, dim=0).cuda()

            target_stack = torch.zeros(len(labeled_target_stack), self.num_classes).cuda().scatter_(1, labeled_target_stack.view(-1, 1), 1)
            new_feat_tensor = torch.cat([labeled_feature_stack, augment_features], dim=0)
            new_target_tensor = torch.cat([target_stack, augment_targets], dim=0)

        balanced_feature_set = data.dataset.TensorDataset(new_feat_tensor, new_target_tensor)
        return balanced_feature_set


    def weight_imprint(self,model):
        model = model.cuda()
        model.eval()

        # labeledloader = data.DataLoader(labeled_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

        with torch.no_grad():
            # bar = Bar('Classifier weight imprinting...', max=len(labeledloader))

            for batch_idx, (inputs, targets, _) in enumerate(self.labeled_trainloader):
                inputs = inputs.cuda()
                features = model(inputs, return_encoding=True)
                output = features.squeeze()   # Note: a flatten is needed here

                if batch_idx == 0:
                    output_stack = output
                    target_stack = targets
                else:
                    output_stack = torch.cat((output_stack, output), 0)
                    target_stack = torch.cat((target_stack, targets), 0)
 
        new_weight = torch.zeros(self.num_classes, model.fc.in_features)

        for i in range(self.num_classes):
            tmp = output_stack[target_stack == i].mean(0)
            new_weight[i] = tmp / tmp.norm(p=2)

        model.fc.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes, bias=False).cuda()
        model.fc.fc.weight.data = new_weight.cuda()

        model.eval()
        return model


    def classifier_train(self,labeled_trainloader, model, optimizer, scheduler, ema_optimizer):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        train_acc = AverageMeter()
        end = time.time()
  
        model.eval()
        for batch_idx in range(self.train_per_step):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()
 

            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)  # targets are one-hot

            outputs = model(inputs_x,classifier=True)

            loss = (-F.log_softmax(outputs, dim=1) * targets_x).sum(dim=1)
            loss = loss.mean()
            acc = (torch.argmax(outputs, dim=1) == torch.argmax(targets_x, dim=1)).float().sum() / len(targets_x)

            # Record loss and acc
            losses.update(loss.item(), inputs_x.size(0))
            train_acc.update(acc.item(), inputs_x.size(0))

            # Compute gradient and apply SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step() 
        return losses.avg, train_acc.avg  

    def evaluate(self,return_group_acc=False,return_class_acc=False):   
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_teacher(self.ema_model, self.ema_teacher_head,self.test_loader, self.val_criterion)
        if self.valset_enable:
            val_loss, val_acc,val_group_acc,val_class_acc = self.eval_teacher(self.ema_model, self.ema_teacher_head,self.val_loader, self.val_criterion) 
        else: 
            val_loss, val_acc,val_group_acc,val_class_acc=test_loss, test_acc ,test_group_acc,test_class_acc
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_group_accs.append(val_group_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.test_group_accs.append(test_group_acc)
        
        if return_group_acc:
            if return_class_acc:
                return val_acc,test_acc,test_group_acc,test_class_acc
            else:
                return val_acc,test_acc,test_group_acc
        if return_class_acc:
            return val_acc,test_acc,test_class_acc
        return [val_acc,test_acc]
    
    def eval_teacher(self, model, head, valloader, criterion):
        
        losses = AverageMeter() 
        
        fusion_matrix = FusionMatrix(self.num_classes)
        func = torch.nn.Softmax(dim=1)

        # switch to evaluate mode
        model.eval() 
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(valloader): 

                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                # compute output
                feats = model(inputs, return_encoding=True)
                outputs = head(feats.squeeze())
                loss = criterion(outputs, targets)
                score_result = func(outputs)
                now_result = torch.argmax(score_result, 1) 
                losses.update(loss.item(), inputs.size(0))
                fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
        
        group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
        class_acc=fusion_matrix.get_acc_per_class()
        acc=fusion_matrix.get_accuracy()    
        # GM = 1
        # for i in range(self.num_classes):
        #     if class_acc[i] == 0:
        #         # To prevent the N/A values, we set the minimum value as 0.001
        #         GM *= (1/(100 * self.num_classes)) ** (1/self.num_classes)
        #     else:
        #         GM *= (class_acc[i]) ** (1/self.num_classes) 
        
        return (losses.avg, acc, group_acc,class_acc)
 