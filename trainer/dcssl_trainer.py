
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
from utils.plot import *
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
from utils import OODDetectFusionMatrix
from models.feature_queue import FeatureQueue
from loss.focal_loss import FocalLoss

class DCSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        
        self.lambda_d=cfg.ALGORITHM.DCSSL.LAMBDA_D 
        self.final_lambda_d=self.lambda_d 
        self.m=cfg.ALGORITHM.DCSSL.M 
        self.debiased_contra_temperture=cfg.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE        
        self.ood_detect_confusion_matrix=OODDetectFusionMatrix(self.num_classes)
        self.loss_contrast= DebiasSoftConLoss(temperature=self.debiased_contra_temperture)
        self.mixup_alpha=0.5
        self.sharpen_temp=0.5
        self.temp_proto=0.5
        self.warmup_epoch=self.cfg.ALGORITHM.DCSSL.WARMUP_EPOCH        
        self.id_pres,self.ood_pres,self.id_recs,self.ood_recs=[],[],[],[]        
        self.data_dist=self.labeled_trainloader.dataset.num_per_cls_list
        self.cls_prob=torch.tensor(self.data_dist/self.data_dist[0]).cuda() 
        self.class_thresh=0.5+self.cls_prob*(self.conf_thres-0.5)
        self.contrast_with_thresh=cfg.ALGORITHM.DCSSL.CONTRAST_THRESH
        
        self.means=torch.zeros(self.num_classes,self.feature_dim).cuda()
        self.covs=torch.zeros(self.num_classes,self.feature_dim).cuda() 
        self.loss_version=self.cfg.ALGORITHM.DCSSL.LOSS_VERSION
        self.logger.info('contrastive loss version {}'.format(self.loss_version))
        
        self.ablation_enable=cfg.ALGORITHM.ABLATION.ENABLE
        self.class_aware_thresh_enable= self.ablation_enable and cfg.ALGORITHM.ABLATION.DCSSL.CT or not self.ablation_enable
        self.sample_weight_enable= self.ablation_enable and cfg.ALGORITHM.ABLATION.DCSSL.SS or not self.ablation_enable
        self.sample_pair_weight_hp_enable= self.ablation_enable and cfg.ALGORITHM.ABLATION.DCSSL.SPS_HP or not self.ablation_enable
        self.sample_pair_weight_hn_enable= self.ablation_enable and cfg.ALGORITHM.ABLATION.DCSSL.SPS_HN or not self.ablation_enable
        self.contrasitive_loss_enable=cfg.ALGORITHM.DCSSL.CONTRASTIVE_LOSS_ENABLE
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
        if cfg.ALGORITHM.DCSSL.ID_MASK_PATH!='':
            self.load_id_masks(cfg.ALGORITHM.DCSSL.ID_MASK_PATH)
        
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_d_ctr=AverageMeter()
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter):
            self.pretraining= self.warmup_enable and self.iter<=self.warmup_iter 
            return_data=self.train_step(self.pretraining)
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
                    if self.epoch%100==0:
                        self.save_checkpoint(file_name="checkpoint_{}.pth".format(self.epoch))
                    else:
                        self.save_checkpoint()
                self.train_losses.append(self.losses.avg)
                self.logger.info("== Pretraining is enable:{}".format(self.pretraining))
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
            
        inputs_x_w=data_x[0]
        # inputs_x_w=data_x[0][0] 
        # inputs_x_s=data_x[0][1] 
        # inputs_x_s2=data_x[0][2] 
        targets_x=data_x[1]
        
        # DU   
        inputs_u_w=data_u[0][0]
        inputs_u_s=data_u[0][1]
        inputs_u_s1=data_u[0][2]
        u_index=data_u[2]
        u_index=u_index.long().cuda()
        
# D        if isinstance(inputs_x,list):
#             inputs_x=inputs_x[0]
        # inputs = torch.cat(
        #         [inputs_x_w,inputs_x_s,inputs_x_s2, inputs_u_w, inputs_u_s, inputs_u_s1],
        #         dim=0).cuda()
        inputs = torch.cat(
                [inputs_x_w,inputs_u_w, inputs_u_s, inputs_u_s1],
                dim=0).cuda()
        
        targets_x=targets_x.long().cuda()
        
        encoding = self.model(inputs,return_encoding=True)
        features=self.model(encoding,return_projected_feature=True)
        logits=self.model(encoding,classifier=True)
        batch_size=inputs_x_w.size(0)
        logits_x = logits[:batch_size]
        # logits_u_w, logits_u_s, _ = logits[3*batch_size:].chunk(3)
        # f_l_w, f_l_s1, f_l_s2 = features[:3*batch_size].chunk(3)
        # f_u_w, f_u_s1, f_u_s2 = features[3*batch_size:].chunk(3)
        logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
        f_l_w= features[:batch_size]
        f_u_w, f_u_s1, f_u_s2 = features[batch_size:].chunk(3)
        
        # 1. ce loss           
        loss_cls=self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        with torch.no_grad(): 
            self.update_mean_cov(f_l_w.clone().detach(), targets_x.clone().detach())
         
        # 2. cons loss 
        # filter out low confidence pseudo label by self.cfg.threshold 
        with torch.no_grad(): 
            probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, pred_class = torch.max(probs_u_w, dim=-1)          
        
        if self.class_aware_thresh_enable:
            loss_weight = max_probs.ge(self.class_thresh[pred_class]).float() 
        else:
            loss_weight = max_probs.ge(self.conf_thres).float()     
        # loss_weight*=self.id_masks[u_index] 
        # loss_weight=self.select_du_samples(max_probs, pred_class, class_thresh=True,update_thresh=[probs_u_w,pred_class])
        loss_cons = self.ul_criterion(
            logits_u_s, pred_class, weight=loss_weight, avg_factor=logits_u_s.size(0)
        )
        labels = pred_class 
        # 3. ctr loss 
        if self.contrasitive_loss_enable:
            contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
            features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)  
            conf_sample=loss_weight
            # labels = torch.cat([targets_x,pred_class],dim=0) 
            # conf_sample= torch.cat([torch.ones_like(targets_x).cuda(),loss_weight.clone().detach()],dim=0)
            # features =torch.cat([torch.cat([f_l_s1.unsqueeze(1), f_l_s2.unsqueeze(1)], dim=1),torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)],dim=0) 
            # max_probs=torch.cat([torch.ones_like(targets_x).cuda(),max_probs],dim=0)
            # 对于低置信度样本赋予了高权重
            if self.sample_weight_enable:
                sample_weight=conf_sample*(1-max_probs)                        
            else:
                sample_weight=conf_sample
            # 0. 实例对比学习
            if self.loss_version==0:
                loss_d_ctr = self.loss_contrast(features)   
                
            # 1. 高置信度正样本+pi*pj ccssl x
            elif self.loss_version==1:
            
                loss_d_ctr = self.loss_contrast(features, # projected_feature
                                                max_probs=max_probs, # confidence
                                                labels=labels, # pred_class
                                                reduction=None) #torch.Size([2, 128])
                loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
            
            
            # 2. 选择高置信度的样本对作为正对 + pi*pj x
            elif self.loss_version==2:
            
                with torch.no_grad():
                    select_matrix = self.contrast_left_out_p(max_probs)
                loss_d_ctr = self.loss_contrast(features,
                                                max_probs=max_probs,
                                                labels=labels,
                                                reduction=None,
                                                select_matrix=select_matrix)
                loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
            
            # 3. pos weight + neg weight x
            elif self.loss_version==3:
            
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
            
            # 5. 直接使用高维特征计算相似度，不用pi*pj
            elif self.loss_version==5:
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
            
            # 6. 直接使用高维特征计算相似度 +  使用所有样本加权qi
            elif self.loss_version==6:
                contrast_mask=max_probs
                with torch.no_grad(): 
                    cos_sim= 1 - cosine_similarity(encoding[inputs_x.size(0):inputs_x.size(0)+inputs_u_w.size(0)].detach().cpu().numpy())
                    mask = torch.from_numpy(cos_sim).cuda()
                loss_d_ctr = self.loss_contrast(features, 
                                                labels=labels,
                                                mask=mask,
                                                reduction=None)
                loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
            
            # 7. 直接使用低维特征计算相似度 +  使用所有样本加权qi
            elif self.loss_version==7:
                contrast_mask*=max_probs
                with torch.no_grad(): 
                    cos_sim= 1 - cosine_similarity(f_u_s1.detach().cpu().numpy())
                    mask = torch.from_numpy(cos_sim).cuda()
                loss_d_ctr = self.loss_contrast(features, 
                                                labels=labels,
                                                mask=mask,
                                                reduction=None)
                loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
            
            # 8. 单纯的有监督对比
            elif self.loss_version==8: 
                # loss_d_ctr = self.loss_contrast(features, 
                #                                 labels=labels, 
                #                                 reduction=None)
                # loss_d_ctr = (loss_d_ctr * contrast_mask).mean()
                if self.epoch>self.warmup_epoch:
                    loss_d_ctr = (conf_sample* self.get_prototype_contrast_loss(features, self.means.detach(),labels)).mean()
                else:
                    loss_d_ctr=torch.tensor(0.).cuda()
            
            # 9.  cos_sim
            # 10. id: 1-max_probs ood: cos_sim
            # 11. 2-max_probs
            # 12. sample_weight: loss_weight*(1-max_probs)
            # elif 13>=self.loss_version>=9:
            #     # contrast_mask=max_probs.ge(self.contrast_with_thresh).float()
            #     if self.epoch>self.warmup_epoch:
            #         # v9: score
            #         # v10: 2-score
            #         if self.loss_version==9:
            #             sample_weight=self.get_sim_with_prototypes(f_u_s1, pred_class)
            #             id_mask=torch.eye(sample_weight).cuda()
            #         elif self.loss_version==10:
            #             with torch.no_grad():
            #                 sample_weight= 1-self.get_sim_with_prototypes(f_u_s1.detach(), pred_class)
            #             #     id_mask= self.id_masks[u_index]
            #             #     ood_mask=1-id_mask
            #             #     sample_weight= id_mask*(2-max_probs)+ood_mask*sample_weight
            #             # sample_weight=2-max_probs
            #         elif self.loss_version==11:
            #             sample_weight=2-max_probs
            #             id_mask=torch.ones_like(sample_weight).cuda()
            elif self.loss_version==12:
                if self.epoch>self.warmup_epoch: 
                    with torch.no_grad():  
                        f=encoding[batch_size:batch_size+inputs_u_w.size(0)].detach()
                        cos_sim= cosine_similarity(f.detach().cpu().numpy()) 
                        cos_sim = torch.from_numpy(cos_sim).cuda()                    
                        y = labels.contiguous().view(-1, 1)
                        labeled_mask= torch.eq(y, y.T).float() 
                        if self.sample_pair_weight_hp_enable and self.sample_pair_weight_hn_enable:
                            # 强调高置信度的正样本对的硬正分数
                            mask=(1-cos_sim)*labeled_mask + cos_sim*(1-labeled_mask)
                        elif self.sample_pair_weight_hn_enable: # 在有监督基础上加上负样本
                            mask = cos_sim*(1-labeled_mask)+labeled_mask
                        elif self.sample_pair_weight_hp_enable:
                            mask = (1-cos_sim)*labeled_mask
                        else: 
                            mask=labeled_mask  
                        # mask[mask<0.3]=0
                    loss_d_ctr = self.loss_contrast(features,  
                                                    mask=mask,                                                  
                                                    reduction=None)
                    loss_d_ctr = (loss_d_ctr * sample_weight).mean()  
                else:loss_d_ctr=torch.tensor(0.).cuda() 
            
            elif self.loss_version==13: 
                # 对高置信度的样本使用硬负硬正+1，其他就用1    
                if self.epoch>self.warmup_epoch:
                    sample_weight=1-max_probs
                    with torch.no_grad(): 
                        # f=torch.cat([f_l_w,f_u_w],dim=0)
                        # f=f_u_w
                        f=encoding[batch_size:batch_size+inputs_u_w.size(0)].detach()
                        cos_sim= cosine_similarity(f.cpu().numpy()) 
                        cos_sim = torch.from_numpy(cos_sim).cuda()
                        
                        conf_sample=conf_sample.contiguous().view(-1, 1)
                        conf_mask= torch.eq(conf_sample, conf_sample.T).float() 
                        
                        y = labels.contiguous().view(-1, 1)
                        labeled_mask= torch.eq(y, y.T).float() 
                        
                        pos_mask = labeled_mask
                        neg_mask = 1-labeled_mask  
                        if self.sample_pair_weight_hp_enable:
                            # 强调高置信度的正样本对的硬正分数 假阳？
                            pos_mask=(2-cos_sim)*labeled_mask*conf_mask 
                        if self.sample_pair_weight_hn_enable:
                            # 强调高置信度的负样本对的硬负分数
                            probs=probs_u_w
                            neg_mask=self.conduct_negative_pair(probs, labels) 
                            neg_mask=(conf_mask*(1-labeled_mask)+(1-conf_mask)*neg_mask)*(1+cos_sim)               
                                        
                    loss_d_ctr = self.loss_contrast(features,  
                                                    pos_mask=pos_mask,
                                                    neg_mask=neg_mask, 
                                                    labels=labels,                                                   
                                                    reduction=None)            
                    loss_d_ctr = (loss_d_ctr * sample_weight).mean()  
                else:
                    loss_d_ctr=torch.tensor(0.).cuda()
                
                    
                    # pos_mask = labeled_mask
                    # neg_mask = 1-labeled_mask  
                    # if self.sample_pair_weight_hp_enable:
                    #     # 强调高置信度的正样本对的硬正分数
                    #     pos_mask=(1-cos_sim)*labeled_mask*conf_mask+labeled_mask
                    #     # pos_mask=(1-cos_sim)*labeled_mask+labeled_mask
                    # if self.sample_pair_weight_hn_enable:
                    #     # 强调高置信度的负样本对的硬负分数
                    #     # bin_labels=F.one_hot(targets_x, num_classes=self.num_classes).float().cuda()
                    #     # probs=torch.cat([bin_labels,probs_u_w],dim=0) 
                    #     # probs=probs_u_w
                    #     # neg_mask=self.conduct_negative_pair(probs, labels)
                    #     # neg_mask=cos_sim*(1-labeled_mask)*neg_mask+(1-labeled_mask)
                    #     # neg_mask=cos_sim*(1-labeled_mask)+(1-labeled_mask)
                    #     # neg_mask=cos_sim*(1-labeled_mask)*conf_mask+(1-labeled_mask)
                    # # 低置信度样本 ： 类对比损失没有任何改变 对比负类的 
                    # # 找出对那些类是负的 
            elif self.loss_version==14: # 对高置信度的样本使用硬负硬正+1，其他就用1 用硬正加权和取平均
                    
                with torch.no_grad(): 
                    # f=f_u_w
                    f=encoding[batch_size:batch_size+inputs_u_w.size(0)].detach()
                    cos_sim= cosine_similarity(f.detach().cpu().numpy()) 
                    cos_sim = torch.from_numpy(cos_sim).cuda()
                    
                    conf_sample=conf_sample.contiguous().view(-1, 1)
                    conf_mask= torch.eq(conf_sample, conf_sample.T).float() 
                    
                    y = labels.contiguous().view(-1, 1)
                    labeled_mask= torch.eq(y, y.T).float() 
                    
                    pos_mask = labeled_mask
                    neg_mask = 1-labeled_mask  
                    if self.sample_pair_weight_hp_enable:
                        # 强调高置信度的正样本对的硬正分数
                        pos_mask=(1-cos_sim)*labeled_mask*conf_mask+labeled_mask
                    if self.sample_pair_weight_hn_enable:
                        # 强调高置信度的负样本对的硬负分数
                        probs=probs_u_w
                        neg_mask=self.conduct_negative_pair(probs, labels)
                        neg_mask=cos_sim*(1-labeled_mask)*neg_mask+(1-labeled_mask)
                    # 低置信度样本 ： 类对比损失没有任何改变 对比负类的 
                loss_d_ctr = self.loss_contrast(features,  
                                                pos_mask=pos_mask,
                                                neg_mask=neg_mask, 
                                                labels=labels,                                                   
                                                reduction=None)
                loss_d_ctr = (loss_d_ctr * sample_weight).mean() 
                
            # elif self.loss_version==14: #高置信度拉向类中心。低置信度根据cosine相似度拉
            #     if self.epoch>self.warmup_epoch:
            #         select_idx1 = torch.nonzero(conf_sample>0, as_tuple=False).squeeze(1)
            #         select_idx2 = torch.nonzero(conf_sample<=0, as_tuple=False).squeeze(1)
            #         loss_p=self.get_prototype_contrast_loss(features[:][select_idx1][:], self.means.detach(),labels[select_idx1])
            
            #         # if self.sample_weight_enable:
            #         sample_weight=conf_sample*(1-torch.cat([torch.ones_like(targets_x).cuda(),max_probs],dim=0))
            #         # else:
            #         #     sample_weight=conf_sample
                    
            #         with torch.no_grad(): 
            #             f=torch.cat([f_l_w,f_u_w],dim=0)
            #             cos_sim= cosine_similarity(f.detach().cpu().numpy())                        
            #             cos_sim = torch.from_numpy(cos_sim).cuda()
                                        
            #         loss_c=(sample_weight*self.get_local_contrast_loss(features,labels,mask=cos_sim)).mean()
            #     else:
            #         loss_p=torch.tensor(0.).cuda()
            #         loss_c=torch.tensor(0.).cuda()
            #     loss_d_ctr = loss_p + 0.2*loss_c
            
            # elif self.loss_version==15:
            #     # if self.sample_weight_enable:
            #     #     max_probs=torch.cat([torch.ones_like(targets_x).cuda(),max_probs],dim=0)
            #     #     sample_weight=conf_sample*(1.2-max_probs) 
            #     # else:
            #     #    sample_weight=conf_sample
            #     with torch.no_grad(): 
            #         f=torch.cat([f_l_w,f_u_w],dim=0)
            #         cos_sim= cosine_similarity(f.detach().cpu().numpy()) 
            #         cos_sim = torch.from_numpy(cos_sim).cuda()
                    
            #         conf_sample=conf_sample.contiguous().view(-1, 1)
            #         conf_mask= torch.eq(conf_sample, conf_sample.T).float() 
                    
            #         y = labels.contiguous().view(-1, 1)
            #         labeled_mask= torch.eq(y, y.T).float()                
                    
            #         pos_mask=(2-cos_sim)*labeled_mask+ cos_sim*(1-labeled_mask) 
            #         neg_mask = torch.zeros_like(pos_mask).cuda()
            #     loss_d_ctr = self.loss_contrast(features,  
            #                                     pos_mask=pos_mask,
            #                                     neg_mask=neg_mask, 
            #                                     labels=labels,                                                   
            #                                     reduction='mean')
        else:
            loss_d_ctr=loss_d_ctr=torch.tensor(0.).cuda() 
        loss=loss_cls+loss_cons+self.lambda_d*loss_d_ctr
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        self.losses_d_ctr.update(loss_d_ctr.item(),labels.shape[0]) 
        self.losses_x.update(loss_cls.item(), batch_size)
        self.losses_u.update(loss_cons.item(), inputs_u_s.size(0)) 
        self.losses.update(loss.item(),batch_size)
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_c:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,
                        self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_d_ctr.avg))
             
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    def get_prototype_contrast_loss(self,features,prototypes,labels):
      
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #torch.Size([20480])= torch.Size([320, 64])
        class_mask= torch.eq(\
            labels.contiguous().view(-1, 1).cuda(), \
            torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        # compute logits  0.5:temperature
        anchor_dot_contrast = torch.div(
        torch.matmul(contrast_feature, prototypes.T),
        self.temp_proto)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        anchor_count = features.shape[1]
        contrast_count = 1
        class_mask = class_mask.repeat(anchor_count, contrast_count)
        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (class_mask * log_prob).sum(1) / (log_prob).sum(1)
        loss = -mean_log_prob_pos 
        return loss.mean()
    
    def get_local_contrast_loss(self,features,labels,mask=None):
        # mask = torch.eq(labels, labels.T).float().cuda() 
        
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #torch.Size([20480])= torch.Size([320, 64])
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.debiased_contra_temperture)
        
        
        mask = mask.repeat(anchor_count, contrast_count)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # mask-out self-contrast cases 1-torch.eye
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask # mask self-self
        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1        
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        loss = -mean_log_prob_pos 
        
        loss = loss.view(anchor_count, batch_size) #torch.Size([2, 128])
        return loss
    
    def conduct_negative_pair(self, confidence,labels):
        neg_mask=torch.zeros(labels.shape[0],labels.shape[0]).cuda() 
        hard_neg_class_mask=(confidence<0.3).float()
        class_mask=torch.eq(\
            labels.contiguous().view(-1, 1).cuda(), \
            torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        neg_mask = torch.mm(hard_neg_class_mask,class_mask.T).cuda()   
        return neg_mask
    
    def load_id_masks(self,resume):
        self.logger.info(f"Resuming id_masks from: {resume}")
        self.logger.info(' ')
        state_dict = torch.load(resume) 
        self.id_masks=state_dict['id_masks']
        self.ood_masks=state_dict['ood_masks']   
        du_gt=torch.cat([torch.ones(self.ul_num-self.ul_ood_num),torch.zeros(self.ul_ood_num)],dim=0)
        
        
        self.id_detect_fusion.update(self.id_masks,du_gt)
        self.ood_detect_fusion.update(self.ood_masks,1-du_gt)        
          
        id_pre,id_rec=self.id_detect_fusion.get_pre_per_class()[1],self.id_detect_fusion.get_rec_per_class()[1]
        ood_pre,ood_rec=self.ood_detect_fusion.get_pre_per_class()[1],self.ood_detect_fusion.get_rec_per_class()[1]
        tpr=self.id_detect_fusion.get_TPR()
        tnr=self.ood_detect_fusion.get_TPR()        
        self.logger.info("== OOD_Pre:{:>5.3f} ID_Pre:{:>5.3f} OOD_Rec:{:>5.3f} ID_Rec:{:>5.3f}".\
            format(ood_pre*100,id_pre*100,ood_rec*100,id_rec*100))
        self.logger.info("=== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(tpr*100,tnr*100))
        
        with torch.no_grad():
            for  i, (inputs, targets, idx) in enumerate(self.test_unlabeled_trainloader):
                 self.ood_detect_confusion_matrix.update(self.id_masks[idx], targets)

        tprs=self.ood_detect_confusion_matrix.get_TPR_per_class()
        fnrs=self.ood_detect_confusion_matrix.get_FNR_per_class()
        self.ood_detect_confusion_matrix.plot_ood_detect_confusion_bar( tprs=tprs,fnrs=fnrs,labels=[i+1 for i in range(self.num_classes)],save_path=os.path.join(self.pic_dir,'ood_detection_bar.jpg'))
        
        self.logger.info('=== Class TPRS:{}'.format(tprs))
        self.logger.info('=== Class TNRS:{}'.format(fnrs))
        
      
        self.logger.info("Successfully loaded the id_mask.") 
        return
    
    def get_sim_with_prototypes(self,features,pred_class):
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
    
    def select_du_samples(self,confidence, pred_class, fixed_thresh=None,update_thresh=None,class_thresh=False):
        assert fixed_thresh is not None or class_thresh and update_thresh is not None
        if not class_thresh:
            return confidence.ge(fixed_thresh).float()
        else:
            # update_thresh[0]: probability update_thresh[1]:target
            assert update_thresh and len(update_thresh)==2
            self.update_class_thresh(update_thresh[0], update_thresh[1])
            cur_thresh=self.class_thresh[pred_class]
            loss_weight=(confidence > cur_thresh).float()
            return loss_weight
    
    @torch.no_grad()
    def update_class_thresh(self,probs,pred_class):
        for c in range(self.num_classes):
            select_idx=torch.nonzero( pred_class==c , as_tuple=False).squeeze(1)
            if select_idx.shape[0]!=0:
                tmp=probs[select_idx][:,c]
                select_idx=torch.nonzero( tmp>=self.class_thresh[c] , as_tuple=False).squeeze(1)
                if select_idx.shape[0]>0:
                    conf=tmp[select_idx].mean()
                    self.class_thresh[c]= self.m*self.class_thresh[c]+ (1-self.m)*conf
      
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
   
    def plot(self):
        plot_group_acc_over_epoch(group_acc=self.train_group_accs,title="Train Group Average Accuracy",save_path=os.path.join(self.pic_dir,'train_group_acc.jpg'))
        plot_group_acc_over_epoch(group_acc=self.val_group_accs,title="Val Group Average Accuracy",save_path=os.path.join(self.pic_dir,'val_group_acc.jpg'))
        plot_group_acc_over_epoch(group_acc=self.test_group_accs,title="Test Group Average Accuracy",save_path=os.path.join(self.pic_dir,'test_group_acc.jpg'))
        plot_acc_over_epoch(self.train_accs,title="Train average accuracy",save_path=os.path.join(self.pic_dir,'train_acc.jpg'),)
        plot_acc_over_epoch(self.test_accs,title="Test average accuracy",save_path=os.path.join(self.pic_dir,'test_acc.jpg'),)
        plot_acc_over_epoch(self.val_accs,title="Val average accuracy",save_path=os.path.join(self.pic_dir,'val_acc.jpg'),)
        plot_loss_over_epoch(self.train_losses,title="Train Average Loss",save_path=os.path.join(self.pic_dir,'train_loss.jpg'))
        plot_loss_over_epoch(self.val_losses,title="Val Average Loss",save_path=os.path.join(self.pic_dir,'val_loss.jpg'))
        plot_loss_over_epoch(self.test_losses,title="Test Average Loss",save_path=os.path.join(self.pic_dir,'test_loss.jpg'))

        plot_ood_detection_over_epoch([self.id_pres,self.id_recs,self.ood_pres,self.ood_recs],save_path=os.path.join(self.pic_dir,'ood_detector_performance.jpg'))