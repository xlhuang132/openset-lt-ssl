
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn 
import torch.backends.cudnn as cudnn    
import numpy as np 
import pandas as pd
import models 
import time  
import os   
import datetime
import torch.nn.functional as F 
from utils.misc import AverageMeter  
from .base_trainer import BaseTrainer
from utils import FusionMatrix
from models.projector import  Projector
import math
from loss.contrastive_loss import *
from loss.debiased_soft_contra_loss import DebiasSoftConLoss
from models.feature_queue import FeatureQueue
class MOODTrainer(BaseTrainer):
    def __init__(self, cfg):  
              
        super().__init__(cfg)      
        
        # if 'SCL' in self.cfg.OUTPUT_DIR:
        #     self.build_data_loaders_for_dl_contra() 
        self.loss_contrast= DebiasSoftConLoss(temperature=0.007)
        
        self.cfg=cfg
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True) 
        self.feature_decay=0.9
        self.lambda_pap=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        # self.feature_loss_temperature=cfg.ALGORITHM.MOOD.FEATURE_LOSS_TEMPERATURE
        self.id_temp=cfg.ALGORITHM.MOOD.ID_TEMPERATURE
        self.ood_temp=cfg.ALGORITHM.MOOD.OOD_TEMPERATURE
        
        self.alpha = cfg.ALGORITHM.MOOD.MIXUP_ALPHA         
        self.ood_detect_fusion = FusionMatrix(2)   
        self.id_detect_fusion = FusionMatrix(2)         
        self.ablation_enable=cfg.ALGORITHM.ABLATION.ENABLE
        self.dual_branch_enable=cfg.ALGORITHM.ABLATION.MOOD.DUAL_BRANCH
        self.mixup_enable=cfg.ALGORITHM.ABLATION.MOOD.MIXUP
        self.ood_detection_enable=cfg.ALGORITHM.ABLATION.MOOD.OOD_DETECTION
        self.pap_loss_enable=cfg.ALGORITHM.ABLATION.MOOD.PAP_LOSS 
        self.pap_loss_weight=cfg.ALGORITHM.MOOD.PAP_LOSS_WEIGHT
        self.feature_loss_type=cfg.ALGORITHM.MOOD.FEATURE_LOSS_TYPE
        
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
         
        self.load_id_masks()
        
    def train_step(self,pretraining=False):
        return self.train_mood_step()
        
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_pap_id = AverageMeter()
        self.losses_pap_ood = AverageMeter() 
     
    def train_mood_step(self):
        self.model.train()
        loss_dict={}
        # DL  
        try: 
            inputs_x, targets_x,meta = self.labeled_train_iter.next()
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader) 
            inputs_x, targets_x,meta = self.labeled_train_iter.next() 
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # DU  
        try:
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            (inputs_u,inputs_u2),ul_y,u_index = self.unlabeled_train_iter.next()
            
        # OOD  
        if not self.ablation_enable or self.ablation_enable and self.ood_detection_enable :
            try:
                (inputs_ood,inputs_ood2),_,_ = self.unlabeled_ood_train_iter.next()
            except:
                self.unlabeled_ood_train_iter=iter(self.unlabeled_ood_trainloader)
                (inputs_ood,inputs_ood2),_,_  = self.unlabeled_ood_train_iter.next()
            
            inputs_ood,inputs_ood2=inputs_ood.cuda(),inputs_ood2.cuda()
            
            
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        ul_y=ul_y.cuda()            
        u_index=u_index.cuda()    
        
        # id_mask,ood_mask = torch.ones_like(ul_y).cuda(),torch.zeros_like(ul_y).cuda()         
        # id_mask=self.id_masks[u_index].detach() 
        # ood_mask=self.ood_masks[u_index].detach()  
       
        if not self.ablation_enable or self.ablation_enable and self.dual_branch_enable:            
                    
            inputs_dual_x, targets_dual_x=meta['dual_image'],meta['dual_label']  
            inputs_dual_x, targets_dual_x = inputs_dual_x.cuda(), targets_dual_x.cuda(non_blocking=True)        
            
            if not self.ablation_enable or self.ablation_enable and self.mixup_enable:
                # mixup use ood
                if not self.ablation_enable or self.ablation_enable and self.ood_detection_enable:
                    # if 'upper_bound' in self.cfg.OUTPUT_DIR:
                    #     id_mask=(ul_y>1).float()
                    #     ood_mask=1-id_mask
                    # else:
                    #     id_mask=self.id_masks[u_index].detach() 
                    #     ood_mask=self.ood_masks[u_index].detach()   
                    # ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
                    # inputs_dual_x=self.mix_up(inputs_dual_x, inputs_u[ood_index])
                     
                    inputs_dual_x=self.mix_up(inputs_dual_x, inputs_ood)
                else:                    
                    inputs_dual_x=self.mix_up(inputs_dual_x, inputs_u)   
            inputs_x=torch.cat([inputs_x,inputs_dual_x],0)
            targets_x=torch.cat([targets_x,targets_dual_x],0)
            
         # 1. cls loss
        l_encoding = self.model(inputs_x,return_encoding=True)  
        l_logits = self.model(l_encoding,classifier=True)    
        
        # 1. dl ce loss
        cls_loss = self.l_criterion(l_logits, targets_x)
        
        loss_dict.update({"loss_cls": cls_loss})
        # compute 1st branch accuracy
        score_result = self.func(l_logits)
        now_result = torch.argmax(score_result, 1)          

        # 2. cons loss
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_encoding=self.model(ul_images,return_encoding=True) 
        
        if not self.ablation_enable or self.ablation_enable and self.ood_detection_enable :
            ood_images=torch.cat([inputs_ood , inputs_ood2],0)
            ood_encoding=self.model(ood_images,return_encoding=True) 
        
        ul_logits = self.model(ul_encoding,classifier=True) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)  
            confidence, pred_class = torch.max(p, dim=1)

        loss_weight = confidence.ge(self.conf_thres).float()
        
        # loss_weight*=id_mask
        cons_loss = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
        
        # modify hxl 
        l_feature=self.model(l_encoding,return_projected_feature=True) 
        # l_feature = l_feature/(l_feature.norm(dim=1).view(l_feature.size(0),1) + 1e-8)
        with torch.no_grad():  
            self.queue.enqueue(l_feature, targets_x)
             
        # 3. pap loss
        if (not self.ablation_enable  or self.ablation_enable and self.pap_loss_enable ) and self.iter>self.warmup_iter:
            # modify hxl
            ul_feature=self.model(ul_encoding,return_projected_feature=True)   
            ul_feature_weak,ul_feature_strong=ul_feature.chunk(2)
            
            ood_feature=self.model(ood_encoding,return_projected_feature=True)   
            ood_feature_weak,ood_feature_strong=ood_feature.chunk(2)
            
            Lfeature=self.get_feature_loss(l_feature, ul_feature_weak, ul_feature_strong,ood_feature_weak,ood_feature_strong, targets_x, pred_class,loss_weight)
            feature_loss_dict={"{} loss".format(self.feature_loss_type):Lfeature} 
            loss_dict.update(feature_loss_dict)  
           
        loss = sum(loss_dict.values())
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(cls_loss.item(), inputs_x.size(0))
        self.losses_u.update(cons_loss.item(), inputs_u.size(0)) 
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_loss:{:>5.4f} Loss_x:{:>5.4f}  Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val,self.losses_x.avg,self.losses_u.val))
            self.logger.info('================= Loss_pap_id:{:>5.4f} Loss_pap_ood:{:>5.4f} ==============='.format(self.losses_pap_id.val,self.losses_pap_ood.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
    
    def get_feature_loss(self,l_feature, ul_feature_weak, ul_feature_strong,ood_feature_weak,ood_feature_strong, targets_x, pred_class,loss_weight):
        if self.feature_loss_type=='PAP':
            all_features=torch.cat((l_feature,ul_feature_weak),dim=0)
            all_target=torch.cat((targets_x,pred_class),dim=0)
            confidenced_id_mask= torch.cat([torch.ones(l_feature.size(0)).cuda(),loss_weight],dim=0).long()             
            Lidfeat=  self.pap_loss_weight*self.get_id_feature_pull_loss(all_features, all_target)
            self.losses_pap_id.update(Lidfeat.item(), all_target.size(0)) 
         
            Loodfeat=0.       
            Loodfeat=self.pap_loss_weight*self.get_ood_feature_push_loss(ood_feature_weak,ood_feature_strong) 
            if Loodfeat.item()<0:  
                self.logger.info("Loodfeat : {}".format(Loodfeat.item()))    
            self.losses_pap_ood.update(Loodfeat.item(), ood_feature_weak.size(0)) 
            return Lidfeat+Loodfeat
        elif self.feature_loss_type=='ICL':
            return self.get_ICL_feature_loss(ul_feature_weak,ul_feature_strong)
        elif self.feature_loss_type=='PCL':  
            return self.get_PCL_feature_loss(l_feature,ul_feature_weak,targets_x,pred_class,loss_weight)
        else:  
            return self.get_CCL_feature_loss(l_feature,ul_feature_weak,targets_x,pred_class,loss_weight)
    
    # def get_PCL_feature_loss(self,ul_feature_weak,ul_feature_strong):
    #     features = torch.cat([ul_feature_weak.unsqueeze(1), ul_feature_strong.unsqueeze(1)], dim=1)
    #     Licl=self.pap_loss_weight*self.loss_contrast(features)
    #     self.losses_pap_id.update(Licl.item(), ul_feature_weak.size(0)) 
    #     return Licl
    
    def get_ICL_feature_loss(self,ul_feature_weak,ul_feature_strong):
        features = torch.cat([ul_feature_weak.unsqueeze(1), ul_feature_strong.unsqueeze(1)], dim=1)
        Licl=self.pap_loss_weight*self.loss_contrast(features)
        self.losses_pap_id.update(Licl.item(), ul_feature_weak.size(0)) 
         
        return Licl
    
    def get_PCL_feature_loss(self,l_feature,ul_feature_weak,targets_x,pred_class,loss_weight):
        anchor_feature = self.queue.prototypes 
        all_features=torch.cat((l_feature,ul_feature_weak),dim=0)
        all_target=torch.cat((targets_x,pred_class),dim=0)
        confidenced_id_mask= torch.cat([torch.ones(l_feature.size(0)).cuda(),loss_weight],dim=0).long()             
        Lpcl =  self.get_id_feature_pull_loss(all_features, all_target,mode='None')
        Lpcl= self.pap_loss_weight*(confidenced_id_mask*Lpcl).mean()
        self.losses_pap_id.update(Lpcl.item(), all_target.size(0)) 
         
        return Lpcl
    
    def get_CCL_feature_loss(self,l_feature,ul_feature_weak,targets_x,pred_class,loss_weight):
        all_features=torch.cat((l_feature,ul_feature_weak),dim=0)
        all_target=torch.cat((targets_x,pred_class),dim=0)
        confidenced_id_weight= torch.cat([torch.ones(l_feature.size(0)).cuda(),loss_weight],dim=0).long()             
        # tile mask 
        mask=torch.eq(all_target.contiguous().view(-1, 1), \
             all_target.contiguous().view(-1, 1).T).float()
        
        confidenced_id_mask=torch.eq(confidenced_id_weight,confidenced_id_weight.contiguous().view(-1, 1).T).float()
        mask*=confidenced_id_mask  
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.T),
            self.id_temp)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() 
        
        mask = mask * (1-torch.eye((mask.shape[0])).cuda()) # mask self-self
        
         # compute log_prob
        exp_logits = torch.exp(logits) * (1-torch.eye((mask.shape[0])).cuda())
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        loss = -mean_log_prob_pos 
        Lccl = self.pap_loss_weight*(confidenced_id_weight*loss).mean()
        self.losses_pap_id.update(Lccl.item(), all_target.size(0)) 
        
        return Lccl
    
    def get_id_feature_pull_loss(self,feature,targets,mode='mean'):
        anchor_feature = self.queue.prototypes 
         # compute logits
        anchor_dot_contrast = torch.div(
            torch.einsum('ij,kj->ik',feature, anchor_feature),
            self.id_temp)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask 
        mask=torch.eq(targets.contiguous().view(-1, 1).cuda(), \
             torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
   
        # mask*=id_mask.expand(mask.size(1),-1).T 
        
        # compute log_prob
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask 
        if mode=='mean':
            loss = (-mean_log_prob_pos).mean()
        else:
            loss=-mean_log_prob_pos
        return loss 
        
        # prototypes = self.queue.prototypes.cuda()
        # outputs=torch.cat((feature,prototypes),dim=0)
        # B   = outputs.shape[0] 
        # outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
        # similarity_matrix = (1./self.feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:feature.size(0),feature.size(0):]
        # mask_same_c=torch.eq(\
        #     targets.contiguous().view(-1, 1).cuda(), \
        #     torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        # id_mask=id_mask.expand(mask_same_c.size(1),-1).T  
        # mask_same_c*=id_mask
        # similarity_matrix_exp = torch.exp(similarity_matrix)         
        # log_prob = similarity_matrix_exp - torch.log((torch.exp(similarity_matrix_exp) * (1 - mask_same_c)).sum(1, keepdim=True))
        # log_prob_pos = log_prob * mask_same_c  
        # loss = - log_prob_pos.sum() / mask_same_c.sum()       
        # return loss
    
    def get_ood_feature_push_loss(self,features_u,features_u2): 
        batch_size = features_u.shape[0]
        mask = torch.eye(batch_size).cuda() 
        features = torch.cat([features_u.unsqueeze(1), features_u2.unsqueeze(1)], dim=1)  
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 128,11,64
        anchor_feature=torch.cat([torch.cat([features_u2.unsqueeze(1),self.queue.prototypes.unsqueeze(0).repeat(features_u2.size(0),1,1)],dim=1),\
            torch.cat([features_u.unsqueeze(1),self.queue.prototypes.unsqueeze(0).repeat(features_u.size(0),1,1)],dim=1)],dim=0)
       
        # compute logits
        sim=torch.einsum('ijk,imk->im',contrast_feature.unsqueeze(1), anchor_feature)
        anchor_dot_contrast = torch.div( #torch.Size([256,1,64])  torch.Size([256, 11, 64])
            # torch.matmul(contrast_feature.unsqueeze(1), anchor_feature.T),
            sim,
            self.ood_temp)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        anchor_count=contrast_count
        # tile mask 
        mask = torch.cat([torch.ones(1),torch.zeros(self.num_classes)],dim=0)\
            .repeat(logits.size(0),1).cuda() 
         
        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1 
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        loss = -mean_log_prob_pos  
        loss = loss.view(anchor_count, batch_size)  
        loss = (loss).mean() 
        return loss
        
        # prototypes = self.queue.prototypes.cuda()
        # features=torch.cat([features_u,features_u2],0) 
        # all_features=torch.cat([features,prototypes],0)  
        # B   = all_features.shape[0]
        # outputs_norm = all_features/(all_features.norm(dim=1).view(B,1) + 1e-8)
        # similarity_matrix = (1./self.feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:features_u.size(0),features_u.size(0):]
        # mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),self.num_classes))],dim=1)
        # mask_same_c=mask_same_c.cuda()
        # ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
        # mask_same_c*=ood_mask
        # log_prob = similarity_matrix - torch.log((torch.exp(similarity_matrix) * (1 - mask_same_c)).sum(1, keepdim=True))
        # log_prob_pos = log_prob * mask_same_c 
        # loss = - log_prob_pos.sum() / mask_same_c.sum()   
        # return loss   
    
    def get_id_feature_dist_loss(self,features, targets,id_mask):         
        prototypes = self.queue.prototypes.cuda()
         
        pair_dist=-1  *torch.cdist(features,prototypes)  
        logits=torch.div(pair_dist, self.feature_loss_temperature)  
        mask_same_c=torch.eq(\
            targets.contiguous().view(-1, 1).cuda(), \
            torch.tensor([i for i in range(self.num_classes)]).contiguous().view(-1, 1).cuda().T).float()
        id_mask=id_mask.expand(mask_same_c.size(1),-1).T # torch.Size([10,192]) # old 
        mask_same_c*=id_mask
        log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
        loss = - log_prob_pos.sum() / mask_same_c.sum()   
        return loss       
    
    def get_ood_feature_dist_loss(self,features_u,features_u2,ood_mask):         
        features=torch.cat([features_u,features_u2],0)        
        prototypes = self.queue.prototypes.cuda()  # (K, D)
        all_features=torch.cat([features,prototypes],0) # [138,64]
        # all_features=torch.cat([features,self.dl_center],0) # [138,64]
        mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),self.num_classes))],dim=1)
        mask_same_c=torch.cat([mask_same_c,mask_same_c],dim=0).cuda()    # [128,138]       
        pair_dist=-1 *torch.cdist(features,all_features)   # [128,138]  
        logits=torch.div(pair_dist, self.feature_loss_temperature)    # [128,138] 
        ood_mask=torch.cat([ood_mask,ood_mask],0)
        ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
        mask_same_c*=ood_mask
        log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_c)).sum(1, keepdim=True))
        log_prob_pos = log_prob * mask_same_c 
        loss = - log_prob_pos.sum() / mask_same_c.sum()   
        # if loss.item()<0:
        #     print("ood loss < 0")
        return loss   
    
    def mix_up(self,l_images,ul_images):                 
        with torch.no_grad():     
            len_l=l_images.size(0)
            len_aux=ul_images.size(0)
            if len_aux==0: 
                return l_images
            elif len_aux>len_l:
                ul_images=ul_images[:len_l]
            elif len_aux<len_l:
                extend_num= math.ceil(len_l/len_aux)
                tmp=[ul_images]*extend_num
                ul_images=torch.cat(tmp,dim=0)[:len_l]

            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1.0 - lam)
            rand_idx = torch.randperm(l_images.size(0)) 
            mixed_images = lam * l_images + (1.0 - lam) * ul_images[rand_idx]
            return mixed_images  
               
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda()
        self.ema_model = EMAModel(
            self.model,
            self.cfg.MODEL.EMA_DECAY,
            self.cfg.MODEL.EMA_WEIGHT_DECAY,
        )            
  
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
                    "prototypes":self.queue.prototypes, 
                    'id_masks':self.id_masks,
                    'ood_masks':self.ood_masks,                   
                },  os.path.join(self.model_dir, file_name))
        return    
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")
        if not os.path.exists(resume):
            self.logger.info(f"Can\'t resume form {resume}")
            
        state_dict = torch.load(resume)
        model_dict=self.model.state_dict() 
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict["model"].items() if k in self.model.state_dict()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        # self.model.load_state_dict(state_dict["model"])

        # load ema model 
        try:
            self.ema_model.load_state_dict(state_dict["ema_model"])
        except:
            self.logger.warning('load ema failed!')

        # load optimizer and scheduler
        try:
            self.optimizer.load_state_dict(state_dict["optimizer"])  
        except:
            self.logger.warning('load optimizer failed!')
        
        self.id_masks=state_dict['id_masks']
        self.ood_masks=state_dict['ood_masks']
        # gt=torch.cat(torch.ones(self.l_num),self)
        # self.id_detect_fusion.update(self.id_masks,)
        # self.detect_ood()
        # id_pre,id_rec=self.id_detect_fusion.get_pre_per_class()[1],self.id_detect_fusion.get_rec_per_class()[1]
        # ood_pre,ood_rec=self.ood_detect_fusion.get_pre_per_class()[1],self.ood_detect_fusion.get_rec_per_class()[1]
        # tpr=self.id_detect_fusion.get_TPR()
        # tnr=self.ood_detect_fusion.get_TPR()
        # self.logger.info('=='*20)
        # self.logger.info('== load mask ==')
        # self.logger.info("== ood_prec:{:>5.3f} id_prec:{:>5.3f} ood_rec:{:>5.3f} id_rec:{:>5.3f}".\
        #     format(ood_pre*100,id_pre*100,ood_rec*100,id_rec*100))
        # self.logger.info("=== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(tpr*100,tnr*100))
        # self.logger.info('=='*20)
        
        finetune= not 'warmup' in self.cfg.RESUME
        # finetune=True
        if not finetune:
            self._rebuild_models() 
            self._rebuild_optimizer(self.model) 
            self.logger.info('== Rebuild model and optimizer ==')
        else:
            self.logger.info('== Finetune model ==')
            
        
        # except:
        #     self.logger.warning("the id_masks and ood_masks of resume file are none!")
  
        
        self.queue.prototypes=state_dict["prototypes"]
        # self.queue.bank=state_dict['bank']
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.train_per_step)+1 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )
    
    def knn_ood_detect(self,ul_weak):          
        with torch.no_grad():
            feat=self.all_features 
            sim_matrix = torch.mm(ul_weak, feat.t())
            sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1) #  
            d_y=self.all_domain_y[sim_indices]   
            count_idnn=torch.count_nonzero(d_y,dim=1)        
            ones=torch.ones_like(count_idnn).cuda()
            zeros=torch.zeros_like(count_idnn).cuda()
            id_mask = torch.where(count_idnn >= self.k//2,ones,zeros).long().cuda() 
            ood_mask = torch.where(count_idnn <= (self.k)//3,ones,zeros).long().cuda() 
            # ood_mask=1-id_mask            
            return id_mask,ood_mask 
    
    def get_dl_center(self,): 
        features=[]
        labels=[]
        with torch.no_grad():
            for  i, ((inputs,_), targets, _) in enumerate(self.dl_center_loader):
                inputs=inputs.cuda()
                targets=targets.long().cuda()
                feat=self.model(inputs,return_encoding=True)
                feat=self.model(feat,return_projected_feature=True)
                features.append(feat.detach())                 
                labels.append(targets)
            features=torch.cat(features,dim=0)
            labels=torch.cat(labels,dim=0)
            uniq_c = torch.unique(labels)
            for c in uniq_c:
                c = int(c)
                select_index = torch.nonzero(
                    labels == c, as_tuple=False).squeeze(1)
                embedding_temp = features[select_index]  
                mean = embedding_temp.mean(dim=0) 
                self.dl_center[c] = mean
        return 
     
    def update_center(self,features,labels):
        with torch.no_grad():
            if len(labels) > 0:
                uniq_c = torch.unique(labels)
                for c in uniq_c:
                    c = int(c)
                    select_index = torch.nonzero(
                        labels == c, as_tuple=False).squeeze(1)
                    embedding_temp = features[select_index]  
                    mean = embedding_temp.mean(dim=0)
                    var = embedding_temp.var(dim=0, unbiased=False)
                    n = embedding_temp.numel() / embedding_temp.size(1)
                    if n > 1:
                        var = var * n / (n - 1)
                    else:
                        var = var 
                if torch.count_nonzero(self.dl_center[c])>0:
                    self.dl_center[c] =(1 - self.center_decay_ratio)* mean    +  \
                        self.center_decay_ratio* self.dl_center[c]
                else:
                    self.dl_center[c] = mean
        return