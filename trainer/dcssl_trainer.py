
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
class DCSSLTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        # 需要先捕获偏差特征  
        
        self.lambda_d=cfg.ALGORITHM.DCSSL.LAMBDA_D 
        self.dcssl_contra_temperture=cfg.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE
        self.biased_fusion_matrix=FusionMatrix(self.num_classes)
        self.con_loss = DebiasSupConLoss(temperature= self.dcssl_contra_temperture)
        self.losses_d_ctr=AverageMeter()
        self.losses_bx = AverageMeter()
        self.losses_bu = AverageMeter() 
        # self.biased_contrastive_loss=UnsupBiasContrastiveLoss()  
        self.biased_model=self.build_model(cfg).cuda()
        self.gce_loss=GeneralizedCELoss()
        self.biased_optimizer=self.build_optimizer(cfg, self.biased_model)
        self.fp_k=cfg.ALGORITHM.DCSSL.FP_K
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_d_ctr=AverageMeter()
        self.losses_bx = AverageMeter()
        self.losses_bu = AverageMeter() 
        
    def train_step(self,pretraining=False): 
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
        self.train_biased_step(data_x,data_u) 
        return_data=self.train_debiased_step(data_x,data_u)
        return return_data
    
    def train_biased_step(self,data_x,data_u):
        self.biased_model.train()
        loss =0       
        # dataset
        inputs_x=data_x[0][0][0]
        targets_x=data_x[1]        
        inputs_u=data_u[0][0][0]
        inputs=torch.cat([inputs_x,inputs_u],dim=0)
        inputs=inputs.cuda()
        targets_x=targets_x.long().cuda() 
        # logits
        logits=self.biased_model(inputs) 
        logits_x=logits[:inputs_x.size(0)]        
        
        # 1. ce loss
        loss_cls=self.l_criterion(logits_x,targets_x)
        # loss_cls = self.gce_loss(logits_x, targets_x).mean()
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        # 2. cons loss
        u_weak=logits[inputs_x.size(0):]
        with torch.no_grad(): 
            p = u_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(self.conf_thres).float()  
        loss_cons = self.ul_criterion(
            u_weak, pred_class, weight=loss_weight, avg_factor=u_weak.size(0)
        )
        loss=loss_cls+loss_cons
        self.biased_optimizer.zero_grad()
        loss.backward()
        self.biased_optimizer.step()  
        self.losses_bx.update(loss_cls.item(), inputs_x.size(0))
        self.losses_bu.update(loss_cons.item(), inputs_u.size(0))  
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
        # if self.iter % 2==0:
            self.logger.info('== Biased Epoch:{} Step:[{}|{}]  Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                        self.train_per_step,self.losses_bx.avg,self.losses_bu.avg))
        return now_result.cpu().numpy(), targets_x.cpu().numpy()   
       
    def train_debiased_step(self,data_x,data_u):
        self.model.train()
        loss =0         
        
        inputs_x=data_x[0][0][0]
        inputs_x2=data_x[0][0][1]
        inputs_xi=data_x[0][1]
        targets_x=data_x[1]
        
        inputs_u=data_u[0][0][0]
        inputs_u2=data_u[0][0][1] 
        inputs_ui=data_u[0][1] 
         
        inputs=torch.cat([inputs_x,inputs_x2,inputs_u,inputs_u2],dim=0)
        inputs=inputs.cuda()
        targets_x=targets_x.long().cuda()
        
        encodings = self.model(inputs,return_encoding=True) 
        features=self.model(encodings,return_projected_feature=True)  
        # === biased_contrastive_loss 
        with torch.no_grad():
            # inputs_bias=torch.cat([inputs_xi,inputs_ui],dim=0).cuda()
            # biased_encodings=self.biased_model(inputs_bias,return_encoding=True)
            biased_encodings=self.biased_model(inputs,return_encoding=True)            
            logits_b=self.biased_model(biased_encodings,classifier=True)
            biased_encodings=biased_encodings.detach()
            # loss_b=self.l_criterion(logits_b[:inputs_xi.shape[0]],targets_x,reduction_override='none')
            # loss_b=loss_b.detach()
            
        logits=self.model(encodings,classifier=True)
         
        logits_x1,logits_x2=logits[:inputs_x.size(0)*2].chunk(2)
        # 1. ce loss
        logits_x=(logits_x1+logits_x2)*0.5
        
        # cur_ce_loss
        # loss_d = self.l_criterion(logits_x, targets_x,reduction_override='none')
        # loss_d=loss_d.detach()
        # loss_cls_weight = loss_b / (loss_b + loss_d + 1e-8)
        # debiased_loss
        loss_cls = self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
        
        # 2. cons loss
        u_weak,u_strong=logits[inputs_x.size(0)*2:].chunk(2)
        with torch.no_grad(): 
            p = u_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1) 
            _,  top_k_pred_label = p.topk(self.fp_k, dim=1) # [128,5]
            # 避免假阴
            
            
            
        loss_weight = self.get_u_loss_weight(confidence)
        loss_cons = self.ul_criterion(
            u_strong, pred_class, weight=loss_weight, avg_factor=u_strong.size(0)
        )
        
        # 3. ctr loss
        
        # debiased 
        l_feature_w,l_feature_s=features[:inputs_x.shape[0]*2].chunk(2)
        u_feature_w,u_feature_s=features[inputs_x.shape[0]*2:].chunk(2)
        contra_features= torch.cat([l_feature_w,u_feature_w,l_feature_s,u_feature_s])
        # top1
        sample_mask=torch.cat([torch.ones_like(targets_x).cuda(),loss_weight],dim=0)
        sample_mask=sample_mask.repeat(2)
        
        # 假阴过多？排除top5
        loss_d_ctr = self.dcssl_contra_loss(feat=contra_features,targets_x=targets_x,
                                            topk_pred_label=top_k_pred_label,
                                            biased_feat=biased_encodings,
                                            sample_mask=sample_mask,
                                            temperature=self.dcssl_contra_temperture) 
        
        loss=loss_cls+loss_cons+self.lambda_d*loss_d_ctr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.losses_d_ctr.update(loss_d_ctr.item(),pred_class.size(0)) 
        self.losses_x.update(loss_cls.item(), inputs_x.size(0))
        self.losses_u.update(loss_cons.item(), inputs_u.size(0)) 
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
    
    def dcssl_contra_loss(self,feat=None,targets_x=None,topk_pred_label=None,biased_feat=None,sample_mask=None,temperature=0.07):
        # out_1=torch.cat([feat[:targets_x.shape[0]],feat[2*targets_x.shape[0]:2*targets_x.shape[0]+topk_pred_label[0].shape[0]]],dim=0) 
        # out_2=torch.cat([feat[targets_x.shape[0]:2*targets_x.shape[0]],feat[-topk_pred_label[0].shape[0]:]],dim=0)
        
        # similarity  = pairwise_similarity(out_1,out_2,temperature=temperature) 
        # loss        = NT_xent(similarity) 
        
        
        topk_pred_label=topk_pred_label.T
        y=torch.cat([targets_x,targets_x,topk_pred_label[0],topk_pred_label[0]],dim=0)
      
        similarity_matrix = (1./temperature) * torch.mm(feat,feat.transpose(0,1)) 
        # get weight by biased feature
        cos_sim=torch.from_numpy(cosine_similarity(biased_feat.cpu().numpy())).cuda()
        pos_weight = 1 - cos_sim
        neg_weight= cos_sim        
        # yy=torch.cat([targets_x,pred_class,pred_class],dim=0)
        pos_mask=torch.eq(y.contiguous().view(-1, 1),y.contiguous().view(-1, 1).T) 
        neg_mask=~pos_mask
        # pos_weight 需要乘以一个mask，mask掉其他的负类的 通过top-k
        # fp_mask=copy.deepcopy(top1_mask)
        for i in range(1,topk_pred_label.size(0)):
            tmp_y=torch.cat([targets_x,targets_x,topk_pred_label[i],topk_pred_label[i]],dim=0)
            tmp_mask=torch.eq(tmp_y.contiguous().view(-1, 1),tmp_y.contiguous().view(-1, 1).T) 
            pos_mask |= tmp_mask   
            neg_mask &= ~tmp_mask   
                  
         
        pos_mask=pos_mask.float() 
        neg_mask=neg_mask.float()
        
        # f1, f2 = feat.chunk(2)
        # cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
         
        # 去掉top-k个可能假阴，但也会因为可能混淆的样本也被抹去了
        pos_weight=pos_weight*pos_mask+ neg_mask*neg_weight
        pos_weight*=(1-torch.eye(pos_weight.shape[0],pos_weight.shape[0]).cuda())
        
        # for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # exp 
        exp_logits = torch.exp(logits)*pos_weight
        
        # log exp 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # weight
        sum_mask = pos_mask.sum(1) 
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask
        
        
        loss =(-mean_log_prob_pos* sample_mask).mean()
        # con_loss =(con_loss * sample_mask).mean()
        return loss
        # pos_weight=pos_weight*mask
        # 
        # logits_mask=(1-torch.eye(pos_weight.shape[0],pos_weight.shape[0]).cuda())
        # pos_weight = pos_weight * logits_mask # mask self-self

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # sum_mask = pos_mask.sum(1)+neg_mask.sum(1)
        # sum_mask[sum_mask == 0] = 1
        # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / sum_mask
        
        # # mean_log_prob_pos = (mask * log_prob).sum(1) / (((1-mask) * log_prob).sum(1) + 1e-9)

        # # loss
        # loss = -mean_log_prob_pos #torch.Size([384])
        # loss = loss.mean()
        # return loss
        
       
        
        
        # N2  = len(similarity_matrix)
        # # N   = int(len(similarity_matrix) / 2)
        
        # mask=(pos_pair - torch.eye(N2,N2).cuda())
        # # Removing diagonal 
        # # 相似度
        # similarity_matrix_exp = torch.exp(similarity_matrix)
        
        # similarity_matrix_exp= similarity_matrix_exp.mul(pos_weight)+similarity_matrix_exp.mul(neg_weight)
        # similarity_matrix_exp = similarity_matrix_exp * mask
        # log_prob = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
        # # log_prob = logits - torch.log(similarity_matrix_exp.sum(1, keepdim=True) + 1e-9)
        
        # # compute mean of log-likelihood over positive
        # sum_mask= mask.sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(1) /(sum_mask+ 1e-9)
          
        # NT_xent_loss_total  = (1./float(final_index.shape[0])) * torch.sum(mean_log_prob_pos[final_index])

        # return NT_xent_loss_total
        
    def evaluate(self,return_group_acc=False,return_class_acc=False):  
        # eval_model=self.get_val_model() 
        b_model=self.biased_model
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(b_model,self.test_loader, self.val_criterion)
        
        if self.valset_enable:
            val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(b_model,self.val_loader, self.val_criterion)
        else:
            val_loss, val_acc,val_group_acc,val_class_acc = test_loss, test_acc ,test_group_acc,test_class_acc
        self.logger.info('==Biased Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(val_acc*100,test_acc*100))
                
        
        
        d_model=self.model
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(d_model,self.test_loader, self.val_criterion)
        
        if self.valset_enable:
            val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(d_model,self.val_loader, self.val_criterion)
        else:
            val_loss, val_acc,val_group_acc,val_class_acc = test_loss, test_acc ,test_group_acc,test_class_acc
        
            
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
    
    def get_pos_neg_weight(self,biased_encodings,target_x,pred):
        # 求个cosine相似度
        yy=torch.cat([target_x,pred,pred],dim=0)
        pos_mask=torch.eq(yy.contiguous().view(-1, 1),yy.contiguous().view(-1, 1).T).float()
        neg_mask=1-pos_mask
        # 每个正样本的加权
        # 越相似，权重越小
        cos_sim=torch.from_numpy(cosine_similarity(biased_encodings.cpu().numpy())).cuda()
        pos_weight = 1 - cos_sim
        pos_weight*=pos_mask
        neg_weight=neg_mask*cos_sim
        
        return pos_weight,neg_weight
    
    
    def get_u_loss_weight(self,confidence):
        # 根据logits阈值
        loss_weight=confidence.ge(self.conf_thres).float()  
        # 根据能量分数
        
        return loss_weight
    
    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'd_model': self.model.state_dict(),
                    'b_model': self.biased_model.state_dict(),
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'd_optimizer': self.optimizer.state_dict(), 
                    'b_optimizer': self.biased_optimizer.state_dict(),  
                },  os.path.join(self.model_dir, file_name))
        return    
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")
        if not os.path.exists(resume):
            self.logger.info(f"Can\'t resume form {resume}")
            
        state_dict = torch.load(resume) 
        self.model.load_state_dict(state_dict['d_model'])
        self.biased_model.load_state_dict(state_dict["b_model"]) 

        # load optimizer and scheduler 
        self.optimizer.load_state_dict(state_dict["d_optimizer"])  
        self.biased_optimizer.load_state_dict(state_dict["b_optimizer"])   
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.train_per_step)+1 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        )