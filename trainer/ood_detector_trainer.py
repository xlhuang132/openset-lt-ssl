
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

from models.feature_queue import FeatureQueue

class OODDetectorTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        self.temperture=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE  
        self.loss_contrast= DebiasSoftConLoss(temperature=self.temperture)
        
        self.id_pres,self.ood_pres,self.id_recs,self.ood_recs=[],[],[],[]
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME)
    
    
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter): 
            self.train_step()
            if self.iter%self.train_per_step==0:  
                self.train_losses.append(self.losses.avg)
                end_time = time.time()           
                time_second=(end_time - start_time)
                eta_seconds = time_second * (self.max_epoch - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.logger.info('== Avg_Train_loss:{:>5.4f}  epoch_Time:{:>5.2f}min eta:{}'.\
                    format(self.losses.avg,time_second / 60,eta_string))
                self.detect_ood()
                id_pre,id_rec=self.id_detect_fusion.get_pre_per_class()[1],self.id_detect_fusion.get_rec_per_class()[1]
                ood_pre,ood_rec=self.ood_detect_fusion.get_pre_per_class()[1],self.ood_detect_fusion.get_rec_per_class()[1]
                
                self.id_pres.append(id_pre)
                self.id_recs.append(id_rec)
                self.ood_pres.append(ood_pre)
                self.ood_recs.append(ood_rec)
                
                tpr=self.id_detect_fusion.get_TPR()
                tnr=self.ood_detect_fusion.get_TPR()
                
                self.logger.info("== id_prec:{:>5.3f} id_rec:{:>5.3f} ood_prec:{:>5.3f} ood_rec:{:>5.3f}".\
                    format(id_pre*100,id_rec*100,ood_pre*100,ood_rec*100))
                self.logger.info("== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(tpr*100,tnr*100))
                self.logger.info('=='*40) 
                
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint()
                
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.loss_init() 
                start_time = time.time()   
                self.epoch+=1   
                
        self.plot()       
        return
    
    
    def train_step(self,pretraining=False): 
        self.model.train()
        loss =0 
        # DL  
        try:
            (_,inputs_x,inputs_x2), _,_ = self.pre_train_iter.next()    
        except:            
            self.pre_train_iter=iter(self.pre_train_loader)            
            (_,inputs_x,inputs_x2),_,_ = self.pre_train_iter.next()   
            
        inputs=torch.cat([inputs_x,inputs_x2],dim=0).cuda()
        encoding = self.model(inputs,return_encoding=True)  
        features=self.model(encoding,return_projected_feature=True)   
        
        f_u_s1, f_u_s2 = features.chunk(2)     
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1) 
                 
        loss= self.loss_contrast(features)  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f}=='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.avg))
        return 
    
 
    def plot(self):
        plot_loss_over_epoch(self.train_losses,title="Train Average Loss",save_path=os.path.join(self.pic_dir,'train_loss.jpg'))
        plot_ood_detection_over_epoch([self.id_pres,self.id_recs,self.ood_pres,self.ood_recs],save_path=os.path.join(self.pic_dir,'ood_detector_performance.jpg'))