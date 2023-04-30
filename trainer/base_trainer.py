   
import logging
from operator import mod
from tkinter import W
import torch 
import numpy as np
from dataset.build_dataloader import *
from loss.build_loss import build_loss 
import models 
# import sklearn
import time 
import torch.optim as optim
from models.feature_queue import FeatureQueue
import os   
import datetime
# import faiss
from utils.utils import *
import torch.nn.functional as F
from utils.plot import plot_pr
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
from utils.build_optimizer import get_optimizer, get_scheduler
from utils.utils import cal_metric,print_results
from dataset.build_dataloader import _build_loader
from dataset.base import BaseNumpyDataset
from utils import FusionMatrix
from utils.ema_model import EMAModel

from loss.contrastive_loss import * 
from models.projector import  Projector 

def cosine_similarity(x1, x2, eps=1e-12):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


class BaseTrainer():
    def __init__(self,cfg):
        self.local_rank=cfg.LOCAL_RANK
        self.cfg=cfg
        self.logger, _ = self.create_logger(cfg)        
        self.path,self.model_dir,self.pic_dir =self.prepare_output_path(cfg,self.logger)
        self.num_classes=cfg.DATASET.NUM_CLASSES
        self.batch_size=cfg.DATASET.BATCH_SIZE
        # =================== build model =============
        self.ema_enable=False
        self.model = models.__dict__[cfg.MODEL.NAME](cfg=cfg)
        self.ema_model = EMAModel(
            self.model,
            cfg.MODEL.EMA_DECAY,
            cfg.MODEL.EMA_WEIGHT_DECAY, 
        )
        self.model=self.model.cuda()
        # self.ema_model=self.ema_model.cuda()
        # =================== build dataloader =============
        self.ul_test_loader=None
        self.build_data_loaders()
        # =================== build criterion ==============
        self.build_loss()
        # ========== build optimizer ===========         
        self.optimizer = get_optimizer(cfg, self.model)
        
        # ========== build dataloader ==========     
        
        self.max_epoch=cfg.MAX_EPOCH 
        self.train_per_step=cfg.TRAIN_STEP       
        # self.val_step=cfg.VAL_ITERATION
        self.max_iter=self.max_epoch*self.train_per_step+1
        self.func = torch.nn.Softmax(dim=1) 
        
        # ========== accuracy history =========
        self.test_accs=[]
        self.val_accs=[]
        self.train_accs=[]
        self.test_group_accs=[]
        self.val_group_accs=[]
        self.train_group_accs=[]
        self.valset_enable=cfg.DATASET.NUM_VALID!=0
        # ========== loss history =========
        self.train_losses=[]
        self.val_losses=[]
        self.test_losses=[]
        
        self.conf_thres=cfg.ALGORITHM.CONFIDENCE_THRESHOLD   
        
        self.iter=0
        self.best_val=0
        self.best_val_iter=0
        self.best_val_test=0
        self.start_iter=1
        self.epoch=1
        self.save_epoch=cfg.SAVE_EPOCH
        
        # === pretrain ===
        self.pretraining=False  
        self.warmup_enable=cfg.ALGORITHM.PRE_TRAIN.ENABLE
        self.l_num=len(self.labeled_trainloader.dataset)
        self.ul_num=len(self.unlabeled_trainloader.dataset)  
        self.id_masks=torch.ones(self.ul_num).cuda()
        self.ood_masks=torch.zeros(self.ul_num).cuda() 
        self.warmup_temperature=self.cfg.ALGORITHM.PRE_TRAIN.SimCLR.TEMPERATURE
        self.warmup_iter=cfg.ALGORITHM.PRE_TRAIN.WARMUP_EPOCH*self.train_per_step 
        self.feature_dim=64 if self.cfg.MODEL.NAME in ['WRN_28_2','WRN_28_8','Resnet50'] else 128 
        self.k=cfg.ALGORITHM.OOD_DETECTOR.K    
        self.id_thresh_percent=cfg.ALGORITHM.OOD_DETECTOR.THRESH_PERCENT
        self.ood_detect_fusion = FusionMatrix(2)   
        self.id_detect_fusion = FusionMatrix(2)  
        # self.update_domain_y_iter=cfg.ALGORITHM.OOD_DETECTOR.DOMAIN_Y_UPDATE_ITER
        # self.ood_threshold=cfg.ALGORITHM.OOD_DETECTOR.OOD_THRESHOLD
        # self.id_threshold=cfg.ALGORITHM.OOD_DETECTOR.ID_THRESHOLD
        self.rebuild_unlabeled_dataset_enable=False        
        self.opearte_before_resume()
        if cfg.DATASET.NAME!='semi-iNat':
            self.ul_ood_num=self.unlabeled_trainloader.dataset.ood_num  
            
            l_dataset = self.labeled_trainloader.dataset 
            l_data_np,l_transform = l_dataset.select_dataset(return_transforms=True)
            new_l_dataset = BaseNumpyDataset(l_data_np, transforms=l_transform,num_classes=self.num_classes)
            self.test_labeled_trainloader = _build_loader(self.cfg, new_l_dataset,is_train=False)
            
            ul_dataset = self.unlabeled_trainloader.dataset 
            ul_data_np,ul_transform = ul_dataset.select_dataset(return_transforms=True)
            new_ul_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_transform,num_classes=self.num_classes)
            self.test_unlabeled_trainloader = _build_loader(self.cfg, new_ul_dataset,is_train=False)
        
            
    def prepare_output_path(self,cfg,logger):
        return prepare_output_path(cfg,logger)
    
    def create_logger(self,cfg) :
        return create_logger(cfg) 
    
    def opearte_before_resume(self):
        pass     
    @classmethod
    def build_model(cls, cfg)  :
        model = models.__dict__[cfg.MODEL.NAME](cfg)
        return model
    
    @classmethod
    def build_optimizer(cls, cfg , model )  :
        return get_optimizer(cfg, model)
    
    @classmethod
    def build_scheduler(cls, cfg , optimizer )  :
        return get_scheduler(cfg, optimizer)
    
    def build_loss(self):
        self.l_criterion,self.ul_criterion,self.val_criterion = build_loss(self.cfg)
        return 
    
    def build_data_loaders(self,)  :
        # l_loader, ul_loader, val_loader, test_loader           
        # self.labeled_trainloader, self.unlabeled_trainloader, self.val_loader, self.test_loader = build_data_loaders(cfg) 
        # self.unlabeled_train_iter = iter(self.unlabeled_trainloader)        
        # self.labeled_train_iter = iter(self.labeled_trainloader)   
        dataloaders=build_dataloader(self.cfg,self.logger)
        
        if self.cfg.DATASET.NAME=='semi-iNat':
            self.labeled_trainloader=dataloaders[0]
            self.labeled_train_iter=iter(self.labeled_trainloader) 
            self.unlabeled_trainloader=dataloaders[1]
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)   
            self.test_loader=dataloaders[2] 
            return
        
        self.domain_trainloader=dataloaders[0]
        self.labeled_trainloader=dataloaders[1]
        self.labeled_train_iter=iter(self.labeled_trainloader)        
        # DU               
        self.unlabeled_trainloader=dataloaders[2]
        self.unlabeled_train_iter=iter(self.unlabeled_trainloader)   
        self.val_loader=dataloaders[3]
        self.test_loader=dataloaders[4]
        self.pre_train_loader=dataloaders[5]
        # self.pre_train_loader=build_contra_dataloader(cfg=self.cfg) 
        self.pre_train_iter=iter(self.pre_train_loader)  
        return  
    
    # def froze_backbone(self,model):
    #     for name, p in model.named_parameters(): 
    #         if 'fc' not in name:
    #             p.requires_grad = False
                
    def finetune(self,model_resume='best_model.pth'):
        self.logger.info("*************** Finetuning ***************")
        self.finetune_iters=self.cfg.FINETUNE_STEP
        model_path=os.path.join(self.model_dir,model_resume)
        
        self.load_checkpoint(model_path)
        # self.froze_backbone(self.model)
        self.model.froze_backbone()
        self.model.reset_classifier()   
        self._rebuild_optimizer(self.model)
        self.build_balanced_dataloader()
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        self.epoch=1
        self.start_iter=1
        for self.iter in range(self.start_iter, self.start_iter+self.finetune_iters):
            return_data=self.finetune_step()
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
                if self.best_val<results[0]:
                    self.best_val=results[0]
                    self.best_val_test=results[1]
                    self.best_val_iter=self.iter
                    self.save_checkpoint(file_name="best_model_finetune.pth")
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint(file_name="checkpoint_finetune.pth")
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.loss_init()             
                start_time = time.time()   
                self.operate_after_epoch()
                self.epoch= (self.iter // self.train_per_step)+1   
        return
    
    def finetune_step(self):
        # self.model.train() 
        # loss =0
        # # DL  
        # try:
        #     data = self.labeled_train_iter.next()    
        # except:            
        #     self.labeled_train_iter=iter(self.labeled_trainloader)            
        #     data= self.labeled_train_iter.next()  
        # inputs=data[0].cuda()
        # targets_x=data[1].long().cuda() 
        
        # logits = self.model(inputs)   
        # score_result = self.func(logits)
        # now_result = torch.argmax(score_result, 1)  
         
        # loss=self.l_criterion(logits, targets_x) 
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.losses.update(loss.item(),inputs.size(0))
        
        # if self.iter % self.cfg.SHOW_STEP==0:
        #     self.logger.info('== Finetune Epoch:{} Step:[{}|{}] Avg_Loss_x:{:>5.4f}   =='\
        #         .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val))
        # return  now_result.cpu().numpy(), targets_x.cpu().numpy()  
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
         
        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()          
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0) 
        
        # fixmatch pipelines
        logits_concat = self.model(x)
        num_labels=inputs_x.size(0)
        logits_x = logits_concat[:num_labels]

        # loss computation 
        lx=self.l_criterion(logits_x, targets_x.long()) 
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        with torch.no_grad():
            # compute pseudo-label
            p = logits_weak.softmax(dim=1)  # soft pseudo labels
            confidence, pred_class = torch.max(p.detach(), dim=1) 
            loss_weight = confidence.ge(self.conf_thres).float()
         
        lu = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=pred_class.size(0)
        ) 
        loss+=lx+lu
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(lx.item(), inputs_x.size(0))
        self.losses_u.update(lu.item(), inputs_u.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        if self.iter % self.cfg.SHOW_STEP==0:
             self.logger.info('== Finetune Epoch:{} Step:[{}|{}] Avg_Loss_x:{:>5.4f} Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses_x.avg,self.losses_u.avg))
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    
    def build_balanced_dataloader(self):
        l_dataset = self.labeled_trainloader.dataset
        l_data_np,transform= l_dataset.select_dataset(return_transforms=True) 
        new_l_dataset = BaseNumpyDataset(l_data_np, transforms=transform,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_l_dataset,sampler_name='ClassBalancedSampler')
        self.labeled_trainloader=new_loader
        self.labeled_train_iter=iter(self.labeled_trainloader)
        return   
    
    def train_warmup_step_by_dl_contra(self):
        self.model.train()
        loss =0
        # DL  
        try:
            (inputs_x,inputs_x2), targets,_ = self.pre_train_iter.next()  
        except:            
            self.pre_train_iter=iter(self.pre_train_loader)            
            (inputs_x,inputs_x2),targets,_ = self.pre_train_iter.next()  
            
        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda()         
        out_1 = self.model(inputs_x,return_encoding=True) 
        out_2 = self.model(inputs_x2,return_encoding=True)  
        out_1=self.model(out_1,return_projected_feature=True) 
        out_2=self.model(out_2,return_projected_feature=True) 
        similarity  = pairwise_similarity(out_1,out_2,temperature=self.warmup_temperature) 
        mask= torch.eq(\
            targets.contiguous().view(-1, 1).cuda(), \
            targets.contiguous().view(-1, 1).cuda().T).float()
        
        loss        = SCL(similarity,mask) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val,self.losses_x.avg,self.losses_u.val))
        return 
    
    def train_warmup_step(self): #
        self.model.train() 
        loss =0
        # DL  
        try:
            (inputs_x,inputs_x2), _,_ = self.pre_train_iter.next()    
        except:            
            self.pre_train_iter=iter(self.pre_train_loader)            
            (inputs_x,inputs_x2),_,_ = self.pre_train_iter.next()  
        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda() 
        
        out_1 = self.model(inputs_x,return_encoding=True) 
        out_2 = self.model(inputs_x2,return_encoding=True)  
        out_1=self.model(out_1,return_projected_feature=True)
        out_2=self.model(out_2,return_projected_feature=True)
                
        similarity  = pairwise_similarity(out_1,out_2,temperature=self.warmup_temperature) 
        loss        = NT_xent(similarity) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.update(loss.item(),inputs_x.size(0))
        
        if self.ema_enable:
            current_lr = self.optimizer.param_groups[0]["lr"]
            ema_decay =self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,self.train_per_step,self.losses.val,self.losses_x.avg,self.losses_u.val))
        return 
    
    def train_step(self,pretraining=False):
        pass
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        
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
    
    def build_data_loaders_for_dl_contra(self,)  :  
        l_dataset = self.labeled_trainloader.dataset
        l_data_np= l_dataset.select_dataset()
        _,transform= self.pre_train_loader.dataset.select_dataset(return_transforms=True)
        new_l_dataset = BaseNumpyDataset(l_data_np, transforms=transform,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_l_dataset,is_train=False)
        self.pre_train_loader=new_loader
        self.pre_train_iter=iter(self.pre_train_loader)
        return  
    
    def rebuild_unlabeled_dataset(self,selected_inds):
        ul_dataset = self.unlabeled_trainloader.dataset
        ul_data_np,ul_transform = ul_dataset.select_dataset(indices=selected_inds,return_transforms=True)

        new_ul_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_transform,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_ul_dataset)
        self.unlabeled_trainloader=new_loader
        if self.ul_test_loader is not None:
            _,ul_test_transform=self.ul_test_loader.dataset.select_dataset(return_transforms=True)
            new_ul_test_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_test_transform,num_classes=self.num_classes)
            self.ul_test_loader = _build_loader(
                self.cfg, new_ul_test_dataset, is_train=False, has_label=False
            )
        self.unlabeled_train_iter = iter(new_loader)
    
    def detect_ood(self):
        # normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)        
        l_feat,l_y=self.prepare_feat(self.test_labeled_trainloader)
        # l_feat=normalizer(l_feat)
        u_feat,u_y=self.prepare_feat(self.test_unlabeled_trainloader) 
        # u_feat=normalizer(u_feat)
        du_gt=torch.zeros(self.ul_num) 
        id_mask=torch.zeros(self.ul_num).long().cuda()
        ood_mask=torch.zeros(self.ul_num).long().cuda()
        du_gt=(u_y>=0).float().long().cuda() 
        index = faiss.IndexFlatL2(l_feat.shape[1])
        index.add(l_feat)
        D, _ = index.search(u_feat, self.k) 
        novel = -D[:,-1] # -最大的距离
        D2, _ = index.search(l_feat, self.k)
        known= - D2[:,-1] # -最大的距离 
        known.sort() # 从小到大排序 负号
        thresh = known[round((1-self.id_thresh_percent)*self.l_num)] #known[50] #  known[round(0.05 * self.l_num)]        
        id_masks= (torch.tensor(novel)>=thresh).float()
        ood_masks=1-id_masks 
        self.id_masks= id_masks
        self.ood_masks=1-id_masks
        self.id_masks=self.id_masks.cuda()
        self.ood_masks=self.ood_masks.cuda()
        self.id_detect_fusion.reset()
        self.ood_detect_fusion.reset()
        self.id_detect_fusion.update(id_masks.numpy(),du_gt) 
        self.ood_detect_fusion.update(ood_masks,1-du_gt)  
        if self.rebuild_unlabeled_dataset_enable and self.iter==self.warmup_iter:
            id_index=torch.nonzero(id_mask == 1, as_tuple=False).squeeze(1)
            id_index=id_index.cpu().numpy()
            self.rebuild_unlabeled_dataset(id_index)   
    
    
    def operate_after_epoch(self): 
        if self.warmup_enable:
            if self.iter<=self.warmup_iter:            
                self.detect_ood()
            if self.iter==self.warmup_iter: 
                self.save_checkpoint(file_name="warmup_model.pth")
            id_pre,id_rec=self.id_detect_fusion.get_pre_per_class()[1],self.id_detect_fusion.get_rec_per_class()[1]
            ood_pre,ood_rec=self.ood_detect_fusion.get_pre_per_class()[1],self.ood_detect_fusion.get_rec_per_class()[1]
            
            # ood_pre,id_pre=self.id_detect_fusion.get_pre_per_class()
            # ood_rec,id_rec=self.id_detect_fusion.get_rec_per_class()
            tpr=self.id_detect_fusion.get_TPR()
            tnr=self.ood_detect_fusion.get_TPR()
            self.logger.info("== ood_prec:{:>5.3f} id_prec:{:>5.3f} ood_rec:{:>5.3f} id_rec:{:>5.3f}".\
                format(ood_pre*100,id_pre*100,ood_rec*100,id_rec*100))
            self.logger.info("=== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(tpr*100,tnr*100))
            self.logger.info('=='*40)     
        else:
            self.logger.info("=="*30)
        pass
    
    def get_val_model(self,):
        return self.model
    
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda()
        self.ema_model = EMAModel(
            self.model,
            self.cfg.MODEL.EMA_DECAY,
            self.cfg.MODEL.EMA_WEIGHT_DECAY,
        )
        # .cuda() 
        
    def _rebuild_optimizer(self, model):
        self.optimizer = self.build_optimizer(self.cfg, model)
        
        torch.cuda.empty_cache()
    
    
    def get_test_best(self):
        return self.best_val_test
    
    def evaluate(self,return_group_acc=False,return_class_acc=False):  
        eval_model=self.get_val_model() 
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(eval_model,self.test_loader, self.val_criterion)
        if self.valset_enable:
            val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(eval_model,self.val_loader, self.val_criterion) 
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
    
    
    def get_case_pred_gt(self): 
        model=self.get_val_model()
        model.eval() 
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                # compute output
                outputs = model(inputs,training=False) 
                probs_u_w = torch.softmax(outputs.detach(), dim=-1)
                max_probs, pred_class = torch.max(probs_u_w, dim=-1)  
                return max_probs[max_probs.shape[0]//3],pred_class[pred_class.shape[0]//3] 

    def get_case_cosine(self): 
        model=self.get_val_model()
        model.eval() 
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                # compute output
                encoding = model(inputs,return_encoding=True,training=False) 
                feature=model(encoding,return_projected_feature=True,training=False) 
                cos_sim= cosine_similarity(feature.detach(),feature.detach()) 
                return cos_sim.cpu().numpy()
    
    def get_test_data_pred_gt_feat(self):
        model=self.get_val_model()
        model.eval()
        pred=[]
        gt=[]
        feat=[]
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

                # compute output
                outputs = model(inputs,training=False)
                feature=model(inputs,return_encoding=True)
                # feature=model(feature,return_projected_feature=True)
                score_result = self.func(outputs)
                now_result = torch.argmax(score_result, 1)   
                gt.append(targets.cpu())   
                pred.append(now_result.cpu())
                feat.append(feature.cpu())
            pred=torch.cat(pred,dim=0)
            gt=torch.cat(gt,dim=0)
            feat=torch.cat(feat,dim=0)
        return gt,pred,feat
               
    def get_train_dl_data_pred_gt_feat(self):
        model=self.get_val_model()
        model.eval()
        pred=[]
        gt=[]
        feat=[]
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_labeled_trainloader):
                if isinstance(inputs,list):
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                
                # compute output
                outputs = model(inputs)
                feature=model(inputs,return_encoding=True)
                feature=model(feature,return_projected_feature=True)
                score_result = self.func(outputs)
                now_result = torch.argmax(score_result, 1)   
                gt.append(targets.cpu())   
                pred.append(now_result.cpu())
                feat.append(feature.cpu())
            pred=torch.cat(pred,dim=0)
            gt=torch.cat(gt,dim=0)
            feat=torch.cat(feat,dim=0)
        return gt,pred,feat
    
    def get_train_du_data_pred_gt_feat(self):
        model=self.get_val_model()
        model.eval()
        pred=[]
        gt=[]
        feat=[]
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(self.test_unlabeled_trainloader):
                if isinstance(inputs,list):
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                
                # compute output
                outputs = model(inputs)
                feature=model(inputs,return_encoding=True)
                feature=model(feature,return_projected_feature=True)
                score_result = self.func(outputs)
                now_result = torch.argmax(score_result, 1)   
                gt.append(targets.cpu())   
                pred.append(now_result.cpu())
                feat.append(feature.cpu())
            pred=torch.cat(pred,dim=0)
            gt=torch.cat(gt,dim=0)
            feat=torch.cat(feat,dim=0)
        return gt,pred,feat    
    
    def eval_loop(self,model,valloader,criterion):
        losses = AverageMeter() 
        # switch to evaluate mode
        model.eval()
 
        fusion_matrix = FusionMatrix(self.num_classes)
        func = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(valloader):
                # measure data loading time 

                inputs, targets = inputs.cuda(), targets.long().cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)
                if len(outputs)==2 and len(outputs)!=len(targets):
                    outputs=outputs[0]
                loss = criterion(outputs, targets)

                # measure accuracy and record loss 
                losses.update(loss.item(), inputs.size(0)) 
                score_result = func(outputs)
                now_result = torch.argmax(score_result, 1) 
                fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
                 
        group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
        class_acc=fusion_matrix.get_acc_per_class()
        acc=fusion_matrix.get_accuracy()    
        return (losses.avg, acc, group_acc,class_acc)
  
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
        try:
            self.optimizer.load_state_dict(state_dict["optimizer"])  
        except: 
            self.logger.warning('load optimizer wrong!')
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
     
    def count_p_d(self):
        dl_results=self.prepare_feat(self.test_labeled_trainloader,return_confidence=True)
        du_results=self.prepare_feat(self.test_unlabeled_trainloader,return_confidence=True)
        test_results=self.prepare_feat(self.test_loader,return_confidence=True)
        
        prototypes=self.get_prototypes(dl_results[0],dl_results[1])
        
        c1=dl_results[2][1]
        # p1=prototypes[dl_results[1]]
        d1=cosine_similarity(dl_results[0],prototypes)  
        
        c2=du_results[2][1]
        # p2=prototypes[du_results[2][1]]        
        d2=cosine_similarity(du_results[0],prototypes) 
        
        c=torch.cat([c1,c2],dim=0)
        d=torch.cat([d1,d2],dim=0)
        
        train_dc=torch.zeros(11,11)
        for i in range(c.shape[0]):
            for j in range(self.num_classes):
                x,y=int(c[i][j].item()/0.1),int((d[i][j].item()+1)/0.2)
                train_dc[x][y]+=1
        train_dc=train_dc.numpy()
        
        test_dc=torch.zeros(11,11)
        test_c=test_results[2][1]
        # test_p=prototypes[test_results[2][1]]  
        test_d=cosine_similarity(test_results[0],prototypes) 
        for i in range(test_c.shape[0]):
            for j in range(self.num_classes):
                x,y=int(test_c[i][j].item()/0.1),int((test_d[i][j].item()+1)/0.2)
                test_dc[x][y]+=1
        test_dc=test_dc.numpy()
        return  train_dc,test_dc
    
    def get_prototypes(self,feat,y):
        prototypes=torch.zeros(self.num_classes,self.feature_dim)
        for c in range(self.num_classes):
            select_index= torch.nonzero(y == c, as_tuple=False).squeeze(1)
            if select_index.shape[0]>0: 
                prototypes[c] = feat[select_index].mean(dim=0) 
        return prototypes
         
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
    
    def get_class_counts(self,dataset):
        """
            Sort the class counts by class index in an increasing order
            i.e., List[(2, 60), (0, 30), (1, 10)] -> np.array([30, 10, 60])
        """
        return np.array(dataset.num_per_cls_list)
        # class_count = dataset.num_samples_per_class

        # # sort with class indices in increasing order
        # class_count.sort(key=lambda x: x[0])
        # per_class_samples = np.asarray([float(v[1]) for v in class_count])
        # return per_class_samples
    
    def get_label_dist(self, dataset=None, normalize=None):
        """
            normalize: ["sum", "max"]
        """
        if dataset is None:
            dataset = self.labeled_trainloader.dataset

        class_counts = torch.from_numpy(self.get_class_counts(dataset)).float()
        class_counts = class_counts.cuda()

        if normalize:
            assert normalize in ["sum", "max"]
            if normalize == "sum":
                return class_counts / class_counts.sum()
            if normalize == "max":
                return class_counts / class_counts.max()
        return class_counts

    def build_labeled_loss(self, cfg , warmed_up=False)  :
        loss_type = cfg.MODEL.LOSS.LABELED_LOSS
        num_classes = cfg.MODEL.NUM_CLASSES
        assert loss_type == "CrossEntropyLoss"

        class_count = self.get_label_dist(device=self.device)
        per_class_weights = None
        if cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE and warmed_up:
            loss_override = cfg.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE
            beta = cfg.MODEL.LOSS.COST_SENSITIVE.BETA
            if beta < 1:
                # effective number of samples;
                effective_num = 1.0 - torch.pow(beta, class_count)
                per_class_weights = (1.0 - beta) / effective_num
            else:
                per_class_weights = 1.0 / class_count

            # sum to num_classes
            per_class_weights = per_class_weights / torch.sum(per_class_weights) * num_classes

            if loss_override == "":
                # CE loss
                loss_fn = build_loss(
                    cfg, loss_type, class_count=class_count, class_weight=per_class_weights
                )

            elif loss_override == "LDAM":
                # LDAM loss
                loss_fn = build_loss(
                    cfg, "LDAMLoss", class_count=class_count, class_weight=per_class_weights
                )

            else:
                raise ValueError()
        else:
            loss_fn = build_loss(
                cfg, loss_type, class_count=class_count, class_weight=per_class_weights
            )

        return loss_fn

    def get_dl_embed_center(self):
        model=self.get_val_model().eval()
        emb = []
        centers = [0 for c in range(self.num_classes)]
        cnt = [0 for c in range(self.num_classes)]
        with torch.no_grad():
            for batch_idx,(inputs, targets, _) in enumerate(self.labeled_trainloader):
                if len(inputs)==2 or len(inputs)==3:
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs=self.model(inputs,return_encoding=True)
                outputs=self.model(outputs,return_projected_feature=True)                
                emb.append(outputs.cpu())
                for ii in range(targets.size(0)):
                    cnt[targets[ii].item()] = cnt[targets[ii].item()] + 1
                    centers[targets[ii].item()] = centers[targets[ii].item()] + outputs[ii].cpu()
        for c in range(self.num_classes):
            centers[c] = (centers[c] / cnt[c]).unsqueeze(0)
        centers = torch.cat(centers, dim=0)
        emb = torch.cat(emb, dim=0)
        return emb, centers   
    
    def prepare_feat(self,dataloader,return_confidence=False):
        model=self.get_val_model().eval()
        n=dataloader.dataset.total_num
        feat=torch.zeros((n,self.feature_dim)) 
        targets_y=torch.zeros(n).long()
        confidence=torch.zeros(n)  
        probs=torch.zeros(n,self.num_classes) 
        with torch.no_grad():
            for batch_idx,(inputs, targets, idx) in enumerate(dataloader):
                if len(inputs)==2 or len(inputs)==3:
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda()
                encoding=self.model(inputs,return_encoding=True)
                outputs=self.model(encoding,return_projected_feature=True) 
                logits=self.model(encoding,classifier=True)
                prob = torch.softmax(logits.detach(), dim=-1) 
                max_probs, pred_class = torch.max(prob, dim=-1)  
                
                feat[idx] =   outputs.cpu()  
                targets_y[idx] = targets.cpu()                 
                confidence[idx]=max_probs.cpu()
                probs[idx]=prob.cpu()
                
        # feat=torch.cat(feat,dim=0)
        # targets_y=torch.cat(targets_y,dim=0)
        if return_confidence:
            return feat,targets_y,[confidence,probs]
            
        return feat,targets_y
    
    def pred_unlabeled_data(self):
        model=self.get_val_model() 
        count=[0]*self.num_classes
        fusionMatrix=FusionMatrix(self.num_classes)
        with torch.no_grad():
            for batch_idx,(inputs, targets, idx) in enumerate(self.unlabeled_trainloader):
                inputs, targets = inputs[0].cuda(), targets.cuda()
                logits_u_w=self.model(inputs) 
                probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
                max_probs, pred_class = torch.max(probs_u_w, dim=-1)     
                select_index=torch.nonzero(targets == -1, as_tuple=False).squeeze(1)
                ood_pred=pred_class[select_index]
                ood_probs=max_probs[select_index] 
                if select_index.shape[0]>0:
                    miscls_idx=torch.nonzero(ood_probs >=self.conf_thres, as_tuple=False).squeeze(1)
                    for item in miscls_idx:
                        count[ood_pred[item]]+=1
                
                id_select_index=torch.nonzero(targets != -1, as_tuple=False).squeeze(1) 
                if id_select_index.shape[0]>0:
                    cls_idx=torch.nonzero(max_probs[id_select_index] >=self.conf_thres, as_tuple=False).squeeze(1)
                    if cls_idx.shape[0]>0: 
                        fusionMatrix.update(pred_class[id_select_index][cls_idx].cpu().numpy(),targets[id_select_index][cls_idx].cpu().numpy())
        acc=fusionMatrix.get_acc_per_class()
        return count,acc
                    
    def pred_test_data(self):
        model=self.model.eval()
        # count=[0]*self.num_classes
        # fusionMatrix=FusionMatrix(self.num_classes)
        fusionMatrix2=FusionMatrix(self.num_classes)
        with torch.no_grad():
            for batch_idx,(inputs, targets, idx) in enumerate(self.test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                logits_u_w=self.model(inputs) 
                probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
                max_probs, pred_class = torch.max(probs_u_w, dim=-1)    
                # cls_idx=torch.nonzero(max_probs >=self.conf_thres, as_tuple=False).squeeze(1)
                fusionMatrix2.update(pred_class.cpu().numpy(),targets.cpu().numpy())
                # if cls_idx.shape[0]>0: 
                #     fusionMatrix.update(pred_class[cls_idx].cpu().numpy(),targets[cls_idx].cpu().numpy())
        # acc=fusionMatrix.get_acc_per_class()
        acc0=fusionMatrix2.get_acc_per_class()
        return acc0
              
    
    def ood_detection_knn_pr_test(self,):
        all_features=torch.zeros(self.l_num+self.ul_num
                                 ,self.feature_dim).cuda()
        du_features=torch.zeros(self.ul_num,
                                self.feature_dim).cuda() 
        all_domain_y=torch.cat([torch.zeros(self.ul_num),
                                torch.ones(self.l_num)],dim=0).long().cuda()
        du_gt=torch.zeros(self.ul_num).long().cuda()
        
        id_mask=torch.zeros(self.ul_num).long().cuda()
        ood_mask=torch.zeros(self.ul_num).long().cuda()
        with torch.no_grad():
            for  i, (data, target, idx) in enumerate(self.unlabeled_trainloader):
                inputs=data[0]
                inputs=inputs.cuda() 
                target=target.cuda()
                feat=self.model(inputs,return_encoding=True)
                feat=self.model(feat,return_projected_feature=True)
                du_features[idx]=feat.detach()
                all_features[idx]=feat.detach()                 
                ones=torch.ones_like(target).long().cuda()
                zeros=torch.zeros_like(target).long().cuda()
                gt=torch.where(target>=0,ones,zeros)
                du_gt[idx]=gt
            for  i, (inputs, _, idx) in enumerate(self.labeled_trainloader):
                if len(inputs)==2 or len(inputs)==3:
                    inputs=inputs[0]
                inputs=inputs.cuda()                 
                feat=self.model(inputs,return_encoding=True)
                feat=self.model(feat,return_projected_feature=True)               
                all_features[idx+self.ul_num]=feat.detach()   
        threshes=[i/self.k for i in range(self.k+1)]
        id_precs=[]
        id_recs=[]
        ood_precs=[]
        ood_recs=[]
        old_ood_mask=copy.deepcopy(ood_mask)
        old_id_mask=copy.deepcopy(id_mask)
        for thresh_ in threshes: 
            for i in range(self.update_domain_y_iter): # 迭代10次更新
                select_index=torch.nonzero(id_mask == 0, as_tuple=False).squeeze(1)
                if select_index.size(0)==0:
                    break
                ood_feat=du_features[select_index]
                # [B, K]        
                sim_matrix = torch.mm(ood_feat, all_features.t())
                sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1) # 
                # new_d_y=[]
                # for item in sim_indices: # [n,50] 
                d_y=all_domain_y[sim_indices]        
                count_idnn=torch.count_nonzero(d_y,dim=1)        
                ones=torch.ones_like(count_idnn).cuda()
                zeros=torch.zeros_like(count_idnn).cuda()
                new_d_y_id = torch.where(count_idnn >= int(self.k*thresh_),ones,zeros).long().cuda() 
                new_d_y_ood = torch.where(count_idnn < int(self.k*thresh_),ones,zeros).long().cuda() 
                all_domain_y[select_index]=new_d_y_id 
                id_mask[select_index]=new_d_y_id
                ood_mask[select_index]=new_d_y_ood
                ood_mask[select_index]*=(1-new_d_y_id)  # 避免重复 
            self.ood_detect_fusion.update(ood_mask,1-du_gt)    
            self.id_detect_fusion.update(id_mask,du_gt)  
            id_precs.append(self.id_detect_fusion.get_pre_per_class()[1])
            id_recs.append(self.id_detect_fusion.get_rec_per_class()[1])
            ood_precs.append(self.ood_detect_fusion.get_pre_per_class()[1])
            ood_recs.append(self.ood_detect_fusion.get_rec_per_class()[1]) 
            self.ood_detect_fusion.reset()
            self.id_detect_fusion.reset()
            ood_mask=copy.deepcopy(old_ood_mask)
            id_mask=copy.deepcopy(old_id_mask) 
            all_domain_y=torch.cat([torch.zeros(self.ul_num),
                                    torch.ones(self.l_num)],dim=0).long().cuda()
          
        root_path= os.path.join(get_DL_dataset_alg_DU_dataset_OOD_path(self.cfg),'a_pic')   
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        plot_pr(id_precs,id_recs,save_path=os.path.join(root_path,'knn_id_pr.jpg'))
        plot_pr(ood_precs,ood_recs,save_path=os.path.join(root_path,'knn_ood_pr.jpg'))
        
        return  
    
    def ood_detection_opencos_pr_test(self):
        du_gt=torch.zeros(self.ul_num).long().cuda() 
        id_mask=torch.zeros(self.ul_num).long().cuda()
        ood_mask=torch.zeros(self.ul_num).long().cuda()
        emb_l, center_l = self.get_dl_embed_center()
        emb_u =torch.zeros(self.ul_num,self.feature_dim) 
        with torch.no_grad():
            for  i, (data, target, idx) in enumerate(self.unlabeled_trainloader):
                inputs=data[0]
                inputs=inputs.cuda() 
                target=target.cuda()
                feat=self.model(inputs,return_encoding=True)
                feat=self.model(feat,return_projected_feature=True)
                emb_u[idx]=feat.detach().cpu()                
                ones=torch.ones_like(target).long().cuda()
                zeros=torch.zeros_like(target).long().cuda()
                gt=torch.where(target>=0,ones,zeros)
                du_gt[idx]=gt
        
        sim_l = cosine_similarity(emb_l, center_l) # N_l x C
        sim_u = cosine_similarity(emb_u, center_l) # N_u x C 
        scores_l = torch.max(sim_l, dim=1)[0] 
        mean = torch.mean(scores_l)
        std = torch.std(scores_l)
        itas=[i*0.005 for i in range(1,5001)]
        id_precs=[]
        id_recs=[]
        ood_precs=[]
        ood_recs=[]
        for ita in itas: 
            id_mask = (torch.max(sim_u, dim=1)[0] > mean - ita*std).float() # OOD sample
            ood_mask = (torch.max(sim_u, dim=1)[0] < mean - ita*std).float() # OOD sample
            self.ood_detect_fusion.update(ood_mask,1-du_gt)    
            self.id_detect_fusion.update(id_mask.numpy(),du_gt)
            id_precs.append(self.id_detect_fusion.get_pre_per_class()[1])
            id_recs.append(self.id_detect_fusion.get_rec_per_class()[1])
            ood_precs.append(self.ood_detect_fusion.get_pre_per_class()[1])
            ood_recs.append(self.ood_detect_fusion.get_rec_per_class()[1]) 
            self.ood_detect_fusion.reset()
            self.id_detect_fusion.reset()
        root_path= os.path.join(get_DL_dataset_alg_DU_dataset_OOD_path(self.cfg),'a_pic')   
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        plot_pr(id_precs,id_recs,save_path=os.path.join(root_path,'opencos_id_pr.jpg'))
        plot_pr(ood_precs,ood_recs,save_path=os.path.join(root_path,'opencos_ood_pr.jpg'))
     
    def extract_feature(self):
        model=self.get_val_model()
        model.eval()
        with torch.no_grad():
            labeled_feat=[] 
            labeled_y=[] 
            labeled_idx=[]
            for i,data in enumerate(self.test_labeled_trainloader):
                inputs_x, targets_x,idx=data[0],data[1],data[2]
                if len(inputs_x)<=3:
                    inputs_x=inputs_x[0]
                inputs_x=inputs_x.cuda()
                feat=model(inputs_x,return_encoding=True)
                # feat=model(feat,return_projected_feature=True)
                labeled_feat.append(feat.cpu())
                labeled_y.append(targets_x)
                labeled_idx.append(idx)
            labeled_feat=torch.cat(labeled_feat,dim=0).cpu()
            labeled_y=torch.cat(labeled_y,dim=0) 
            labeled_idx=torch.cat(labeled_idx,dim=0) 
            unlabeled_feat=[]
            unlabeled_y=[] 
            unlabeled_idx=[]
            for i,data in enumerate(self.unlabeled_trainloader):
                inputs_u=data[0][0]
                inputs_u=inputs_u.cuda()
                target=data[1]      
                idx=data[2]      
                feat=model(inputs_u,return_encoding=True)
                # feat=model(feat,return_projected_feature=True)
                unlabeled_feat.append(feat.cpu())
                unlabeled_y.append(target)
                unlabeled_idx.append(idx)
            unlabeled_feat=torch.cat(unlabeled_feat,dim=0)
            unlabeled_y=torch.cat(unlabeled_y,dim=0) 
            unlabeled_idx=torch.cat(unlabeled_idx,dim=0) 
            test_feat=[]
            test_y=[]
            test_idx=[]
            for i,data in enumerate(self.test_loader):
                inputs_x, target,idx=data[0],data[1],data[2]
                inputs_x=inputs_x.cuda()
                feat=model(inputs_x,return_encoding=True)                
                # feat=model(feat,return_projected_feature=True)
                test_feat.append(feat.cpu())            
                test_y.append(target)        
                test_idx.append(idx)
            test_feat=torch.cat(test_feat,dim=0)
            test_y=torch.cat(test_y,dim=0) 
            return (labeled_feat,labeled_y,labeled_idx),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,test_idx)
    
    def get_id_thresh(self):
        model=self.get_val_model().eval()
        emb = []
        centers = [0 for c in range(self.num_classes)]
        cnt = [0 for c in range(self.num_classes)]
        id_targets=[]
        with torch.no_grad():
            for batch_idx,(inputs, targets, _) in enumerate(self.labeled_trainloader):
                if len(inputs)==2 or len(inputs)==3:
                    inputs=inputs[0]
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs=self.model(inputs,return_encoding=True)
                outputs=self.model(outputs,return_projected_feature=True)                
                emb.append(outputs.cpu())
                id_targets.append(targets)
                for ii in range(targets.size(0)):
                    cnt[targets[ii].item()] = cnt[targets[ii].item()] + 1
                    centers[targets[ii].item()] = centers[targets[ii].item()] + outputs[ii].cpu()
        for c in range(self.num_classes):
            centers[c] = (centers[c] / cnt[c]).unsqueeze(0)
        centers = torch.cat(centers, dim=0)        
        emb = torch.cat(emb, dim=0)
        id_targets=torch.cat(id_targets,dim=0)
        mean=torch.zeros(self.num_classes)
        std=torch.zeros(self.num_classes)
        sim_l = cosine_similarity(emb, centers) # N_l x C
        for c in range(self.num_classes):
            idx=torch.nonzero(id_targets == c, as_tuple=False).squeeze(1)
            score=sim_l[idx]
            c_mean=torch.mean(score,dim=0)
            c_std=torch.std(score,dim=0)
            mean[c]=c_mean[c]
            std[c]=c_std[c] 
        return mean,std,centers