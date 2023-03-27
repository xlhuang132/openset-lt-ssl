

import torch  
from utils.ema_model import EMAModel
from utils.misc import AverageMeter  
from .base_trainer import BaseTrainer 
from loss.contrastive_loss import *
 
class PCSSLTrainer(BaseTrainer):
    def __init__(self, cfg):        
        super().__init__(cfg)      
        
        self.losses_p = AverageMeter()
        self.lambda_p=cfg.ALGORITHM.PCSSL.LAMBDA_P
        self.temperature=cfg.ALGORITHM.PCSSL.TEMPERATURE
      
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
        
    def train_step(self,pretraining=False):
        self.model.train()
        loss_dict={}
        # DL  
        try: 
            inputs_x, targets_x,_ = self.labeled_train_iter.next()
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader) 
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # DU  
        try:
            (inputs_u,inputs_u2),_,_ = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            (inputs_u,inputs_u2),_,_ = self.unlabeled_train_iter.next()
            
            
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda() 
        l_logits = self.model(inputs_x)   
        # 1. dl ce loss
        cls_loss = self.l_criterion(l_logits, targets_x)
        loss_dict.update({"loss_cls": cls_loss})
        score_result = self.func(l_logits)
        now_result = torch.argmax(score_result, 1) 
        
        # 2. du cons loss
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_feature=self.model(ul_images,return_encoding=True) 
        ul_logits = self.model(ul_feature,classifier=True) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(self.conf_thres).float()
         
        cons_loss = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
        
        # partial contrast loss
             
        ul_feature_weak,ul_feature_strong=ul_feature.chunk(2)
        similarity=pairwise_partial_similarity(ul_feature_weak,ul_feature_strong)
        partial_ctr_loss=NT_xent(similarity)
        
        loss_dict.update({"loss_p_ctr":self.lambda_p * partial_ctr_loss})
        
        loss = sum(loss_dict.values())
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(cls_loss.item(), inputs_x.size(0))
        self.losses_u.update(cons_loss.item(), inputs_u.size(0)) 
        self.losses_p.update(self.lambda_p*partial_ctr_loss.item(), inputs_u.size(0)) 
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_loss:{:>5.4f} Loss_x:{:>5.4f}  Loss_u:{:>5.4f} Loss_p:{:>5.4f} =='.format(
                self.epoch,
                self.iter%self.train_per_step if self.iter%self.train_per_step>0 else self.train_per_step,
                self.train_per_step,
                self.losses.avg,
                self.losses_x.avg,
                self.losses_u.avg,
                self.losses_p.avg
                ))
            
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()  
        
        
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_p = AverageMeter() 
         
            
   
  