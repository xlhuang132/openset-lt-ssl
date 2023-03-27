from __future__ import print_function
from config.defaults import update_config,_C as cfg
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
import csv 
import json
import copy
import numpy as np
import math
from loss.contrastive_loss import *
from loss.build_loss import build_loss 
from models.feature_queue import FeatureQueue
from dataset.build_dataloader import _build_loader
from models.classifier import Classifier
from utils import FusionMatrix

from dataset.build_dataloader import *
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.distributed as dist
import diffdist.functional as distops 
import model_loader
import models
from models.projector import Projector
from dataset.build_dataloader import build_contra_dataloader
import time
import datetime
import pandas as pd
import faiss
from loss.contrastive_loss import pairwise_similarity,NT_xent

# Download packages from following git #
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars, version 0.1.2
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler
import os
import sys
import argparse

def parser():

    parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning of Visual Representation')
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/moving_center_cifar10.yaml",
        type=str,
    ) 
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument('--train_type', default='contrastive_learning', type=str, help='standard')
    parser.add_argument('--lr', default=1.5, type=float, help='learning rate, LearningRate = 0.3 × BatchSize/256 for ImageNet, 0.5,1.0,1.5 for CIFAR')
    parser.add_argument('--lr_multiplier', default=1.0, type=float, help='learning rate multiplier, 5,10,15 -> 0.5,1.0,1.5 for CIFAR')
    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100/lsun/imagenet-resize/svhn')
    parser.add_argument('--dataroot', default='/data', type=str, help='PATH TO dataset cifar-10, cifar-100, svhn')
    parser.add_argument('--tinyroot', default='/data/tinyimagenet/tiny-imagenet-200/train/', type=str, help='PATH TO tinyimagenet dataset')
    parser.add_argument('--resume', '-r', default='', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet50", type=str,
                        help='model type (default: ResNet50)')
    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=7, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=1000, type=int, 
                        help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR, 1.0 for ImageNet')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')
 
    ##### arguments for linear evaluation #####
    parser.add_argument('--multinomial_l2_regul', default=0.5, type=float, help='regularization for multinomial logistic regression')

    ##### arguments for distributted parallel ####
    parser.add_argument('--local_rank', type=int, default=0)   
    parser.add_argument('--ngpu', type=int, default=2)

    parser.add_argument('--ood_data', type=str, default=None)   

    args = parser.parse_args()

    return args

def print_args(args):
    for k, v in vars(args).items():
        print('{:<16} : {}'.format(k, v))
    

def pairwise_similarity(outputs,temperature=0.5):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    '''
    outputs_1, outputs_2 = outputs.chunk(2)
    gather_t_1 = [torch.empty_like(outputs_1) for _ in range(dist.get_world_size())]
    gather_t_2 = [torch.empty_like(outputs_2) for _ in range(dist.get_world_size())]
    gather_t_1 = distops.all_gather(gather_t_1, outputs_1) # 
    gather_t_2 = distops.all_gather(gather_t_2, outputs_2)
    outputs_1 = torch.cat(gather_t_1)
    outputs_2 = torch.cat(gather_t_2)
    outputs = torch.cat([outputs_1, outputs_2])

    B   = outputs.shape[0]

    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    similarity_matrix = (1./temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))

    return similarity_matrix


args = parser()
cfg.defrost()
cfg.LOCAL_RANK=args.local_rank 
cfg.freeze() 

update_config(cfg, args)
if args.local_rank == 0:
    print_args(args)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

# Data
if args.local_rank == 0:
    print('==> Preparing data..')  
    logger, _ = create_logger(cfg)    
labeled_trainloader,labeled_sampler,\
    unlabeled_trainloader,unlabeled_sampler,\
        val_loader,test_loader,\
             =build_mood_dataloader(args,cfg)
             
labeled_iter=iter(labeled_trainloader)
unlabeled_iter=iter(unlabeled_trainloader)
if args.local_rank == 0:
    print('Number of dl training data: ', len(labeled_trainloader.dataset))
    print('Number of du training data: ', len(unlabeled_trainloader.dataset))

# Model
if args.local_rank == 0:
    print('==> Building model..')
torch.cuda.set_device(args.local_rank)
model = model_loader.get_model(cfg)
biased_model = model_loader.get_model(cfg)
if args.model == 'wide_resnet':
    projector = Projector(expansion=0)
else:
    projector = Projector(expansion=4)

# Log and saving checkpoint information #
if not os.path.isdir('results') and args.local_rank % args.ngpu == 0:
    os.mkdir('results')
args.name += (args.train_type + '_' +args.model + '_' + args.dataset)
data_name='{}_IF{}_R{}'.format(cfg.DATASET.NAME,cfg.DATASET.DL.IMB_FACTOR_L,cfg.DATASET.DU.OOD.RATIO)
loginfo = 'results/log_' +cfg.ALGORITHM.NAME+'_'+ data_name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

if args.local_rank == 0:
    print ('Training info...')
    print (loginfo)
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)        
 

if cfg.RESUME!='':
    state_dict = torch.load(cfg.RESUME)
    model.load_state_dict(state_dict["model"]) 
    projector.load_state_dict(state_dict["projector"]) 
    biased_model.load_state_dict(state_dict["biased_model"])
    optimizer.load_state_dict(state_dict["optimizer"]) 
    scheduler_warmup.load_state_dict(state_dict['scheduler'])
    start_epoch=state_dict['epoch']
    rng_state=state_dict['rng_state']
    torch.set_rng_state(rng_state)
             
ngpus_per_node = torch.cuda.device_count()
print(torch.cuda.device_count())
cudnn.benchmark = True
print('Using CUDA..')

# Model upload to GPU # 
model.cuda()
projector.cuda() 
biased_model=biased_model.cuda()
model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
biased_model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(biased_model)
biased_model       = torch.nn.parallel.DistributedDataParallel(
                biased_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
projector   = torch.nn.parallel.DistributedDataParallel(
                projector,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
   
# # Aggregating model parameter & scheduler_warmupprojection parameter #
model_params = []
model_params += model.parameters()
model_params += biased_model.parameters()
model_params += projector.parameters()
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
# Cosine learning rate annealing (SGDR) & Learning rate warmup #
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

l_criterion,ul_criterion,val_criterion = build_loss(cfg)
best_epoch=0
best_acc=0
best_val_acc=0
num_classes=cfg.DATASET.NUM_CLASSES

def dcssl_contra_loss(feat=None,targets_x=None,topk_pred_label=None,biased_feat=None,sample_mask=None,temperature=0.07):
        topk_pred_label=topk_pred_label.T
        y=torch.cat([targets_x,targets_x,topk_pred_label[0],topk_pred_label[0]],dim=0)
        similarity_matrix= pairwise_similarity_dcssl(feat,temperature=temperature)
        
        # get weight by biased feature
        cos_sim=torch.from_numpy(cosine_similarity(biased_feat.cpu().numpy())).cuda()
        pos_weight = 1 - cos_sim
        neg_weight= cos_sim        
        
        pos_mask=torch.eq(y.contiguous().view(-1, 1),y.contiguous().view(-1, 1).T) 
        neg_mask=~pos_mask
        # pos_weight 需要乘以一个mask，mask掉其他的负类的 通过top-k 
        for i in range(1,topk_pred_label.size(0)):
            tmp_y=torch.cat([targets_x,targets_x,topk_pred_label[i],topk_pred_label[i]],dim=0)
            tmp_mask=torch.eq(tmp_y.contiguous().view(-1, 1),tmp_y.contiguous().view(-1, 1).T) 
            pos_mask |= tmp_mask   
            neg_mask &= ~tmp_mask   
                  
         
        pos_mask=pos_mask.float() 
        neg_mask=neg_mask.float()
        
        
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
        
        
        con_loss =(-mean_log_prob_pos* sample_mask).mean() 
        return con_loss

def get_u_loss_weight(confidence,conf_thres):
        # 根据logits阈值
        loss_weight=confidence.ge(conf_thres).float()  
        # 根据能量分数
        
        return loss_weight
    
def train(epoch,biased_model,model,projector,\
    l_criterion,ul_criterion,conf_thres=0.95,cfg=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    biased_model.train()
    projector.train() 
    lambda_d=cfg.ALGORITHM.DCSSL.LAMBDA_D
    fp_k=cfg.ALGORITHM.DCSSL.FP_K
    labeled_sampler.set_epoch(epoch)
    unlabeled_sampler.set_epoch(epoch)
    train_loss = 0  
    train_per_step=cfg.TRAIN_STEP
    dcssl_contra_temperture=cfg.ALGORITHM.DCSSL.DCSSL_CONTRA_TEMPERTURE
    for it in range(1,1+train_per_step):
        # DL  
        try: 
            data_x = labeled_iter.next()
        except:
            labeled_iter=iter(labeled_trainloader) 
            data_x =  labeled_iter.next()
        
        # DU  
        try:
            data_u = unlabeled_iter.next()
        except:
            unlabeled_iter=iter(unlabeled_trainloader)
            data_u = unlabeled_iter.next()
                   
        loss =0       
        # dataset
        inputs_x=data_x[0][0][0]
        targets_x=data_x[1]        
        inputs_u=data_u[0][0][0]
        inputs=torch.cat([inputs_x,inputs_u],dim=0)
        inputs=inputs.cuda()
        targets_x=targets_x.long().cuda() 
        # ============ biased model =============
        # logits
        logits=biased_model(inputs) 
        logits_x=logits[:inputs_x.size(0)]        
        
        # 1. ce loss
        lb_cls=l_criterion(logits_x,targets_x)         
        # 2. cons loss
        u_weak=logits[inputs_x.size(0):]
        with torch.no_grad(): 
            p = u_weak.detach().softmax(dim=1)  
            confidence, pred_class = torch.max(p, dim=1)
        loss_weight = confidence.ge(conf_thres).float()  
        lb_cons = ul_criterion(
            u_weak, pred_class, weight=loss_weight, avg_factor=u_weak.size(0)
        ) 
        
        # if it % cfg.SHOW_STEP==0:
        #     print('== Biased Epoch:{} Step:[{}|{}]  Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='\
        #         .format(epoch,it%cfg.train_per_step if it%cfg.train_per_step>0 else cfg.train_per_step,
        #                 cfg.train_per_step,self.losses_bx.avg,self.losses_bu.avg))
        
        # ============ debiased model =============
        
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
        
        encodings = model(inputs,return_encoding=True) 
        features=projector(encodings)  
        # === biased_contrastive_loss 
        with torch.no_grad():
            biased_encodings=biased_model(inputs,return_encoding=True)            
            logits_b=biased_model(biased_encodings,classifier=True)
            biased_encodings=biased_encodings.detach()
                        
        logits=model(encodings,classifier=True)
         
        logits_x1,logits_x2=logits[:inputs_x.size(0)*2].chunk(2)
        # 1. ce loss
        logits_x=(logits_x1+logits_x2)*0.5
        
        ld_cls = l_criterion(logits_x, targets_x) 
        
        # 2. cons loss
        u_weak,u_strong=logits[inputs_x.size(0)*2:].chunk(2)
        with torch.no_grad(): 
            p = u_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1) 
            _,  top_k_pred_label = p.topk(fp_k, dim=1) # [128,5]
        
                   
        loss_weight = get_u_loss_weight(confidence,conf_thres)
        ld_cons = ul_criterion(
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
        ld_ctr = dcssl_contra_loss(feat=contra_features,targets_x=targets_x,
                                            topk_pred_label=top_k_pred_label,
                                            biased_feat=biased_encodings,
                                            sample_mask=sample_mask,
                                            temperature=dcssl_contra_temperture) 
        
        loss=lb_cls+lb_cons+ld_cls+ld_cons+lambda_d*ld_ctr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        if cfg.LOCAL_RANK==0 and it % cfg.SHOW_STEP==0:
            logger.info('== Biased Epoch:{} Step:[{}|{}] Avg_Lb_x:{:>5.4f}  Avg_Lb_u:{:>5.4f} =='\
                .format(epoch,it%train_per_step if it%train_per_step>0 else train_per_step,
                        train_per_step,lb_cls,lb_cons)) 
            logger.info('== Debiased Epoch:{} Step:[{}|{}] Avg_Ld_x:{:>5.4f}  Avg_Ld_u:{:>5.4f} Avg_Ld_ctr:{:>5.4f} =='\
                .format(epoch,it%train_per_step if it%train_per_step>0 else train_per_step,
                        train_per_step,ld_cls,ld_cons,ld_ctr)) 
         
        train_loss+=loss.item() 
    
    scheduler_warmup.step()
    return train_loss

def evaluate(dataloader):
    model.eval()
    projector.eval() 
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    func = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for  i, (inputs, targets, _) in enumerate(dataloader):
            # measure data loading time 

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs) 
            if len(outputs)==2:
                outputs=outputs[0]
            # loss = l_criterion(outputs, targets)
            logits=model(inputs) 
            # # measure accuracy and record loss 
            
            score_result = func(logits)
            now_result = torch.argmax(score_result, 1) 
            fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
                
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS)
    class_acc=fusion_matrix.get_acc_per_class()
    acc=fusion_matrix.get_accuracy() 
    return acc,group_acc,class_acc
    

def test(epoch):
    global best_val_acc,best_epoch,best_acc,cfg
    test_acc,test_group_acc,_=evaluate(test_loader)
    if cfg.DATASET.NUM_VALID>0:
        val_acc,val_group_acc,_=evaluate(val_loader)
    else:
        val_acc,val_group_acc= test_acc,test_group_acc
    print('== Epoch {} Test:'.format(epoch))
    print('== Val: Acc:{} Many:{} Medium:{} Few:{}'.format(epoch,val_acc,val_group_acc[0],val_group_acc[1],val_group_acc[2]))   
    print('== Test: Acc:{} Many:{} Medium:{} Few:{}'.format(epoch,test_acc,test_group_acc[0],test_group_acc[1],test_group_acc[2]))      
    test_loss=0
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        best_epoch=epoch
        best_acc=test_acc
        checkpoint(model,biased_model, projector,test_loss, epoch, args,cfg, optimizer,scheduler_warmup,mode='best')
        
        
        
    # Save at the last epoch #       
    if epoch == start_epoch + args.epoch - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, projector,classifier,test_loss, epoch, args,cfg, optimizer)
         
    # Save at every 10 epoch #
    elif epoch > 1 and epoch %10 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model,projector,classifier, test_loss, epoch, args,cfg,  optimizer)
         
    
    return val_acc,val_group_acc, test_acc,test_group_acc,best_acc
        

def checkpoint(model,biased_model, projector,acc, epoch, args,cfg, optimizer,scheduler_warmup,mode=''):
    # Save checkpoint.
    
    global best_val_acc,best_epoch,best_acc
    print('Saving..')
    state = {
        'epoch': epoch,
        'acc': acc,
        'model': model.state_dict(),
        'projector': projector.state_dict(),
        'biased_model':biased_model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler':scheduler_warmup.state_dict(),
        'rng_state': torch.get_rng_state(),
        'best_val_acc':best_val_acc,
        'best_epoch':best_epoch,
        'best_acc':best_acc,
    }
    data_name='IF{}_R{}'.format(cfg.DATASET.DL.IMB_FACTOR_L,cfg.DATASET.DU.OOD.RATIO)
    save_name = './checkpoint/' + data_name + '_' + str(args.seed) 
    if mode!='':
        save_name+='_'+mode
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_name)

##### Log file #####
if args.local_rank % ngpus_per_node == 0:     
    if start_epoch==0:
        if os.path.exists(logname):
            os.remove(logname)
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'test_acc', 'test_many','test_medium','test_few','val_acc', 'val_many','val_medium','val_few','train_loss','epoch_Time','eta'])


##### Training #####  
# if args.local_rank % ngpus_per_node == 0:
for epoch in range(start_epoch, args.epoch):    
    start_time = time.time()   
    train_loss = train(epoch,model,biased_model,projector,l_criterion,ul_criterion,cfg=cfg)
    end_time = time.time()           
    time_second=(end_time - start_time)
    eta_seconds = time_second * (args.epoch - epoch)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    start_time = time.time()   
    if args.local_rank % ngpus_per_node == 0:
        val_acc,val_group_acc, test_acc,test_group_acc,best_acc = test(epoch)
        logger.info([epoch,test_acc,test_group_acc[0],\
                test_group_acc[1],test_group_acc[2],val_acc,val_group_acc[0],\
                val_group_acc[1],val_group_acc[2],train_loss/100,time_second / 60,eta_string])
        logger.info("===============================================")


# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model wide_resnet --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny