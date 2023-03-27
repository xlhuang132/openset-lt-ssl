from __future__ import print_function
from config.defaults import update_config,_C as cfg
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
import csv 
import json
import copy
import numpy as np
import math

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

    parser.add_argument('--ooc_data', type=str, default=None)   

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
if args.model == 'wide_resnet':
    projector = Projector(expansion=0)
    classifier=Classifier(128, cfg.DATASET.NUM_CLASSES)
else:
    projector = Projector(expansion=4)
    classifier=Classifier(512*4, cfg.DATASET.NUM_CLASSES)

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
    try:
        classfier.load_state_dict(state_dict['classifier'])
    except:
        print('load classifier wrong!')
    try:
        optimizer.load_state_dict(state_dict["optimizer_state"]) 
    except:
        optimizer.load_state_dict(state_dict["optimizer"]) 
    try:
        scheduler_warmup.load_state_dict(state_dict['scheduler'])
    except:
        print('load scheduler wrong!')
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
classifier.cuda()
model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
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
classifier   = torch.nn.parallel.DistributedDataParallel(
                classifier,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)

   
# # Aggregating model parameter & scheduler_warmupprojection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()
# model_params += classifier.parameter()
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
# Cosine learning rate annealing (SGDR) & Learning rate warmup #
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)


# # Aggregating model parameter & projection parameter #
# model_params = []
# model_params += model.parameters()
# model_params += projector.parameters()

# LARS optimizer from KAKAO-BRAIN github
# # "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
# base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
# optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# # Cosine learning rate annealing (SGDR) & Learning rate warmup #
# # git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

# build dataloader for ood detection
l_dataset = labeled_trainloader.dataset 
l_data_np,l_transform = l_dataset.select_dataset(return_transforms=True)
new_l_dataset = BaseNumpyDataset(l_data_np, transforms=l_transform,num_classes=cfg.DATASET.NUM_CLASSES)
test_labeled_trainloader = _build_loader(cfg, new_l_dataset,is_train=False)

ul_dataset = unlabeled_trainloader.dataset 
ul_data_np,ul_transform = ul_dataset.select_dataset(return_transforms=True)
new_ul_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_transform,num_classes=cfg.DATASET.NUM_CLASSES)
test_unlabeled_trainloader = _build_loader(cfg, new_ul_dataset,is_train=False)
queue= FeatureQueue(cfg, classwise_max_size=None, bal_queue=True) 

l_criterion,ul_criterion,val_criterion = build_loss(cfg)
best_epoch=0
best_acc=0
best_val_acc=0
num_classes=cfg.DATASET.NUM_CLASSES

def prepare_feat(model,projector,dataloader):
    model=model.eval()
    projector=projector.eval()
    n=dataloader.dataset.total_num
    feature_dim=128
    feat=torch.zeros((n,feature_dim)) 
    targets_y=torch.zeros(n).long()
    with torch.no_grad():
        for batch_idx,(inputs, targets, idx) in enumerate(dataloader):
            if len(inputs)==2 or len(inputs)==3:
                inputs=inputs[0]
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs=model(inputs)
            outputs=projector(outputs) 
            feat[idx]=   outputs.cpu()  
            targets_y[idx]=targets.cpu()  
    return feat,targets_y 

def ood_detect(model,projector):
    l_feat,l_y=prepare_feat(model,projector,test_labeled_trainloader)
    l_feat=normalizer(l_feat)
    u_feat,u_y=prepare_feat(model,projector,test_unlabeled_trainloader)
    u_feat=normalizer(u_feat)
    ul_num=len(test_unlabeled_trainloader.dataset)
    l_num=len(test_labeled_trainloader.dataset)
    du_gt=torch.zeros(ul_num) 
    id_mask=torch.zeros(ul_num).long()
    ood_mask=torch.zeros(ul_num).long()
    du_gt=(u_y>=0).float().long()
    index = faiss.IndexFlatL2(l_feat.shape[1])
    index.add(l_feat)
    k=50
    D, _ = index.search(u_feat, k) 
    novel = -D[:,-1] # -最大的距离
    D2, _ = index.search(l_feat, k)
    known=-D2[:,-1] # -最大的距离 
    known.sort()
    thresh = known[round(0.05 * l_num)] #known[50] #  known[]        
    id_masks= (torch.tensor(novel)>=thresh).float()
    ood_masks=1-id_masks  
    id_masks=id_masks
    id_detect_fusion=FusionMatrix(2) 
    id_detect_fusion.update(id_masks.numpy(),du_gt)  
    ood_pre,id_pre=id_detect_fusion.get_pre_per_class()
    ood_rec,id_rec=id_detect_fusion.get_rec_per_class()
    print("== ood_prec:{:>5.3f} id_prec:{:>5.3f} ood_rec:{:>5.3f} id_rec:{:>5.3f}".\
                format(ood_pre*100,id_pre*100,ood_rec*100,id_rec*100))
    print("=== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(id_rec*100,ood_rec*100))
    print('=='*40)    
    return id_masks,du_gt

def get_id_feature_contrast_loss(feature,targets,id_mask):
    global num_classes
    gather_t_1 = [torch.empty_like(feature) for _ in range(dist.get_world_size())]
    gather_t_2 = [torch.empty_like(targets) for _ in range(dist.get_world_size())]
    gather_t_3 = [torch.empty_like(id_mask) for _ in range(dist.get_world_size())]
    gather_t_1 = distops.all_gather(gather_t_1, feature) # 
    gather_t_2 = distops.all_gather(gather_t_2, targets)
    gather_t_3 = distops.all_gather(gather_t_3, id_mask)
    feature = torch.cat(gather_t_1)
    targets = torch.cat(gather_t_2)
    id_mask = torch.cat(gather_t_3)
    
    feature_loss_temperature=0.5
    prototypes = queue.prototypes
    outputs=torch.cat((feature,prototypes),dim=0)
    B   = outputs.shape[0]
    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    similarity_matrix = (1./feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:feature.size(0),feature.size(0):]
    mask_same_c=torch.eq(\
        targets.contiguous().view(-1, 1).cuda(), \
        torch.tensor([i for i in range(num_classes)]).contiguous().view(-1, 1).cuda().T).float()
    id_mask=id_mask.expand(mask_same_c.size(1),-1).T # torch.Size([10,192]) # old 
    mask_same_c*=id_mask
    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)         
    log_prob = similarity_matrix_exp - torch.log((torch.exp(similarity_matrix_exp) * (1 - mask_same_c)).sum(1, keepdim=True))
    log_prob_pos = log_prob * mask_same_c # mask掉其他的负类
    loss = - log_prob_pos.sum() / mask_same_c.sum()  
    return loss

def get_ood_feature_contrast_loss(features_u,features_u2,ood_mask): 
    gather_t_1 = [torch.empty_like(features_u) for _ in range(dist.get_world_size())]
    gather_t_2 = [torch.empty_like(features_u2) for _ in range(dist.get_world_size())]
    gather_t_3 = [torch.empty_like(ood_mask) for _ in range(dist.get_world_size())]
    gather_t_1 = distops.all_gather(gather_t_1, features_u) # 
    gather_t_2 = distops.all_gather(gather_t_2, features_u2)
    gather_t_3 = distops.all_gather(gather_t_3, ood_mask)
    features_u = torch.cat(gather_t_1)
    features_u2 = torch.cat(gather_t_2)
    ood_mask = torch.cat(gather_t_3)   
    
    prototypes = queue.prototypes
    features=torch.cat([features_u,features_u2],0) 
    all_features=torch.cat([features,prototypes],0) # [138,64]
    B   = all_features.shape[0]
    outputs_norm = all_features/(all_features.norm(dim=1).view(B,1) + 1e-8)
    feature_loss_temperature=0.5
    similarity_matrix = (1./feature_loss_temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))[:features_u.size(0),features_u.size(0):]
    mask_same_c=torch.cat([torch.eye(features_u.size(0)),torch.zeros((features_u.size(0),num_classes))],dim=1)
    mask_same_c=mask_same_c.cuda()
    ood_mask=ood_mask.expand(mask_same_c.size(1),-1).T
    mask_same_c*=ood_mask
    log_prob = similarity_matrix - torch.log((torch.exp(similarity_matrix) * (1 - mask_same_c)).sum(1, keepdim=True))
    log_prob_pos = log_prob * mask_same_c 
    loss = - log_prob_pos.sum() / mask_same_c.sum()    
    return loss   

def mix_up(l_images,ul_images,alpha=0.5):                 
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

            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1.0 - lam)
            rand_idx = torch.randperm(l_images.size(0)) 
            mixed_images = lam * l_images + (1.0 - lam) * ul_images[rand_idx]
            return mixed_images  
    
  
def train(epoch,id_masks,
          ablation_enable=False,
          dual_branch_enable=False,
          mixup_enable=False,
          ood_detection_enable=False,
          pap_loss_enable=False,
          ):
    print('\nEpoch: %d' % epoch)

    scheduler_warmup.step()
    model.train()
    projector.train()
    classifier.train()
    pap_loss_weight=0.2
    conf_thres=0.95
    labeled_sampler.set_epoch(epoch)
    unlabeled_sampler.set_epoch(epoch)
    ood_masks=1-id_masks
    train_loss = 0
    reg_loss = 0
    loss_dict={}
    for it in range(100):
        # DL  
        try: 
            inputs_x, targets_x,meta = labeled_iter.next()
        except:
            labeled_iter=iter(labeled_trainloader) 
            inputs_x, targets_x,meta =  labeled_iter.next()
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        # DU  
        try:
            (inputs_u,inputs_u2),ul_y,u_index = unlabeled_iter.next()
        except:
            unlabeled_iter=iter(unlabeled_trainloader)
            (inputs_u,inputs_u2),ul_y,u_index = unlabeled_iter.next()
            
            
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        ul_y=ul_y.cuda()            
        u_index=u_index.cuda()    
        
        id_mask,ood_mask = torch.ones_like(ul_y).cuda(),torch.zeros_like(ul_y).cuda()         
        
        
        if not ablation_enable or ablation_enable and dual_branch_enable:               
            inputs_dual_x, targets_dual_x=meta['dual_image'],meta['dual_label']            
            inputs_dual_x, targets_dual_x = inputs_dual_x.cuda(), targets_dual_x.cuda(non_blocking=True)        
            
            if not ablation_enable or ablation_enable and mixup_enable:
                # mixup use ood
                if not ablation_enable or ablation_enable and ood_detection_enable:
                    # if 'upper_bound' in cfg.OUTPUT_DIR:
                    #     id_mask=(ul_y>1).float()
                    #     ood_mask=1-id_mask
                    # else:
                    id_mask=id_masks[u_index].detach() 
                    ood_mask=ood_masks[u_index].detach()   
                    ood_index = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)  
                    inputs_dual_x=mix_up(inputs_dual_x, inputs_u[ood_index])
                else:                    
                    inputs_dual_x=mix_up(inputs_dual_x, inputs_u)
                # concat dl 
            inputs_x=torch.cat([inputs_x,inputs_dual_x],0)
            targets_x=torch.cat([targets_x,targets_dual_x],0)
            
            # 1. cls loss
        l_encoding = model(inputs_x)  
        # l_logits = model(l_encoding,classifier=True)    
        l_logits=classifier(l_encoding)
        
        # 1. dl ce loss
        cls_loss = l_criterion(l_logits, targets_x)
        
        loss_dict.update({"loss_cls": cls_loss})
        # compute 1st branch accuracy
        # score_result = func(l_logits)
        # now_result = torch.argmax(score_result, 1)          

        # 2. cons loss
        ul_images=torch.cat([inputs_u , inputs_u2],0)
        ul_encoding=model(ul_images) 
        ul_logits = classifier(ul_encoding) 
        logits_weak, logits_strong = ul_logits.chunk(2)
        with torch.no_grad(): 
            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels 
            confidence, pred_class = torch.max(p, dim=1)

        loss_weight = confidence.ge(conf_thres).float()
        
        loss_weight*=id_mask
        cons_loss = ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=logits_weak.size(0)
        )
        loss_dict.update({"loss_cons": cons_loss})
        
        # modify hxl
        # l_feature=l_encoding
        l_feature2=projector(l_encoding)
        l_feature=l_feature2
        # l_feature = normalizer(l_encoding)
        l_feature = l_feature/(l_feature.norm(dim=1).view(l_feature.shape[0],1) + 1e-8)
        with torch.no_grad():  
            queue.enqueue(l_feature.clone().detach(), targets_x.clone().detach())
                
        # 3. pap loss
        if (not ablation_enable  or ablation_enable and pap_loss_enable ) and epoch>0:
            # modify hxl
            ul_feature=projector(ul_encoding) 
            # ul_feature = normalizer(ul_feature)
            # ul_feature=ul_encoding /(ul_encoding.norm(dim=1).view(ul_encoding.shape[0],1) + 1e-8)
            # ul_feature2=ul_feature                   
            ul_feature_weak,ul_feature_strong=ul_feature.chunk(2)
            all_features=torch.cat((l_feature,ul_feature_weak),dim=0)
            all_target=torch.cat((targets_x,pred_class),dim=0)
            confidenced_id_mask= torch.cat([torch.ones(l_feature.size(0)).cuda(),id_mask*loss_weight],dim=0).long() 
            
            # Lid3=get_id_feature_contrast_loss(ul_feature_weak, pred_class, id_mask*loss_weight)
            Lidfeat=  pap_loss_weight*get_id_feature_contrast_loss(all_features, all_target, confidenced_id_mask)
            
            # losses_pap_id.update(Lidfeat.item(), l_feature.size(0)+id_mask.sum()) 
            
            Loodfeat=0.
            if ood_mask.sum()>0:        
                
                Loodfeat=pap_loss_weight*get_ood_feature_contrast_loss(ul_feature_weak,ul_feature_strong,ood_mask) 
                if Loodfeat.item()<0:  
                    print("Loodfeat : {}".format(Loodfeat.item()))    
                # losses_pap_ood.update(Loodfeat.item(), ood_mask.sum()) 
            loss_dict.update({"pap_loss":Lidfeat+Loodfeat})  
            
        loss = sum(loss_dict.values())
        train_loss+=loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    return train_loss

def evaluate(dataloader):
    model.eval()
    projector.eval()
    classifier.eval() 
    fusion_matrix = FusionMatrix(cfg.DATASET.NUM_CLASSES)
    func = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for  i, (inputs, targets, _) in enumerate(dataloader):
            # measure data loading time 

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)
            logits=classifier(outputs)
            if len(outputs)==2:
                outputs=outputs[0]
            # loss = l_criterion(outputs, targets)

            # # measure accuracy and record loss 
            # losses.update(loss.item(), inputs.size(0)) 
            score_result = func(logits)
            now_result = torch.argmax(score_result, 1) 
            fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
                
    group_acc=fusion_matrix.get_group_acc(cfg.DATASET.GROUP_SPLITS)
    class_acc=fusion_matrix.get_acc_per_class()
    acc=fusion_matrix.get_accuracy() 
    return acc,group_acc,class_acc
    

def test(epoch):
    global best_val_acc,best_epoch,best_acc
    val_acc,val_group_acc,_=evaluate(val_loader)
    test_acc,test_group_acc,_=evaluate(test_loader)
    print('== Epoch {} Test:'.format(epoch))
    print('== Val: Acc:{} Many:{} Medium:{} Few:{}'.format(epoch,val_acc,val_group_acc[0],val_group_acc[1],val_group_acc[2]))   
    print('== Test: Acc:{} Many:{} Medium:{} Few:{}'.format(epoch,test_acc,test_group_acc[0],test_group_acc[1],test_group_acc[2]))      
    test_loss=0
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        best_epoch=epoch
        best_acc=test_acc
        checkpoint(model, projector,classifier,test_loss, epoch, args,cfg, optimizer,scheduler_warmup,mode='best')
        
        
        
    # Save at the last epoch #       
    if epoch == start_epoch + args.epoch - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, projector,classifier,test_loss, epoch, args,cfg, optimizer)
         
    # Save at every 10 epoch #
    elif epoch > 1 and epoch %10 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model,projector,classifier, test_loss, epoch, args,cfg,  optimizer)
         
    
    return val_acc,val_group_acc, test_acc,test_group_acc,best_acc
        

def checkpoint(model, projector,classifier,acc, epoch, args,cfg, optimizer,scheduler_warmup,mode=''):
    # Save checkpoint.
    
    global best_val_acc,best_epoch,best_acc
    print('Saving..')
    state = {
        'epoch': epoch,
        'acc': acc,
        'model': model.state_dict(),
        'projector': projector.state_dict(),
        'classifier':classifier.state_dict(),
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

    # id_masks,du_gt=ood_detect(model,projector)
    # idmask_name = 'results/log_' + data_name + '_' + str(args.seed)+'_idmask'
    # idmask_name = (idmask_name+ '.csv') 
    # with open(idmask_name,'w') as idmask_file:
    #     logwriter = csv.writer(idmask_file, delimiter=',')
    #     # logwriter.writerow(['idx','ifID','GT'])
    #     for i in range(len(id_masks)):
    #         logwriter.writerow([i,id_masks[i].item(),du_gt[i].item()])
                
    # id_masks=id_masks.cuda()
idmask_name = 'results/' + data_name + '_' + str(args.seed)+'_idmask'
idmask_name = (idmask_name+ '.csv') 
idmask_pd=pd.read_csv(idmask_name,delimiter=',',names=['idx','ifID','GT'])
pre=idmask_pd['ifID']
du_gt=idmask_pd['GT']

du_gt=torch.tensor(np.array(du_gt))
id_masks=torch.tensor(np.array(pre))
id_detect_fusion=FusionMatrix(2) 
id_detect_fusion.update(id_masks.numpy(),du_gt)  
ood_pre,id_pre=id_detect_fusion.get_pre_per_class()
ood_rec,id_rec=id_detect_fusion.get_rec_per_class()
print("== ood_prec:{:>5.3f} id_prec:{:>5.3f} ood_rec:{:>5.3f} id_rec:{:>5.3f}".\
            format(ood_pre*100,id_pre*100,ood_rec*100,id_rec*100))
print("=== TPR : {:>5.2f}  TNR : {:>5.2f} ===".format(id_rec*100,ood_rec*100))
print('=='*40)    

id_masks=id_masks.cuda()

##### Training #####  
# if args.local_rank % ngpus_per_node == 0:
for epoch in range(start_epoch, args.epoch):    
    start_time = time.time()   
    train_loss = train(epoch,id_masks)
    end_time = time.time()           
    time_second=(end_time - start_time)
    eta_seconds = time_second * (args.epoch - epoch)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    start_time = time.time()   
    if args.local_rank % ngpus_per_node == 0:
        val_acc,val_group_acc, test_acc,test_group_acc,best_acc = test(epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch,test_acc,test_group_acc[0],\
                test_group_acc[1],test_group_acc[2],val_acc,val_group_acc[0],\
                val_group_acc[1],val_group_acc[2],train_loss/100,time_second / 60,eta_string])



# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model wide_resnet --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny