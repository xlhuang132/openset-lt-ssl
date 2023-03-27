import torch 
import argparse
from config.defaults import update_config,_C as cfg
from trainer.build_trainer import build_trainer
# from utils.set_seed import set_seed
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn   
import torch.distributed as dist
import diffdist.functional as distops 
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
    
    ##### arguments for distributted parallel ####
    parser.add_argument('--local_rank', type=int, default=0)   
    parser.add_argument('--ngpu', type=int, default=2) 

    args = parser.parse_args()

    return args

def print_args(args):
    for k, v in vars(args).items():
        print('{:<16} : {}'.format(k, v))
    
 
args = parser() 
update_config(cfg, args) 
if args.local_rank == 0:
    print_args(args)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
seed=7
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# if args.seed != 0:
#     torch.manual_seed(args.seed)

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

IF=cfg.DATASET.IFS# 10,50,100
ood_r=cfg.DATASET.OODRS # basline不用mixup的话不用考虑r 0.0,0.25, 0.5, 0.75,1.0 randomsampler+classreversedsampler没有用到mixup

for if_ in IF:  # if
    # 同分布
    for r in ood_r:  
        cfg.defrost()
        cfg.DATASET.DL.IMB_FACTOR_L=if_
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
        cfg.SEED=seed
        cfg.LOCAL_RANK=args.local_rank
        cfg.DATASET.DU.OOD.RATIO=r
        print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
        cfg.freeze() 
        trainer=build_trainer(cfg)
        trainer.train()
        print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        