from __future__ import print_function
from config.defaults import update_config,_C as cfg
from utils import AverageMeter, accuracy, create_logger,\
    plot_group_acc_over_epoch,prepare_output_path,plot_loss_over_epoch,plot_acc_over_epoch
import csv 
import json
import copy
import numpy as np
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
    # logger,log_path=create_logger(cfg) 

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
trainloader, traindst, train_sampler = build_contra_dataloader(args,cfg)
if args.local_rank == 0:
    print('Number of training data: ', len(traindst))

# Model
if args.local_rank == 0:
    print('==> Building model..')
torch.cuda.set_device(args.local_rank)
model = model_loader.get_model(cfg)
if args.model == 'wide_resnet':
    projector = Projector(expansion=0)
else:
    projector = Projector(expansion=4)

# Log and saving checkpoint information #
if not os.path.isdir('results') and args.local_rank % args.ngpu == 0:
    os.mkdir('results')
args.name += (args.train_type + '_' +args.model + '_' + args.dataset)
data_name='{}_IF{}_R{}'.format(cfg.DATASET.NAME,cfg.DATASET.DL.IMB_FACTOR_L,cfg.DATASET.DU.OOD.RATIO)
loginfo = 'results/log_' + data_name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

if args.local_rank == 0:
    print ('Training info...')
    print (loginfo)

# Model upload to GPU # 
model.cuda()
projector.cuda()
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


# Aggregating model parameter & projection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()
# "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

if cfg.RESUME!='':
    state_dict = torch.load(cfg.RESUME)
    # load model 
    # model_dict=model.state_dict()  
    # pretrained_dict = {k: v for k, v in state_dict["model"].items() if k in self.model.state_dict()}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(state_dict["model"]) 
    projector.load_state_dict(state_dict["projector"]) 
    optimizer.load_state_dict(state_dict["optimizer_state"])  
    start_epoch=state_dict['epoch']
    rng_state=state_dict['rng_state']
    torch.set_rng_state(rng_state)
             
ngpus_per_node = torch.cuda.device_count()
print(torch.cuda.device_count())
cudnn.benchmark = True
print('Using CUDA..')

# # Aggregating model parameter & projection parameter #
# model_params = []
# model_params += model.parameters()
# model_params += projector.parameters()

# LARS optimizer from KAKAO-BRAIN github
# # "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
# base_optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
# optimizer       = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# Cosine learning rate annealing (SGDR) & Learning rate warmup #
# git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

def train(epoch):
    print('\nEpoch: %d' % epoch)

    scheduler_warmup.step()
    model.train()
    projector.train()
    train_sampler.set_epoch(epoch)

    train_loss = 0
    reg_loss = 0
    
    for batch_idx, ((inputs_1, inputs_2), targets,_) in enumerate(trainloader):
        inputs_1, inputs_2 = inputs_1.cuda() ,inputs_2.cuda()
        inputs  = torch.cat((inputs_1,inputs_2))

        outputs = projector(model(inputs))

        similarity  = pairwise_similarity(outputs,temperature=args.temperature) 
        loss        = NT_xent(similarity)

        train_loss += loss.data


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx%10==0 and args.local_rank==0:
        #     logger.info('==Epoch: {} [{}|{}] Loss: {:>5.3f}  | Reg: {:>5.5f} '.format(epoch,batch_idx, len(trainloader),
        #                         train_loss/(batch_idx+1), reg_loss/(batch_idx+1)))
        # progress_bar(batch_idx, len(trainloader),
        #              'Loss: %.3f | Reg: %.5f'
        #              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1)))

    return (train_loss/batch_idx, reg_loss/batch_idx)


def test(epoch):
    model.eval()
    projector.eval()

    test_loss = 0

    # Save at the last epoch #       
    if epoch == start_epoch + args.epoch - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, projector,test_loss, epoch, args,cfg, optimizer)
         
    # Save at every 10 epoch #
    elif epoch > 1 and epoch %10 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model,projector, test_loss, epoch, args,cfg,  optimizer)
         
    return (test_loss)

def checkpoint(model, projector,acc, epoch, args,cfg, optimizer):
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'acc': acc,
        'model': model.state_dict(),
        'projector': projector.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }
    data_name='{}_IF{}_R{}'.format(cfg.DATASET.NAME,cfg.DATASET.DL.IMB_FACTOR_L,cfg.DATASET.DU.OOD.RATIO)
    save_name = './checkpoint/' + data_name + '_' + str(args.seed) 

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
            logwriter.writerow(['epoch', 'train loss', 'reg loss','epoch_Time','eta'])


##### Training #####
for epoch in range(start_epoch, args.epoch):    
    start_time = time.time()   
    train_loss, reg_loss = train(epoch)
    end_time = time.time()           
    time_second=(end_time - start_time)
    eta_seconds = time_second * (args.epoch - epoch)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    start_time = time.time()   
    _ = test(epoch)
    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss.item(), reg_loss,time_second / 60,eta_string])



# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model wide_resnet --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny
# python3 -m torch.distributed.launch --nproc_per_node=8 train_contrastive.py --dataset cifar-100 --model ResNet50 --batch-size 128 --name c100_Utiny40000 --ngpu 8 --ooc_data tiny