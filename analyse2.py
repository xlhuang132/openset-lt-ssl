from dataset.build_dataloader import build_test_dataloader
import torch
import numpy as np
import os
import torch.nn as nn
import pandas as pd
from utils.plot import plot_accs_zhexian
from utils.validate_model import validate
from utils.utils import *
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
from dataset.build_dataloader import *
from models.ood_detector import ood_detect_ODIN,get_fpr95
import argparse
import models
import copy
from utils.set_seed import set_seed
from utils.plot import *
# from train_ours import prepare_feat,knn_ood_detect
from utils.utils import load_checkpoint
from trainer import *
from models.projector import  Projector
 
algorithms=algorithms=[
    "Supervised",
    "FixMatch", 
    "MixMatch",
    # "PseudoLabel",
    # "MTCF",
    "DASO",
    "CReST",
    "MOOD"
    # "DS3F"
    ]
ood_dataset=["TIN"]
ood_r=[0.25,0.5,0.75]
IF=[10,50,100]  

def parse_args():
    parser = argparse.ArgumentParser(description="codes for analysing model")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/analyse_cifar10.yaml",
        type=str,
    ) 
    
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

def analyse_1(cfg):
    trainer=build_trainer(cfg)
    _, avg_acc ,group_acc,class_acc= trainer.evaluate(return_class_acc=True,return_group_acc=True)     
    gt1,pred1,feat1 = trainer.get_test_data_feat(model='biased')
    gt2,pred2,feat2 = trainer.get_test_data_feat(model='debiased')  
    return 
def analyse(cfg,analyse_type):
    if analyse_type==1:    
        analyse_1(cfg)
    else:
        print("Invalid analyse type!")
        
    return 

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    
    cudnn.benchmark = True 
    analyse_type=cfg.ANALYSE_TYPE 
    print("========== Start analysing ==========")
    for item in analyse_type:
        assert item in analyse_type_map.keys()
        print("=== Start type {} analysing : {}===".format(item,analyse_type_map[item]))
        analyse(cfg,item)
    print("=========== Analyse done! ==========")