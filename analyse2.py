from dataset.build_dataloader import build_test_dataloader
import torch
import numpy as np
import os
import torch.nn as nn
import pandas as pd
from utils.plot import *
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
    "DASO",
    "CReST",
    "MOOD" ,
    'CCSSL',
    'DCSSL'
    ]
ood_dataset=["TIN"]
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="codes for analysing model")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/analyse2_cifar10.yaml",
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
       
def analyse1(cfg): 
    
    cfg.defrost()
    cfg.ALGORITHM.NAME='FixMatch' 
    cfg.freeze() 
    
    # ===== prepare pic path ====
    root_path=get_root_path(cfg) 
    pic_save_path= os.path.join(root_path, "a_pic")
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)
    
    model_path=os.path.join(root_path,'models','best_model.pth')
    trainer=build_trainer(cfg)
    trainer.load_checkpoint(model_path)
    train_results,test_results=trainer.count_p_d()       
    # plot group acc  
    name=cfg.ALGORITHM.NAME+'_train_pc.jpg'
    save_path=os.path.join(pic_save_path,name)    
    plot_pd_heatmap(train_results,save_path=save_path)
    
    name=cfg.ALGORITHM.NAME+'_test_pc.jpg'
    save_path=os.path.join(pic_save_path,name)    
    plot_pd_heatmap(test_results,save_path=save_path)
           
     
    print("== Type 1 analysis has been done! ==")
    return
  
def analyse(cfg,analyse_type):
    
    if analyse_type==1:    
        analyse1(cfg)
     
    else:
        print("Invalid analyse type!")
        
    return 


seed=7
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 
np.random.seed(seed)

args = parse_args()
update_config(cfg, args) 
IF=cfg.DATASET.IFS 
ood_r=cfg.DATASET.OODRS  
analyse_type=cfg.ANALYSE_TYPE 

for if_ in IF:  # if
    # 同分布
    for r in ood_r:  
        cfg.defrost()
        cfg.DATASET.DL.IMB_FACTOR_L=if_
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
        cfg.SEED=seed
        cfg.DATASET.DU.OOD.RATIO=r
        cfg.freeze() 
        
        print("========== Start analysing ==========")
        for item in analyse_type: 
            print("=== Start type {} analysing ===".format(item))
            analyse(cfg,item)
        print("=========== Analyse done! ==========")
        
    
   
   