import logging
import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from typing import Tuple  
from .build_dataset import build_dataset
from .build_sampler import build_sampler 


def build_data_loaders(cfg): 
    train_dataset,val_dataset,aux_dataset = build_dataset(cfg)
    total_samples=len(train_dataset) 
    train_sampler=build_sampler(cfg, train_dataset,sampler_type=cfg.TRAIN.SAMPLER,total_samples=total_samples)
    # 创建DL数据加载器
    train_loader=DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS, 
        sampler=train_sampler,
        shuffle=False
    )
    # 创建DU数据加载器
    aux_loader=None 
    if aux_dataset is not None:
        if cfg.DATASET.AUX_DATASET.NUM_LIMITED:
            aux_sampler=build_sampler(cfg, aux_dataset,total_samples=total_samples)
            aux_loader=DataLoader(
            aux_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            sampler=aux_sampler,
            shuffle=False
        )  
        else:
            # 不对辅助数据的数量进行限制，那么就每个bs与训练数据不一样
            aux_sampler=build_sampler(cfg, aux_dataset,total_samples=len(aux_dataset))
            batch_size=int(cfg.TRAIN.BATCH_SIZE*(len(aux_dataset)/total_samples))
            aux_loader=DataLoader(
            aux_dataset,
            batch_size=batch_size,
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            sampler=aux_sampler,
            shuffle=False, 
        )  
        
    test_batch_size=cfg.TEST.BATCH_SIZE
    val_loader= DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        num_workers=cfg.TEST.NUM_WORKERS,  
        shuffle=True, 
    )
    return train_loader, val_loader, aux_loader