import logging

import torchvision
from yacs.config import CfgNode
import numpy as np
from .base import BaseNumpyDataset
from .build_transform import  build_simclr_transform
from .transform import build_transforms
from .utils_ import make_imbalance, map_dataset, split_trainval, split_val_from_train, x_u_split,ood_inject

def get_stl10(root, out_dataset, start_label=0,ood_ratio=0, 
                 transform_train=None, transform_val=None,test_mode=False,
                 transform_train_ul=None,
                 download=True,cfg=None,logger=None):
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head=cfg.DATASET.DL.NUM_LABELED_HEAD
    imb_factor_l=cfg.DATASET.DL.IMB_FACTOR_L 
    num_ul_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD 
    imb_factor_ul=cfg.DATASET.DU.ID.IMB_FACTOR_UL 
    num_valid = cfg.DATASET.NUM_VALID
    reverse_ul_dist = cfg.DATASET.REVERSE_UL_DISTRIBUTION
    
    num_classes = cfg.DATASET.NUM_CLASSES
    seed = cfg.SEED
    
    ood_r=cfg.DATASET.DU.OOD.RATIO if cfg.DATASET.DU.OOD.ENABLE else 0
    ood_dataset=cfg.DATASET.DU.OOD.DATASET
    ood_root=cfg.DATASET.DU.OOD.ROOT
    # fmt: on

    logger = logging.getLogger()
    
    l_train = map_dataset(torchvision.datasets.STL10(root, split="train", download=True))
    ul_train = map_dataset(torchvision.datasets.STL10(root, split="unlabeled", download=True))
    stl10_test = map_dataset(torchvision.datasets.STL10(root, split="test", download=True))

    # train - valid set split
    stl10_valid = None
    if num_valid > 0:
        l_train, stl10_valid = split_trainval(l_train, num_valid, seed=seed)

    if ood_r>0:
        ul_train=ood_inject(ul_train,ood_root,ood_r,ood_dataset)
    # unlabeled sample generation unber SSL setting
    if algorithm == "Supervised":
        ul_train = None

    # whether to shuffle the class order
    class_inds = list(range(num_classes))

    # make synthetic imbalance for labeled set
    if imb_factor_l > 1:
        l_train, class_inds = make_imbalance(
            l_train, num_l_head, imb_factor_l, class_inds, seed=seed
        )

    
    labeled_data_num=len(l_train['labels'])
    domain_labels=np.hstack((np.ones_like(l_train['labels'],dtype=np.float32),np.zeros_like(ul_train['labels'],dtype=np.float32)))
    
    total_train={'images':np.vstack((l_train['images'],ul_train['images'])),
                 'labels':np.hstack((l_train['labels'],ul_train['labels']))}
    
    l_train = STL10Dataset(l_train, transforms=transform_train)
    if cfg.DATASET.DU.OOD.INCLUDE_ALL:        
        ul_train=ood_inject(ul_train,ood_root,ood_dataset,include_all=True)
    else:
        if ood_r>0:
            ul_train=ood_inject(ul_train,ood_root,ood_r,ood_dataset)
        
    if ul_train is not None:
        ul_train = STL10Dataset(ul_train, transforms=transform_train_ul, is_ul_unknown=True)

    if stl10_valid is not None:
        stl10_valid = STL10Dataset(stl10_valid, transforms=transform_val)
    stl10_test = STL10Dataset(stl10_test, transforms=transform_val)

    logger.info("class distribution of labeled dataset:{}".format(l_train.num_per_cls_list)) 
    logger.info(
        "=> number of labeled data: {}\n".format(
            sum( l_train.num_per_cls_list)
        )
    )
    if ul_train is not None:
       
        logger.info(
            "=> number of unlabeled OOD data: {}\n".format( ul_train.ood_num)
        ) 

    # if cfg.ALGORITHM.PRE_TRAIN.SimCLR.ENABLE:
    train_dataset =STL10Dataset(total_train,transforms=transform_train_ul,num_classes=num_classes)
    if cfg.ALGORITHM.NAME=='OODDetect':
        transform_pre=transform_train_ul
    else:
        transform_pre=build_simclr_transform(cfg)
    pre_train_dataset  =  STL10Dataset(total_train,transforms=transform_pre,num_classes=num_classes)
    return l_train, ul_train, train_dataset, stl10_valid, stl10_test,pre_train_dataset
    # else:
        
    #     train_dataset =STL10Dataset(total_train,transforms=transform_train_ul,num_classes=num_classes)
    #     return l_train, ul_train, train_dataset, stl10_valid, stl10_test


class STL10Dataset(BaseNumpyDataset):

    def __init__(self, *args, **kwargs):
        super(STL10Dataset, self).__init__(*args, **kwargs)
