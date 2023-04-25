from .cifar10 import *
from .cifar100 import *
from .svhn import get_svhn
from .stl10 import get_stl10
from .build_transform import *
from dataset.ood_dataset_map import ood_dataset_map
from dataset.ood_dataset import OOD_Dataset
from .imagenet import get_imagenet_ssl_dataset

def build_test_dataset(cfg):
    dataset_name=cfg.DATASET.NAME
    root=cfg.DATASET.ROOT
    _,_,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        test_dataset=get_cifar10_test_dataset(root,transform_val=transform_val)
        
    else:
        raise "Dataset name {} is not valid!".format(dataset_name)
    print("Test data distribution:"+str(test_dataset.num_per_cls_list))
    return test_dataset
    
def build_domain_dataset_for_ood_detection(cfg):
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    _,_,transform_val=build_transform(cfg)
    train_domain_set=None  
    if dataset_name=='cifar10':
        train_domain_set=get_cifar10_for_ood_detection(dataset_root,ood_dataset, 
                  transform_val=transform_val,
                 download=True,cfg=cfg)
    else:
        raise "Domain dataseet name is not valid!"
    return train_domain_set
 

def build_dataset(cfg,logger=None,test_mode=False):
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    
    if dataset_name=='semi-iNat':
        transform_labeled = ListTransform(cfg.SEMI_INAT.LPIPELINES)
        transform_ulabeled = ListTransform(cfg.SEMI_INAT.UPIPELINES)
        transform_val = BaseTransform(cfg.SEMI_INAT.VPIPELINE )
        return get_imagenet_ssl_dataset(root=dataset_root,
                                        percent=cfg.SEMI_INAT.PERCENT,
                                        transform_labeled=transform_labeled,
                                        transform_ulabeled=transform_ulabeled,
                                        transform_val=transform_val,
                                        cfg=cfg)
    
    
    assert cfg.DATASET.DU.OOD.DATASET in ood_dataset_map.keys()
    
    
    
    
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    
    ood_ratio=cfg.DATASET.DU.OOD.RATIO
    transform_train,transform_train_ul,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        datasets=get_cifar10(dataset_root,  ood_dataset,ood_ratio=ood_ratio, 
                 transform_train=transform_train,transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='cifar100':
        datasets=get_cifar100(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='stl10':
        datasets=get_stl10(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='svhn':
        datasets=get_svhn(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='semi-iNat':
        
        percent=cfg.DATASET.IMAGENET.PERCENT, 
        datasets=get_imagenet_ssl_dataset(dataset_root,percent, 
                 transform_train, transform_train_ul, transform_val,
                 cfg=cfg,logger=logger)
    else:
        raise "Dataset is not valid!"
    
    return datasets

def build_contra_dataset(cfg):
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    assert cfg.DATASET.DU.OOD.DATASET in ood_dataset_map.keys()
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    
    ood_ratio=cfg.DATASET.DU.OOD.RATIO
    transform_train,transform_train_ul,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        datasets=get_contra_cifar10(cfg)
    elif dataset_name=='cifar100':
        datasets=get_contra_cifar100(cfg)
    elif dataset_name=='stl10':
        datasets=get_stl10(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='svhn':
        datasets=get_svhn(dataset_root,  ood_dataset, ood_ratio=ood_ratio, 
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    else:
        raise "Dataset is not valid!"
    
    return datasets
        
    