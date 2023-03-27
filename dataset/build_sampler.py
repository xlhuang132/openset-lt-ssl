
from .class_aware_sampler import * 
from .random_sampler import *
from .class_balanced_sampler import *
from .class_reversed_sampler import *
 

def build_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples) 
        
    elif sampler_type == "ClassAwareSampler":
        sampler = ClassAwareSampler(dataset, total_samples)
        print(
                "ClassAwareSampler is enabled.  "
                "per_class probabilities: {}".format(
                    ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
                )
            )
    elif sampler_type == "ClassBalancedSampler": # 每个类的采样概率一样
        sampler = ClassBalancedSampler(dataset, total_samples)
        print(
                "ClassBalancedSampler is enabled.  " 
            )
    elif sampler_type == "ClassReversedSampler": # 逆类概率采样
        sampler = ClassReversedSampler(dataset, total_samples)
        print(
                "ClassReversedSampler is enabled.  " 
            )
    else:
        raise ValueError
    
    
    return sampler

def build_dist_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None,args=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples) 
        
    elif sampler_type == "ClassAwareSampler":
        sampler = ClassAwareSampler(dataset, total_samples)
        print(
                "ClassAwareSampler is enabled.  "
                "per_class probabilities: {}".format(
                    ", ".join(["{:.4f}".format(v) for v in sampler.per_cls_prob])
                )
            )
    elif sampler_type == "ClassBalancedSampler": # 每个类的采样概率一样
        sampler = ClassBalancedSampler(dataset, total_samples)
        print(
                "ClassBalancedSampler is enabled.  " 
            )
    elif sampler_type == "ClassReversedSampler": # 逆类概率采样
        sampler = DistClassReversedSampler(dataset, total_samples,num_replicas=args.ngpu,
                                    rank=args.local_rank)
        print(
                "ClassReversedSampler is enabled.  " 
            )
    else:
        raise ValueError
    
    
    return sampler