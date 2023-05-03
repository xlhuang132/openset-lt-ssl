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


import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
import copy
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
from utils.build_optimizer import get_optimizer
import models 
import time 
import torch.optim as optim 
import os   
from utils.plot import *
import datetime
import torch.nn.functional as F   
from loss.contrastive_loss import * 
from utils import FusionMatrix 
from dataset.base import BaseNumpyDataset 
from utils.misc import AverageMeter  
from loss.debiased_soft_contra_loss import *
from utils import OODDetectFusionMatrix
from models.feature_queue import FeatureQueue
from loss.focal_loss import FocalLoss

def cosine_similarity(x1, x2, eps=1e-12):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

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
       
# 1. 统计置信度和与类中心距离的矩阵并热力图可视化
def analyse1(cfg): 
    algs=['MixMatch','FixMatch' ,'CCSSL','DCSSL']
    for item in algs:
        cfg.defrost()
        cfg.ALGORITHM.NAME=item
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
 
# 2. 统计OOD样本被错分为哪个类
def analyse2(cfg):
    algs=['FixMatch','DCSSL','OpenMatch','CReST','DASO' ,'DCSSL']#['MixMatch','FixMatch', 'OpenMatch' ,'CCSSL','DCSSL']
    mis_ood=[]
    cls_id=[]
    legends=['FixMatch-0.95','FixMatch-CT','OpenMatch','CReST','DASO','DeCAB (Ours)']
    i=0
    print(plt.rcParams)
    for item in algs:
        cfg.defrost()
        cfg.ALGORITHM.NAME=item
        if i==2:cfg.ALGORITHM.DCSSL.LOSS_VERSION=12 
        cfg.freeze() 
        
        # ===== prepare pic path ====
        root_path=get_root_path(cfg) 
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)
        result=trainer.pred_unlabeled_data() #pred_test_data() # count,acc
        mis_ood.append(result[0])
        cls_id.append(result[1])
        i+=1
        # mis_ood.append(list(result))
        # cls_id.append(list(result))
    path=os.path.join(get_DL_dataset_alg_DU_dataset_path(cfg),'mis_du_ood.jpg') 
    plot_zhexian_with_bg(mis_ood, legends, 
                         'Number of Misclassified OOD Data @CIFAR100-LT($IF$100, TIN)', 
                         np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),
                         group_split=cfg.DATASET.GROUP_SPLITS,
                         xlabel='Class ID',
                         save_path=path)    
    # path=os.path.join(get_DL_dataset_alg_DU_dataset_path(cfg),'cls_du_id.jpg')
    # plot_accs_zhexian(cls_id, algs, 'Unlabeled ID Data', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),save_path=path)
    # print(cls_id)
    # path=os.path.join(get_DL_dataset_alg_DU_dataset_path(cfg),'cls_test.jpg')
    # plot_accs_zhexian(cls_id, algs, 'Test Data ACC', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),save_path=path)
    # print(mis_ood)
    # path=os.path.join(get_DL_dataset_alg_DU_dataset_path(cfg),'cls_test.jpg')
    # plot_accs_zhexian(mis_ood, algs, 'Test Data ACC', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),save_path=path)
       
# 3. 可视化特征空间
def analyse3(cfg):
    algs=['MixMatch','FixMatch', 'OpenMatch' ,'CCSSL','DCSSL']    
    gts,preds,feats=[],[],[]
    # ===== prepare pic path ====
    
    for item in algs:
        cfg.defrost()
        cfg.ALGORITHM.NAME=item
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)
       
        gt,pred,feat = trainer.get_test_data_pred_gt_feat()     
        gts.append(gt)
        preds.append(pred)
        feats.append(feat)
        # (labeled_feat,labeled_y,_),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,_)= trainer.extract_feature()  
        # # 画labeled tsne图
        # plot_feat_tsne(labeled_feat,labeled_y,'Labeled features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"{}_labeled_feat_tsne.jpg".format(item)))
        # # 画unlabeled tsne图        
        # plot_feat_tsne(unlabeled_feat,unlabeled_y,'Unlabeled features TSNE',cfg.DATASET.NUM_CLASSES+1,save_path=os.path.join(pic_save_path,"{}_unlabeled_feat_tsne.jpg".format(item)))
        # # 画test tsne图
        # plot_feat_tsne(test_feat,test_y,'Test features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"{}_test_feat_tsne.jpg".format(item)))
 
    titles=['(a) ','(b) ','(c) ','(d) ','(e) ','(f) ']
    for i in range(len(algs)):
        titles[i]=titles[i]+algs[i]
    save_path=os.path.join(pic_save_path,'test_feat_tsne.png' )
    plot_problem_feat_tsne(gts,preds,feats,num_classes=cfg.DATASET.NUM_CLASSES,
                        titles=titles,      
                    save_path=save_path)
    print("Analyse3 done!")
    
# 4. motivation 实验，统计硬正硬负分数 test data
def count_hardp_hardn(algs,mode='train',datatype='labeled'):
    hp_means,hn_means,hp_counts,hn_counts=[],[],[],[]
    # ===== prepare pic path ====
    for alg in algs:
        cfg.defrost()
        cfg.ALGORITHM.NAME=alg
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)
        if mode=='train' and datatype=='labeled':
            gt,pred,feat = trainer.get_train_dl_data_pred_gt_feat()   
        elif mode=='train' and datatype=='unlabeled':
            gt,pred,feat = trainer.get_train_du_data_pred_gt_feat()   
        else:
            gt,pred,feat = trainer.get_test_data_pred_gt_feat()  
             
        # prototypes=trainer.get_prototypes(feat,gt)
        # cos_sim = cosine_similarity(feat,prototypes[gt])
        # cos_sim = cosine_similarity(feat.cpu().numpy(),feat.cpu().numpy())    
        y=gt.cpu().view(-1).contiguous().view(-1, 1)
        labeled_mask= torch.eq(y, y.T).float()     
        cos_sim = cosine_similarity(feat.cpu(),feat.cpu())
        cos_sim=torch.tensor(np.array(cos_sim))
        
        p_cos = labeled_mask*(1-torch.eye(labeled_mask.shape[0]))*cos_sim
        # (labeled_mask*cos_sim+(1-labeled_mask)).min(dim=1)[0]
        n_cos = (1-labeled_mask)*cos_sim
        # ((1-labeled_mask)*cos_sim).max(dim=1)[0]
        
        hp_mean=[0]*cfg.DATASET.NUM_CLASSES
        hp_count=[0]*cfg.DATASET.NUM_CLASSES
        hn_mean=[0]*cfg.DATASET.NUM_CLASSES
        hn_count=[0]*cfg.DATASET.NUM_CLASSES
       
        for i in range(cfg.DATASET.NUM_CLASSES):
            select_index=torch.nonzero(i==gt,as_tuple=False).squeeze(1)
            hp_mean[i]=p_cos[select_index].sum()/labeled_mask[select_index].sum()
            hn_mean[i]=n_cos[select_index].sum()/(1-labeled_mask[select_index]).sum()
            hp_count[i]=((p_cos+1-labeled_mask)[select_index]<0.5).float().sum(dim=1).mean()
            hn_count[i]=(n_cos[select_index]>=0.5).float().sum(dim=1).mean()
            
        hp_means.append(copy.deepcopy(hp_mean))
        hn_means.append(copy.deepcopy(hn_mean))  
        hp_counts.append(copy.deepcopy(hp_count))
        hn_counts.append(copy.deepcopy(hn_count)) 
       
    file_name='{}_{}'.format(mode,datatype) 
    
    save_path=os.path.join(pic_save_path,'hp_count_{}.png'.format(file_name) )    
    plot_accs_zhexian(hp_counts,algs,'Number of Hard Positive Pairs', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),ylabel='',save_path=save_path)
    
    save_path=os.path.join(pic_save_path,'hn_count_{}.png'.format(file_name) )    
    plot_accs_zhexian(hn_counts,algs,'Number of Hard Negative Pairs', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),ylabel='',save_path=save_path)
    
    save_path=os.path.join(pic_save_path,'pos_mean_{}.png'.format(file_name) )    
    plot_accs_zhexian(hp_means,algs,'Mean Cosine Similarity of Positive Pairs', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),ylabel='',save_path=save_path)
    
    save_path=os.path.join(pic_save_path,'neg_mean_{}.png'.format(file_name) )    
    plot_accs_zhexian(hn_means,algs,'Mean Cosine Similarity of Negative Pairs', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),ylabel='',save_path=save_path)
    
    return hp_counts,hn_counts

def analyse4(cfg):
    algs=['FixMatch','MixMatch','OpenMatch','CCSSL','DCSSL']
    hp_counts1,hn_counts1=count_hardp_hardn(algs,mode='train',datatype='labeled')    
    hp_counts2,hn_counts2=count_hardp_hardn(algs,mode='test',datatype='')    
    hp_counts3,hn_counts3=count_hardp_hardn(algs,mode='train',datatype='unlabeled')
    # print(hp_counts1,hn_counts1)
    # print(hp_counts2,hn_counts2)
    # print(hp_counts3,hn_counts3)
    print("Analyse4 done!")
    
# 5. 分析泛化性能，||train-test||特征均值 方差 之差的绝对值。
def analyse5(cfg):
    algs=['FixMatch','MixMatch','OpenMatch','CCSSL','DCSSL']
    mean_diffs,var_diffs=[],[]
    for alg in algs:
        cfg.defrost()
        cfg.ALGORITHM.NAME=alg
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path) 
        gt1,pred1,feat1 = trainer.get_train_dl_data_pred_gt_feat()    
        gt2,pred2,feat2 = trainer.get_train_du_data_pred_gt_feat()  
        gt1=torch.cat([gt1,gt2],dim=0)
        pred1=torch.cat([pred1,pred2],dim=0)
        feat1=torch.cat([feat1,feat2],dim=0)
        
        gt3,pred3,feat3 = trainer.get_test_data_pred_gt_feat()  
        mean_diff=[0]*cfg.DATASET.NUM_CLASSES
        var_diff=[0]*cfg.DATASET.NUM_CLASSES        
        
        select_index=torch.nonzero(gt3==pred3,as_tuple=False).squeeze(1)
        feat3= feat3[select_index]
        pred3=pred3[select_index]
        gt3=gt3[select_index]
        
        for i in range(cfg.DATASET.NUM_CLASSES):
            select_index1=torch.nonzero(i==pred1,as_tuple=False).squeeze(1)
            select_index3=torch.nonzero(i==pred3,as_tuple=False).squeeze(1)
             
            embedding1 = feat1[select_index1]  
            mean1 = embedding1.mean(dim=0)
            var1 = embedding1.var(dim=0, unbiased=False)
            
            embedding3 = feat3[select_index3]  
            mean3 = embedding3.mean(dim=0)
            var3 = embedding3.var(dim=0, unbiased=False)
            
            mean_diff[i]=F.pairwise_distance(mean1.unsqueeze(0),mean3.unsqueeze(0),p=2)
            var_diff[i]=F.pairwise_distance(var1.unsqueeze(0),var3.unsqueeze(0),p=2)
            
        mean_diffs.append(mean_diff)
        var_diffs.append(var_diff)
        
    save_path=os.path.join(pic_save_path,'mean_diffs.png' )    
    plot_accs_zhexian(mean_diffs,algs,'mean difference w.r.t. test set', np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),
                      ylabel='feature mean distance',xlabel='class index',save_path=save_path)
    
    save_path=os.path.join(pic_save_path,'var_diffs.png' )    
    plot_accs_zhexian(var_diffs,algs,'cov difference w.r.t. test set', 
                      np.array([i+1 for i in range(cfg.DATASET.NUM_CLASSES)]),
                       ylabel='feature mean distance',xlabel='class index',save_path=save_path)

# 6. 是否有OOD样本的混淆矩阵
def analyse6(cfg):
    algs=['FixMatch'] #,'MixMatch','OpenMatch','CCSSL','DCSSL']
    mean_diffs,var_diffs=[],[]
    titles=['FixMatch' ]#,'(b) MixMatch','(c) OpenMatch','(d) CCSSL','(e) DCSSL']
    i=0
    for alg in algs:
        fusions=[]
        
        # clean
        cfg.defrost()
        cfg.ALGORITHM.NAME=alg
        cfg.DATASET.DU.OOD.INCLUDE_ALL=False
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)  
        test_fusion_matrix=FusionMatrix(cfg.DATASET.NUM_CLASSES) 
        gt3,pred3,_ = trainer.get_test_data_pred_gt_feat()        
        test_fusion_matrix.update(pred3, gt3)
        fusions.append(test_fusion_matrix.matrix)
        # === OOD
        cfg.defrost()
        cfg.ALGORITHM.NAME=alg
        cfg.DATASET.DU.OOD.RATIO=0.75
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)  
        test_fusion_matrix=FusionMatrix(cfg.DATASET.NUM_CLASSES) 
        gt3,pred3,_ = trainer.get_test_data_pred_gt_feat()        
        test_fusion_matrix.update(pred3, gt3)
        fusions.append(test_fusion_matrix.matrix)
        
        # # balanced
        # cfg.defrost()
        # cfg.ALGORITHM.NAME=alg
        # cfg.DATASET.DL.NUM_LABELED_HEAD=372
        # cfg.DATASET.DL.IMB_FACTOR_L=1
        # cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD=744
        # cfg.DATASET.DU.ID.IMB_FACTOR_UL=1 
        # cfg.DATASET.DU.OOD.INCLUDE_ALL=False
        # cfg.freeze()    
        # root_path=get_root_path(cfg)    
        # model_path=os.path.join(root_path,'models','best_model.pth')
        # trainer=build_trainer(cfg)
        # trainer.load_checkpoint(model_path)  
        # test_fusion_matrix=FusionMatrix(cfg.DATASET.NUM_CLASSES) 
        # gt3,pred3,_ = trainer.get_test_data_pred_gt_feat()        
        # test_fusion_matrix.update(pred3, gt3)
        # fusions.insert(0,test_fusion_matrix.matrix)
        
        save_path=os.path.join(pic_save_path,'fusion_matrix.jpg')
        plot_pd_heatmaps(fusions,save_path=save_path,r=1,c=2,title='',subtitles=['(a) Imbal.(IF-100)','(b) Imbal.(IF-100 w. OOD)'])
        i+=1
        
# 7. cosine 热力图case
def analyse7(cfg):
    cfg.defrost()
    cfg.ALGORITHM.NAME='DCSSL'
    cfg.freeze()    
    root_path=get_root_path(cfg)   
    pic_save_path= os.path.join(root_path, "a_pic")
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)
    model_path=os.path.join(root_path,'models','best_model.pth')
    trainer=build_trainer(cfg)
    trainer.load_checkpoint(model_path) 
    cos_sim = trainer.get_case_cosine()  
    plot_pd_heatmap(cos_sim,save_path=os.path.join(root_path,'cosine_sim.jpg'))

# 8. test data 置信度与距离类中心的欧式距离
def analyse8(cfg):
    
    algs=['FixMatch','MixMatch','OpenMatch','DCSSL']
    confidence,distance=[],[] 
    i=0
    titles=['(a) FixMatch','(b) MixMatch','(c) OpenMatch','(d) DeCAB (Ours)']
    for alg in algs: 
        cfg.defrost()
        cfg.ALGORITHM.NAME=alg
        cfg.freeze()    
        root_path=get_root_path(cfg)   
        pic_save_path= os.path.join(root_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)
        model_path=os.path.join(root_path,'models','best_model.pth')
        trainer=build_trainer(cfg)
        trainer.load_checkpoint(model_path)  
        # test_fusion_matrix=FusionMatrix(cfg.DATASET.NUM_CLASSES) 
        gt,pred,feat,conf = trainer.get_test_data_pred_gt_feat(return_confidence=True)     
        prototypes=torch.zeros(cfg.DATASET.NUM_CLASSES,feat.shape[-1]) 
        
        # tmp_gt=gt
        # tmp_feat=feat
        # correct_conf=conf
        correct_index=torch.nonzero(gt==pred,as_tuple=False).squeeze()  
        tmp_gt=gt[correct_index]
        tmp_feat=feat[correct_index]
        correct_conf=conf[correct_index]
        tmp_conf=[]
        tmp_dis=[]
        for c in range(cfg.DATASET.NUM_CLASSES): #[7,8,9]:  #
            select_index= torch.nonzero(tmp_gt == c, as_tuple=False).squeeze(1)
            prototypes[c] = tmp_feat[select_index].mean(dim=0) 
            c_conf=correct_conf[select_index]
            c_dist=1-F.cosine_similarity(tmp_feat[select_index], prototypes[c].unsqueeze(0),dim=1)
            # c_dist=F.pairwise_distance(tmp_feat[select_index], prototypes[c], p=2)
            tmp_conf.append(c_conf)
            tmp_dis.append(c_dist)
        tmp_conf=torch.cat(tmp_conf,dim=0)
        tmp_dis=torch.cat(tmp_dis,dim=0)
         
        # proto=prototypes[tmp_gt]
        # dist=F.pairwise_distance(tmp_feat, proto, p=2)
        distance.append(copy.deepcopy(tmp_dis.numpy()))
        confidence.append(copy.deepcopy(tmp_conf.squeeze().numpy()))
    save_path=os.path.join(pic_save_path,'conf_dist.jpg')
    plot_dots(x=confidence,y=distance,titles=titles,save_path=save_path,mode='c')
        
def analyse(cfg,analyse_type):
    
    if analyse_type==1:    
        analyse1(cfg)
    elif analyse_type==2:
        analyse2(cfg)
    elif analyse_type==3:
        analyse3(cfg)
    elif analyse_type==4:
        analyse4(cfg)
    elif analyse_type==5:
        analyse5(cfg)
    elif analyse_type==6:
        analyse6(cfg)
    elif analyse_type==7:
        analyse7(cfg)
    elif analyse_type==8:
        analyse8(cfg)
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
        
    
   
   