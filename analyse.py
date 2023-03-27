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

analyse_type_map={
    1:"the influence of single|dual branch with mixup",
    2:"the influence of r of ood",
    3:"the influence of ood_r in du with mixup",
    4:"the influence of different IF of DL and DU to OOD detector",
    5:"the tsne fig of feat with final model",
    6:"the tsne fig of feat with warmup model",
    7:"the influence of different IF of DL and DU to model",
    8:"the task of our work", # 纯画图
    9:"the prob of dual sample branch",# 纯画图
    10:"the ablation experiments of model",
    11:"analyse11_test_feature_TSNE_ablation",
    12:"problem  test data feature tsne",
    13:'the zhexian fig of generalization performance',
    14:"the influence of OOD to tail class",
    15:"the tail tsne of OOD to tail class",
    16:"zhexian of lambda pap"
    }
branches={
        # single branch
        'Single-Branch':[
         "RandomSampler",
         "RandomSampler_mixup",
         "ClassReversedSampler", 
         "ClassReversedSampler_mixup",
         "ClassAwareSampler",
         "ClassAwareSampler_mixup",
         "ClassBalancedSampler",
         "ClassBalancedSampler_mixup",
         ],
        # dual branch
        'Dual-Branch':[
         "RandomSampler-ClassReversedSampler",
         "RandomSampler-ClassReversedSampler_mixup",
         "RandomSampler_mixup-ClassReversedSampler_mixup",
         "RandomSampler-ClassAwareSampler",
         "RandomSampler-ClassAwareSampler_mixup",
         "RandomSampler_mixup-ClassAwareSampler_mixup",
         ]}
single_mixup_branches=[
         "RandomSampler_mixup",
         "ClassReversedSampler_mixup",
         "ClassBalancedSampler_mixup",
         "ClassAwareSampler_mixup",]
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
sampler=["RandomSampler","ClassAwareSampler","ClassBalancedSampler","ClassReversedSampler"]
dual_sampler=["ClassAwareSampler","ClassReversedSampler"]
if_dual_sampler_enable=[True,False]
if_mixup=[True,False]

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
       
def analyse1_branch_w_o_mixup(model,test_loader,cfg,criterion): 
    
    # ===== prepare pic path ====
    save_path=get_DL_dataset_alg_DU_dataset_path(cfg)
    branch_acc_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(branch_acc_save_path):
        os.mkdir(branch_acc_save_path)
    for branch in branches:
        group_accs , class_accs=[],[]
        for item in branches[branch]:
            model_path=get_root_path(cfg,Branch_setting=item)
            model_path=os.path.join(model_path,'models','best_model.pth')
            [model]=load_checkpoint(model, model_path)
            avg_loss, avg_acc,group_acc,class_acc=validate(test_loader, model, criterion,cfg=cfg,return_class_acc=True)
            group_accs.append(group_acc)
            class_accs.append(class_acc)
        # plot group acc 
        title=branch+' Group Acc' 
        name=cfg.ALGORITHM.NAME+'_{}_analysis_group_acc.jpg'.format(branch)
        save_path=os.path.join(branch_acc_save_path,name)
        plot_accs_zhexian(group_accs,branches[branch],title,["Many","Medium","Few"],save_path=save_path)
        # plot class acc 
        title=branch+' Class Acc'
        name=cfg.ALGORITHM.NAME+'_{}_analysis_class_acc.jpg'.format(branch)        
        save_path=os.path.join(branch_acc_save_path,name)        
        plot_accs_zhexian(class_accs,branches[branch],title,[i for i in range(cfg.DATASET.NUM_CLASSES)],save_path=save_path)
    print("== Branch mixup with ID/OOD data pic have been saved in {}".format(branch_acc_save_path))
    print("== Type 1 analysis has been done! ==")
    return

def analyse2_ood_r(cfg): 
    # ood data parent path
    save_path=get_DL_dataset_path(cfg)
                
    final_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(final_save_path):
        os.mkdir(final_save_path) 
        
     
    # baseline 单独计算
    # retain_algors=copy.deepcopy(algorithms)
    # retain_algors.remove("Supervised")
    # for dataset in ood_dataset:
    #     # item : the name of ood dataset
    #     avg_accs=[]
    #     model_path=get_root_path(cfg,ood_dataset=dataset,ood_r=0,dual_sampler_enable=True)
    #     model_path=os.path.join(model_path,'models','best_model.pth')
    #     cfg.defrost() 
    #     cfg.ALGORITHM.NAME='baseline' 
    #     cfg.DATASET.DU.OOD.RATIO=0
    #     cfg.freeze() 
    #     trainer=build_trainer(cfg)
    #     trainer.load_checkpoint( model_path)
    #     _, baseline_avg_acc=trainer.evaluate()

    #     for alg in retain_algors:
    #         if alg=='MOOD':continue
    #         accs=[]
    #         for r in ood_r:
    #             model_path=get_root_path(cfg,ood_dataset=dataset,algorithm=alg,ood_r=r)                 
    #             model_path=os.path.join(model_path,'models','best_model.pth')
    #             cfg.defrost() 
    #             cfg.ALGORITHM.NAME=alg
    #             cfg.DATASET.DU.OOD.RATIO=r
    #             cfg.freeze() 
    #             trainer=build_trainer(cfg)
    #             trainer.load_checkpoint(model_path)
    #             avg_acc=trainer.get_test_best()
    #             accs.append(100*avg_acc) 
    #     # 每个ood数据集的不同比例的r
    #     # plot group acc 
    #         avg_accs.append(accs)
    #     avg_accs.insert(0,[100*baseline_avg_acc]*len(ood_r))
    #     print(avg_accs)
    #     title='Test Avg Acc' 
    #     name='motivation.jpg'
    #     save_path=os.path.join(final_save_path,name)
    algorithms[-1]='MOOD (ours)'
   
    test_group_accs=[
        [[79.36666666666667, 79.36666666666667, 79.36666666666667],  
         [92.8, 89.16666666666666, 90.26666666666668], 
         [91.83333333333333, 90.39999999999999, 92.8],
         [91.73333333333335, 90.0, 88.26666666666667],
         [89.96666666666667, 87.46666666666665, 88.06666666666668],
         [92.33,90.57,87.87]],
        [[57.36666666666667, 57.36666666666667, 57.36666666666667] ,
         [66.73333333333332, 62.5, 59.5], 
         [60.46666666666667, 64.56666666666666, 63.43333333333333], 
         [65.23333333333333, 63.66666666666667, 64.9], 
         [62.86666666666666, 62.866666666666674, 56.56666666666666],
         [69.73,69.57, 63.30 ]], 
        [[53.075, 53.075, 53.075],  
         [51.0, 39.95, 32.2], 
         [50.4, 33.0, 35.325],
         [61.0, 50.2, 47.35],
         [49.62500000000001, 41.300000000000004, 23.85],
         [65.22 ,64.25 ,59.52]]]
    algorithms[0]="Supervised\nTraining"
    # plot_accs_zhexian(test_group_accs, algorithms, "", np.array(ood_r),xlabel="R",save_path=final_save_path)
    labeled_dist=np.array([1500, 899, 539, 323, 193, 116, 69, 41, 25, 15]) 
    unlabeled_dist=np.array([3000, 1798, 1078, 646, 387, 232, 139, 83, 50, 30])
    save_path=os.path.join(save_path,"openset-lt-ssl-task.jpg") 
    final_save_path1=os.path.join(final_save_path,"openset-lt-ssl-task.png") 
    plot_task_together(labeled_dist, unlabeled_dist,test_group_accs[-1], algorithms,'',np.array(ood_r),xlabel=r"${R}$",save_path=final_save_path1)
    
    final_save_path1=os.path.join(final_save_path,"group_acc_over_r.png") 
    plot_r_group_together(test_group_accs, algorithms,'',np.array(ood_r),xlabel="$R$",save_path=final_save_path1)
        
    print("== Different ratio of OOD data pic have been saved in {}".format(final_save_path))
    print("== Type 2 analysis has been done! ==")
    return 

def analyse3_mixup_ID_OOD(cfg):
    items=['DUAL_BRANCH','MIXUP','OOD_DETECTION','PAP_LOSS']
    IF=100
    R=0.5
    for item in items:
        cfg.defrost() 
        cfg.ALGORITHM.NAME='MOOD'
        cfg.DATASET.DL.IMB_FACTOR_L=IF
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        cfg.DATASET.DU.OOD.RATIO=R
        cfg.ALGORITHM.ABLATION.ENABLE=True 
        cfg.ALGORITHM.ABLATION[item]=True
        cfg.freeze()  
        # ood data parent path 
        save_path=get_root_path(cfg)                    
        pic_save_path= os.path.join(save_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)     
        trainer=build_trainer(cfg)
        model_path=get_root_path(cfg,algorithm=cfg.ALGORITHM.NAME,imb_factor_l=IF,imb_factor_ul=IF,ood_r=R)
        model_path=os.path.join(model_path,'models','best_model.pth')
        trainer.load_checkpoint(model_path)    
        (labeled_feat,labeled_y,_),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,_)= trainer.extract_feature()  
        # 画labeled tsne图
        plot_feat_tsne(labeled_feat,labeled_y,'Labeled features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"dl_feat_tsne_best.jpg"))
        # 画unlabeled tsne图        
        plot_feat_tsne(unlabeled_feat,unlabeled_y,'Unlabeled features TSNE',cfg.DATASET.NUM_CLASSES+1,save_path=os.path.join(pic_save_path,"du_feat_tsne_best.jpg"))
        # 画test tsne图
        plot_feat_tsne(test_feat,test_y,'Test features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"test_feat_tsne_.jpg"))
     
     
        print("== Mixup with different ratio of OOD data pic have been saved in {}".format(pic_save_path))
    print("== Type 3 analysis has been done! ==")
 
def ood_detect_test(model,domain_trainloader,unlabeled_trainloader,cfg):
    feat,feat_y=prepare_feat(model, domain_trainloader)
    model.eval()
    num_classes=cfg.DATASET.NUM_CLASSES
    total_c=np.zeros(num_classes)
    correct_c=np.zeros(num_classes)
    total_id=0
    total_ood=0
    correct_id=0
    correct_ood=0
    with torch.no_grad():
        for i,data in enumerate(unlabeled_trainloader):
            x=data[0] 
            if len(x)==2:
                x=x[0]
            x=x.cuda()
            class_labels=data[1]
            domain_labels=1-torch.eq(class_labels,-1).float() 
            class_labels=class_labels.cuda()
            domain_labels=domain_labels.cuda()
            _,feature=model(x,return_encoding=True) 
            id_mask,ood_mask=knn_ood_detect(feat, feat_y, feature, cfg)
            c_id=torch.sum(domain_labels)
            c_ood=domain_labels.size(0)-torch.sum(domain_labels)
            id_correct_mask=(torch.eq(domain_labels,id_mask)*torch.eq(domain_labels,torch.ones_like(domain_labels).cuda())).float()
            ood_correct_mask=(torch.eq(domain_labels,id_mask)*torch.eq(domain_labels,torch.zeros_like(domain_labels).cuda())).float()
            for i in range(num_classes):
                total_c[i]+=torch.sum(torch.eq(class_labels,i))
                correct_c[i]+=torch.sum(torch.eq(class_labels,i)*id_correct_mask)
            total_id += c_id
            total_ood += c_ood
            correct_id+= torch.sum(id_correct_mask)
            correct_ood+= torch.sum(ood_correct_mask)
            
    tpr= (correct_id)/(total_id)
    tnr=(correct_ood)/(total_ood) 
    class_tpr=correct_c/total_c
    return tpr,tnr,class_tpr

def analyse4_IF_to_ood_detector(model, cfg):
    # domain OOD:0.5
    
    save_path=get_DL_dataset_path(cfg)                
    acc_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(acc_save_path):
        os.mkdir(acc_save_path)  
    tprs_type=["class_tprs","id_tprs","ood_tprs","total_tprs"] 
    tnrs_type=["class_tnrs","id_tnrs","ood_tnrs","total_tnrs"] 
    dataset=cfg.DATASET.DU.OOD.DATASET
    tprs=[]
    tnrs=[]
    class_tprs=[]
    ood_r=0.5
    for imb_f in IF:
        cfg.defrost()
        cfg.DATASET.DL.IMB_FACTOR_L=imb_f
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=imb_f
        cfg.DATASET.DU.OOD.RATIO=ood_r
        cfg.freeze()        
        dataloaders=build_dataloader(cfg)
        domain_trainloader=dataloaders[0]
        unlabeled_trainloader=dataloaders[2]
        model_path=get_root_path(cfg,imb_factor_l=imb_f,imb_factor_ul=imb_f,ood_r=0.5,algorithm="Ours") 
        model_path=os.path.join(model_path,'models','epoch_200.pth')
        [model]=load_checkpoint(model, model_path)
        results=ood_detect_test(model,domain_trainloader,unlabeled_trainloader,cfg) 
        tprs.append(results[0])
        tnrs.append(results[1])
        class_tprs.append(results[2])
    title='OOD '+dataset+' TPR' 
    name='{}_{}_warmup_epoch_200_IF_ood_detect_class_tprs_best.jpg'.format(cfg.ALGORITHM.NAME,cfg.DATASET.NAME)
    save_path=os.path.join(acc_save_path,name)
    IF_legend=["IF10","IF50","IF100"]
    plot_multi_bars(class_tprs,labels=[i for i in range(cfg.DATASET.NUM_CLASSES)],title="Class TPRs", legend=IF_legend,xlabel="Class ID",ylabel="TPR",save_path=save_path)
    print(tprs)
    print(tnrs)
    print("==  Different IF to ood detector pic have been saved in {}".format(acc_save_path))
    print("== Type 4 analysis has been done! ==")
    return 

def analyse5_feature_TSNE_final_model(cfg): 
    IF=100
    R=0.5
    algs=['MOOD']
    for alg in algs:
        cfg.defrost() 
        cfg.ALGORITHM.NAME=alg
        cfg.DATASET.DL.IMB_FACTOR_L=IF
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        cfg.DATASET.DU.OOD.RATIO=R
        cfg.freeze()  
        # ood data parent path
        save_path=get_DL_dataset_alg_DU_dataset_OOD_path(cfg)
                    
        pic_save_path= os.path.join(save_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path)     
        trainer=build_trainer(cfg)
        model_path=get_root_path(cfg,algorithm=cfg.ALGORITHM.NAME,imb_factor_l=IF,imb_factor_ul=IF,ood_r=R)
        model_path=os.path.join(model_path,'models','best_model.pth')
        trainer.load_checkpoint(model_path)    
        (labeled_feat,labeled_y,_),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,_)= trainer.extract_feature()  
        # 画labeled tsne图
        plot_feat_tsne(labeled_feat,labeled_y,'Labeled features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"dl_feat_tsne_best.jpg"))
        # 画unlabeled tsne图        
        plot_feat_tsne(unlabeled_feat,unlabeled_y,'Unlabeled features TSNE',cfg.DATASET.NUM_CLASSES+1,save_path=os.path.join(pic_save_path,"du_feat_tsne_best.jpg"))
        # 画test tsne图
        plot_feat_tsne(test_feat,test_y,'Test features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"test_feat_tsne_.jpg"))
    # pred_id=trainer.ood_masks
    # pred_ood=trainer.ood_masks
    # 分对的ID用绿色三角形表示，分对的OOD用绿色圆形表示
    # 分错的ID用不同颜色的×形表示，分错的OOD用红色的圆形表示
    # 都为0的用灰色表示，表示没用到这部分数据
    # plot_ood_detect_feat(unlabeled_feat,unlabeled_y,pred_id,pred_ood)
    return 

def analyse6_feature_TSNE_warmup_model(cfg): 
   
    IF=100
    R=0.5 
    cfg.defrost() 
    cfg.ALGORITHM.NAME='MOOD'
    cfg.DATASET.DL.IMB_FACTOR_L=IF
    cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
    cfg.DATASET.DU.OOD.RATIO=R
    cfg.freeze()  
    # ood data parent path
    save_path=get_DL_dataset_alg_DU_dataset_OOD_path(cfg)
                
    pic_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)     
    trainer=build_trainer(cfg)
    model_path=get_root_path(cfg,algorithm=cfg.ALGORITHM.NAME,imb_factor_l=IF,imb_factor_ul=IF,ood_r=R)
    model_path=os.path.join(model_path,'models','best_model.pth')
    trainer.load_checkpoint(model_path)    
    (labeled_feat,labeled_y,_),(unlabeled_feat,unlabeled_y,unlabeled_idx),(test_feat,test_y,_)= trainer.extract_feature()  
    # 画labeled tsne图
    plot_feat_tsne(labeled_feat,labeled_y,'Labeled features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"dl_feat_tsne_warmup.jpg"))
    # 画unlabeled tsne图        
    plot_feat_tsne(unlabeled_feat,unlabeled_y,'Unlabeled features TSNE',cfg.DATASET.NUM_CLASSES+1,save_path=os.path.join(pic_save_path,"du_feat_tsne_warmup.jpg"))
    # 画test tsne图
    plot_feat_tsne(test_feat,test_y,'Test features TSNE',cfg.DATASET.NUM_CLASSES,save_path=os.path.join(pic_save_path,"test_feat_tsne_warmup.jpg"))
    # pred_id=trainer.ood_masks
    # pred_ood=trainer.ood_masks
    # 分对的ID用绿色三角形表示，分对的OOD用绿色圆形表示
    # 分错的ID用不同颜色的×形表示，分错的OOD用红色的圆形表示
    # 都为0的用灰色表示，表示没用到这部分数据
    # plot_ood_detect_feat(unlabeled_feat,unlabeled_y,pred_id,pred_ood)
    return 

def analyse7_IF(model,test_loader,cfg): 
    # ood data parent path
    save_path=get_DL_dataset_path(cfg)
                
    ratio_acc_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(ratio_acc_save_path):
        os.mkdir(ratio_acc_save_path) 
        
    algorithms=[
    "baseline",
    "FixMatch", 
    "MixMatch",
    # "PseudoLabel",
    # "MTCF",
    "DASO",
    "CReST",
    "MOOD"
    # "DS3F"
    ]
    # baseline 单独计算    
    accs=[]           
    for alg in algorithms:
        avg_accs=[]
        for if_ in IF: 
            if alg=="baseline":
                model_path=get_root_path(cfg,algorithm=alg,imb_factor_l=if_,imb_factor_ul=if_,ood_r=0.0)
            else:
                model_path=get_root_path(cfg,algorithm=alg,imb_factor_l=if_,imb_factor_ul=if_,ood_r=0.5)
            model_path=os.path.join(model_path,'models','best_model.pth')
            [model]=load_checkpoint(model, model_path)
            _, avg_acc,_,_=validate(test_loader, model, criterion,cfg=cfg,return_class_acc=True)
            avg_accs.append(copy.deepcopy(avg_acc))
        accs.append(copy.deepcopy(avg_accs))   
    dataset=cfg.DATASET.NAME
    title= dataset +' DU-OOD-R-0.5 Avg Acc' 
    name=dataset+'_different_IF_{}_avg_acc.jpg'.format(dataset)
    save_path=os.path.join(ratio_acc_save_path,name)
    plot_accs_zhexian(accs, algorithms,title,np.array(IF),ylabel="Accuracy",xlabel="IF",save_path=save_path)
    print("== Different IF of DL and DU data pic have been saved in {}".format(ratio_acc_save_path))
    print("== Type 7 analysis has been done! ==")
    return 

def analyse8_task(labeled_trainloader,unlabeled_trainloader,cfg):
     # ood data parent path
    save_path=get_DL_dataset_path(cfg)
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
    labeled_dist=np.array([1500, 899, 539, 323, 193, 116, 69, 41, 25, 15]) 
    unlabeled_dist=np.array([3000, 1798, 1078, 646, 387, 232, 139, 83, 50, 30])
    save_path=os.path.join(save_path,"task.jpg")
    plot_task_stack_bar(labeled_dist,unlabeled_dist, save_path,title="Data Distribution")
    # plot_task_bar(labeled_dist,unlabeled_dist, save_path)
    print("== Type 8 analyse task has been done! ==")
    return 

def analyse9_sample_prob(cfg):
    labeled_dist=np.array([1500, 899, 539, 323, 193, 116, 69, 41, 25, 15])  
    random_prob=labeled_dist/labeled_dist.sum()
    
    n_max=max(labeled_dist)
    per_cls_weights = n_max/np.array(labeled_dist)
    reversed_prob = per_cls_weights/per_cls_weights.sum()
     # ood data parent path
    save_path=os.path.join(get_DL_dataset_path(cfg),"a_pic")
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
    save_path1=os.path.join(save_path,"ramdom_sample_prob.jpg")
    plot_bar(random_prob, save_path1)
    
    save_path2=os.path.join(save_path,"reversed_sample_prob.jpg")
    plot_bar(reversed_prob, save_path2)
    print("== Type 9 analyse task has been done! ==")
    return

def analyse10_ablation(cfg):
    save_path=os.path.join(get_DL_dataset_path(cfg),"a_pic")
    if not os.path.exists(save_path):
        os.mkdir(save_path) 
    avg_accs=[]
    group_accs,class_accs=[],[]
    legends=['ID-1','ID-2','ID-3','ID-4','ID-5']
    variables=["A_Naive","DUAL_BRANCH","MIXUP", "OOD_DETECTION","PAP_LOSS"]
    IF=100
    R=0.5
    for i,item in enumerate(variables):
        cfg.defrost()
        cfg.ALGORITHM.ABLATION.ENABLE=True
        cfg.ALGORITHM.NAME='MOOD'
        cfg.DATASET.DL.IMB_FACTOR_L=IF
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        cfg.DATASET.DU.OOD.RATIO=R
        if i>0:
            cfg.ALGORITHM.ABLATION[item]=True       
        cfg.freeze()  
        trainer=build_trainer(cfg)
        model_path=get_root_path(cfg,algorithm="MOOD",imb_factor_l=100,imb_factor_ul=100,ood_r=0.5)
        model_path=os.path.join(model_path,'models','best_model.pth')
        trainer.load_checkpoint(model_path)
        _, avg_acc ,group_acc,class_acc= trainer.evaluate(return_class_acc=True,return_group_acc=True)  
        avg_accs.append(copy.deepcopy(avg_acc))
        group_accs.append(copy.deepcopy(group_acc))
        class_accs.append(copy.deepcopy(class_acc))
    save_path1=os.path.join(save_path,"ablation_group_acc.jpg")
    plot_multi_bars(group_accs,  ["Many:{0,1,2}","Medium:{3,4,5}","Few:{6,7,8,9}"],xlabel="",ylabel="Accuracy",legend=legends,save_path=save_path1)
    save_path2=os.path.join(save_path,"ablation_class_acc.jpg")
    plot_multi_bars(class_accs, [i for i in range(cfg.DATASET.NUM_CLASSES)],xlabel="Class ID",ylabel="Accuracy",legend=legends,save_path=save_path2)
    print(avg_accs)
    print(group_accs)


def analyse11_test_feature_TSNE_ablation(cfg): 
    legends=['ID-1','ID-2','ID-3','ID-4','ID-5']
    variables=["A_Naive","DUAL_BRANCH","MIXUP", "OOD_DETECTION","PAP_LOSS"]
    IF=100
    R=0.5
    alg='MOOD'
    gts,preds,feats=[],[],[]
    # save_path=get_DL_dataset_path(cfg) 
    # final_save_path= os.path.join(save_path, "a_pic")
    # if not os.path.exists(final_save_path):
        # os.mkdir(final_save_path) 
    for i,item in enumerate(variables):
        cfg.defrost()
        cfg.ALGORITHM.ABLATION.ENABLE=True
        cfg.ALGORITHM.NAME=alg
        cfg.DATASET.DL.IMB_FACTOR_L=IF
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        cfg.DATASET.DU.OOD.RATIO=R
        if i>0:
            cfg.ALGORITHM.ABLATION[item]=True       
        cfg.freeze()  
        # if i==0:continue
        # trainer=build_trainer(cfg)         
        model_path=get_root_path(cfg)
        pic_save_path= os.path.join(model_path, "a_pic")
        if not os.path.exists(pic_save_path):
            os.mkdir(pic_save_path) 
        # model_path=os.path.join(model_path,'models','best_model.pth')
        # trainer.load_checkpoint(model_path)
        # gt,pred,feat = trainer.get_test_data_pred_gt_feat()   
        # # plot_problem_feat_tsne(gt,pred,feat,num_classes=cfg.DATASET.NUM_CLASSES,alg=alg,
        # #             save_path=os.path.join(pic_save_path,'{}_feature_tsne.png'.format(item)))
        # gts.append(gt)
        # preds.append(pred)
        # feats.append(feat)
    # gts=torch.cat(gts,dim=0)
    # preds=torch.cat(preds,dim=0)
    # feats=torch.cat(feats,dim=0)
    # subtitles=[' ',' ',' ',' ']
    subtitles=['(a) w. DB','(b) w. FU','(c) w. FO',"(d) w. PaP"]
    plot_ablation_feat_tsne(gts,preds,feats,num_classes=cfg.DATASET.NUM_CLASSES,alg=alg,
                    subtitles=subtitles,
                    save_path=pic_save_path)    
    return

def analyse12_test_data_feature_tsne(cfg):
    IF=100
    R=0.5
    algs=['baseline','FixMatch','MixMatch','CReST','DASO','MOOD']#
    titles=['(a) ','(b) ','(c) ','(d) ','(e) ','(f) ']
    save_path=get_DL_dataset_path(cfg) 
    final_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(final_save_path):
        os.mkdir(final_save_path)
    gts,preds,feats=[],[],[]
    i=0
    for alg in algs:
        # cfg.defrost() 
        # cfg.ALGORITHM.NAME=alg
        # cfg.DATASET.DL.IMB_FACTOR_L=IF
        # cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        # cfg.DATASET.DU.OOD.RATIO=R if alg!='baseline' else 0.0
        # cfg.freeze()           
        # trainer=build_trainer(cfg)
        # model_path=get_root_path(cfg)
        # model_path=os.path.join(model_path,'models','best_model.pth')
        # trainer.load_checkpoint(model_path)
        # gt,pred,feat = trainer.get_test_data_pred_gt_feat()     
        # gts.append(gt)
        # preds.append(pred)
        # feats.append(feat)
        # # 画labeled tsne图
        if alg=='baseline': 
            titles[i]=titles[i]+'Supervised Training'
        else:
            titles[i]=titles[i]+alg
            if alg=='MOOD':
                titles[i]+=' (ours)'
        i+=1
        # save_path=os.path.join(final_save_path,'test_feat_{}.png'.format(alg) )
        # plot_problem_feat_tsne(gt,pred,feat,num_classes=cfg.DATASET.NUM_CLASSES,
        #                 title=titles[i],      
        #             save_path=save_path)
         
    save_path=os.path.join(final_save_path,'test_feat.png' )
    plot_problem_feat_tsne(gts,preds,feats,num_classes=cfg.DATASET.NUM_CLASSES,
                        titles=titles,      
                    save_path=save_path)
    
    return


def analyse13_test_data_feature_mean_std(cfg):
    IF=100
    R=0.5
    algs=['baseline','FixMatch','MixMatch','CReST','DASO','MOOD']
    means=[]
    vars_=[]
    test_gt_vars=[]
    for alg in algs:
        cfg.defrost() 
        cfg.ALGORITHM.NAME=alg
        cfg.DATASET.DL.IMB_FACTOR_L=IF
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=IF
        cfg.DATASET.DU.OOD.RATIO=R if alg!='baseline' else 0.0
        cfg.freeze()           
        trainer=build_trainer(cfg)
        model_path=get_root_path(cfg)
        model_path=os.path.join(model_path,'models','best_model.pth')
        trainer.load_checkpoint(model_path)
        test_gt,test_pred,test_feat = trainer.get_test_data_pred_gt_feat()   
        train_gt,train_pred,train_feat = trainer.get_train_dl_data_pred_gt_feat()     
        gt_mean_cha,gt_var_cha,pred_mean_cha,pred_var_cha,test_gt_var=compute_mean_std_cha(train_gt,train_pred,train_feat,test_gt,test_pred,test_feat)
        # 画labeled tsne图
        # plot_feat_cha_zhexian(gt_mean_cha,gt_var_cha,pred_mean_cha,pred_var_cha,num_classes=cfg.DATASET.NUM_CLASSES,alg=alg,
        #             save_path=os.path.join(pic_save_path))
        # means.append(pred_mean_cha)
        # vars_.append(pred_var_cha)
        test_gt_vars.append(test_gt_var)
    
    pic_save_path= get_DL_dataset_alg_DU_dataset_OOD_path(cfg)
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path) 
    # plot_feat_diff(means,algs,num_classes=cfg.DATASET.NUM_CLASSES,
    #                 save_path=os.path.join(pic_save_path,'a_pic','test_data_mean_diff_pred.jpg'))
    # plot_feat_diff(vars_,algs,num_classes=cfg.DATASET.NUM_CLASSES,
    #                 save_path=os.path.join(pic_save_path,'a_pic','test_data_var_diff_pred.jpg'))
    plot_feat_diff(test_gt_vars,algs,num_classes=cfg.DATASET.NUM_CLASSES,
                    save_path=os.path.join(pic_save_path,'a_pic','test_data_var_gt.jpg'))
    
    return

def analyse14_ood_2_tail(cfg): 
    # ood data parent path
    save_path=get_DL_dataset_path(cfg)
                
    final_save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(final_save_path):
        os.mkdir(final_save_path)      
    # baseline 单独计算
    retain_algors=copy.deepcopy(algorithms)
    retain_algors.remove("Supervised")
    for dataset in ood_dataset:
        # item : the name of ood dataset
        avg_accs,test_group_accs,test_class_accs=[],[[],[],[]],[]
        model_path=get_root_path(cfg,ood_dataset=dataset,ood_r=0,dual_sampler_enable=True)
        model_path=os.path.join(model_path,'models','best_model.pth')
        cfg.defrost() 
        cfg.ALGORITHM.NAME='baseline' 
        cfg.DATASET.DU.OOD.RATIO=0
        cfg.freeze() 
        trainer=build_trainer(cfg)
        trainer.load_checkpoint( model_path)
        _, baseline_avg_acc,baseline_test_group_acc,baseline_test_class_acc=trainer.evaluate(return_class_acc=True, return_group_acc=True)
        for alg in ['MOOD']:
            accs=[]
            group_accs,class_accs=[[],[],[]],[]
            for r in ood_r:
                model_path=get_root_path(cfg,ood_dataset=dataset,algorithm=alg,ood_r=r)                 
                model_path=os.path.join(model_path,'models','best_model.pth')
                cfg.defrost() 
                cfg.ALGORITHM.NAME=alg
                cfg.DATASET.DU.OOD.RATIO=r
                cfg.freeze() 
                trainer=build_trainer(cfg)
                trainer.load_checkpoint( model_path)
                _, avg_acc,test_group_acc,test_class_acc=trainer.evaluate(return_class_acc=True, return_group_acc=True)
                # avg_acc=trainer.get_test_best()
                accs.append(100*avg_acc) 
                for i in range(3):                    
                    group_accs[i].append(100*test_group_acc[i])
                class_accs.append(list(100*np.array(test_class_acc)))
        # 每个ood数据集的不同比例的r
        # plot group acc 
            avg_accs.append(accs)
            for i in range(3):
                test_group_accs[i].append(group_accs[i])
                 
            test_class_accs.append(class_accs)
        
        for i in range(3): 
            test_group_accs[i].insert(0,[100*baseline_test_group_acc[i]]*len(ood_r))
        avg_accs.insert(0,[100*baseline_avg_acc]*len(ood_r))        
        test_class_accs.insert(0,[list(100*np.array(baseline_test_class_acc))]*len(ood_r))
        print("avg acc:")
        print(avg_accs)
        print("Tail group avg acc:")
        print(test_group_accs)
        print("Class acc:")
        print(test_class_accs)
        # save_path=os.path.join(save_path,"ood2tail.jpg") 
        # plot_accs_zhexian(avg_accs, algorithms, "", np.array(ood_r),xlabel="R",save_path=save_path)
    # final_save_path=os.path.join(final_save_path,"ood2tail.jpg") 
    # test_group_accs=[[ 53.075,53.075, 53.075], 
    #                  [ 51.0, 39.95, 32.2],
    #                  [ 50.4, 33.0, 35.325],
    #                  [61.0, 50.2, 47.35],
    #                  [49.62500000000001, 41.300000000000004, 23.85], 
    #                  [70.0, 60.35, 58.175]]
    # test_class_accs=[[[90.5, 90.0, 57.599999999999994, 78.8, 53.2, 40.1, 68.7, 65.3, 23.599999999999998, 54.7], [90.5, 90.0, 57.599999999999994, 78.8, 53.2, 40.1, 68.7, 65.3, 23.599999999999998, 54.7], [90.5, 90.0, 57.599999999999994, 78.8, 53.2, 40.1, 68.7, 65.3, 23.599999999999998, 54.7]], [[97.1, 97.39999999999999, 83.89999999999999, 77.3, 71.5, 51.4, 63.800000000000004, 49.1, 47.5, 43.6], [96.39999999999999, 91.5, 79.60000000000001, 77.3, 60.8, 49.4, 52.300000000000004, 43.6, 7.000000000000001, 56.89999999999999], [93.8, 98.5, 78.5, 69.69999999999999, 67.80000000000001, 41.0, 47.3, 15.8, 45.7, 20.0]], [[93.89999999999999, 98.2, 83.39999999999999, 68.4, 64.4, 48.6, 67.9, 73.3, 34.5, 25.900000000000002], [96.39999999999999, 98.4, 76.4, 69.19999999999999, 71.2, 53.300000000000004, 63.3, 47.9, 10.299999999999999, 10.5], [96.2, 98.4, 83.8, 70.6, 68.8, 50.9, 60.6, 55.2, 22.7, 2.8000000000000003]], [[91.7, 97.7, 85.8, 78.2, 72.3, 45.2, 80.60000000000001, 54.800000000000004, 59.4, 49.2], [93.7, 99.3, 77.0, 70.89999999999999, 73.8, 46.300000000000004, 80.0, 47.9, 41.6, 31.3], [92.60000000000001, 97.5, 74.7, 58.9, 71.2, 64.60000000000001, 62.4, 55.300000000000004, 39.900000000000006, 31.8]], [[95.7, 96.3, 77.9, 55.900000000000006, 76.0, 56.699999999999996, 75.5, 53.0, 31.0, 39.0], [96.1, 97.5, 68.8, 61.9, 68.10000000000001, 58.599999999999994, 66.2, 43.8, 21.4, 33.800000000000004], [91.9, 96.7, 75.6, 65.7, 62.9, 41.099999999999994, 39.5, 37.4, 5.8999999999999995, 12.6]], [[90.0, 91.4, 69.3, 62.8, 66.5, 67.4, 75.3, 83.89999999999999, 57.3, 63.5], [92.2, 97.2, 81.8, 72.3, 77.2, 59.699999999999996, 76.0, 60.699999999999996, 51.1, 53.6], [93.30000000000001, 95.39999999999999, 81.0, 54.300000000000004, 74.8, 64.9, 73.5, 66.7, 23.799999999999997, 68.7]]]
    # algorithms[0]="Supervised Training"
    # final_save_path=os.path.join(final_save_path,"ood2head.jpg") 
    # test_group_accs=[[79.36666666666667,79.36666666666667, 79.36666666666667],
    #                  [92.8,89.16666666666666,90.26666666666668], 
    #                  [91.83333333333333,90.39999999999999,92.8],
    #                  [91.73333333333335,90.0,88.26666666666667],
    #                  [89.96666666666667, 87.46666666666665,88.06666666666668],
    #                  [83.56666666666666,90.4,89.9,]]
    # plot_accs_zhexian(test_group_accs, algorithms, "", np.array(ood_r),xlabel="R",save_path=final_save_path)
    
    print("== Different ratio of OOD data 2 tail pic have been saved in {}".format(final_save_path))
    print("== Type 14 analysis has been done! ==")
    return 
def analyse15_ood_2_tail_tsne(cfg):
    save_path=get_DL_dataset_path(cfg) 
    save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(save_path):
        os.mkdir(save_path)      
    # baseline 单独计算
    retain_algors=copy.deepcopy(algorithms)
    retain_algors.remove("baseline") 
    # item : the name of ood dataset
    avg_accs,test_group_accs,test_class_accs=[],[],[]
    model_path=get_root_path(cfg,ood_r=0,dual_sampler_enable=True)
    model_path=os.path.join(model_path,'models','best_model.pth')
    cfg.defrost() 
    cfg.ALGORITHM.NAME='baseline' 
    cfg.DATASET.DU.OOD.RATIO=0
    cfg.freeze() 
    trainer=build_trainer(cfg)
    trainer.load_checkpoint( model_path)
    gt,pred,feat = trainer.get_test_data_pred_gt_feat()  
    final_save_path=os.path.join(save_path,"baseline-tail_TSNE.jpg")
    plot_tail_feat_tsne(gt,pred,feat,num_classes=cfg.DATASET.NUM_CLASSES,alg="baseline",
                        save_path=final_save_path)
    # if_=100
    for alg in retain_algors:
        accs=[]
        group_accs,class_accs=[],[]
        for r in ood_r:
            model_path=get_root_path(cfg,algorithm=alg,ood_r=r)                 
            model_path=os.path.join(model_path,'models','best_model.pth')
            cfg.defrost() 
            cfg.ALGORITHM.NAME=alg
            cfg.DATASET.DU.OOD.RATIO=r
            cfg.freeze() 
            trainer=build_trainer(cfg)
            trainer.load_checkpoint( model_path)
            gt,pred,feat = trainer.get_test_data_pred_gt_feat()  # avg_acc=trainer.get_test_best()
            # 画labeled tsne图
            final_save_path=os.path.join(save_path,"{}-r-{}-tail_TSNE.jpg".format(alg,r))
            plot_tail_feat_tsne(gt,pred,feat,num_classes=cfg.DATASET.NUM_CLASSES,alg=alg,
                        save_path=final_save_path)
    print("analyse 15 is done! save in {}".format(save_path))

def analyse16_lambda_pap(cfg):
    x=[0.30, 0.25 ,0.20, 0.15, 0.10 ,0.05] 
    y=[71.59, 73.61 ,73.74, 73.18 ,73.10, 72.7]
    save_path=get_DL_dataset_path(cfg) 
    save_path= os.path.join(save_path, "a_pic")
    if not os.path.exists(save_path):
        os.mkdir(save_path)      
    save_path=os.path.join(save_path,'lambda_pap.png')
    plot_zhexian(x, y, xlabel='$\lambda_{pap}$', ylabel='Accuracy (%)',save_path=save_path)
    
def analyse(cfg,analyse_type):
    
    if analyse_type==1:   
        test_loader=build_test_dataloader(cfg)
        analyse1_branch_w_o_mixup(model,test_loader,cfg,criterion)
    elif analyse_type==2:
        # 不同 OOD R [0.0 0.25 0.5 0.75 1.0]
        # test_loader=build_test_dataloader(cfg)
        analyse2_ood_r(cfg)
    elif analyse_type==3:         
        analyse3_mixup_ID_OOD(cfg)
    elif analyse_type==4:
        analyse4_IF_to_ood_detector(model,cfg)
    elif analyse_type==5:
        analyse5_feature_TSNE_final_model(cfg)
    elif analyse_type==6:
        analyse6_feature_TSNE_warmup_model(cfg)
    elif analyse_type==7:
        test_loader=build_test_dataloader(cfg)
        analyse7_IF(model, test_loader, cfg)
    elif analyse_type==8: 
        analyse8_task(labeled_trainloader=None, unlabeled_trainloader=None, cfg=cfg)
    elif analyse_type==9: 
         analyse9_sample_prob(cfg)
    elif analyse_type==10:  
         analyse10_ablation(cfg)    
    elif analyse_type==11:  
         analyse11_test_feature_TSNE_ablation(cfg)
    elif analyse_type==12:
         analyse12_test_data_feature_tsne(cfg)
    elif analyse_type==13:
        analyse13_test_data_feature_mean_std(cfg)
    elif analyse_type==14:
        analyse14_ood_2_tail(cfg)
    elif analyse_type==15:
        analyse15_ood_2_tail_tsne(cfg)
    elif analyse_type==16:
        analyse16_lambda_pap(cfg)
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