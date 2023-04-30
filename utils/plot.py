import matplotlib.pyplot as plt
import numpy as np
from utils.color_generator import generate_colors,generate_ood_color
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE  
import torch
import os
import pandas as pd
import copy
import seaborn as sns
from sklearn import manifold
from scipy.interpolate import interp1d
import mpl_toolkits.axisartist as axisartist
import csv
import matplotlib.font_manager as fm
import matplotlib
 
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
times_path='/home/aa/xlhuang/Openset-LT-SSL/utils/fonts/times.ttf'
helvetica_path='/home/aa/xlhuang/Openset-LT-SSL/utils/fonts/Helvetica.ttf'
font_h = fm.FontProperties(fname=helvetica_path)
font_t = fm.FontProperties(fname=times_path) 
# plt.rcParams['text.usetex'] = True

def plot_pd_heatmap(results,save_path=''):
    plt.figure(figsize=(11, 9),dpi=300)
    sns.heatmap(data=results)
    if save_path!='': 
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_pd_heatmaps(results,r=1,c=1,save_path='',title='',subtitles=[]):
    assert len(results)==r*c
    fig =plt.figure(figsize=(2+c*4,r*4),dpi=300) 
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
    for i in range(r):
        for j in range(c):
            plt.subplot(r,c,i*r+j+1)
            # if i*r+j+1==len(results):
            #     sns.heatmap(data=results[i*r+j])
            # else:
            
            h=sns.heatmap(data=results[i*r+j])
            # h=sns.heatmap(data=results[i*r+j],cbar=False)
            # h3 =plt.contourf(results[i*r+j],cmap = plt.cm.coolwarm,norm = norm)
            # h3 =plt.contourf(results[i*r+j],cmap = plt.cm.coolwarm)
            plt.title(subtitles[i*r+j])
    plt.suptitle(title)
    fig.subplots_adjust(wspace =0.05)
    # cb = plt.colorbar(h.collections[0]) #显示colorbar
    # cb.ax.tick_params()  # 设置colorbar刻度字体大小。
    # #colorbar 左 下 宽 高 
    # l = 0.92
    # b = 0.12
    # w = 0.015
    # h = 1 - 2*b 

    # #对应 l,b,w,h；设置colorbar位置；
    # rect = [l,b,w,h] 
    # cbar_ax = fig.add_axes(rect) 
    # cb = plt.colorbar(h3, cax=cbar_ax)

    # #设置colorbar标签字体等
    # cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
    # font = {'family' : 'serif',
    # #       'color'  : 'darkred',
    #     'color'  : 'black',
    #     'weight' : 'normal',
    #     'size'   : 16,
    #     } 
    if save_path!='': 
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_pr(precision,recall,save_path=''):
    assert len(precision)==len(recall) and len(precision)>0
    fontsize=16
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision, markersize=3)
    # plt.xticks([])
    # plt.yticks([])
    if save_path!='': 
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_ood_detection_over_epoch(results=[],save_path=''):
    assert results!=[]
    x=[i+1 for i in range(len(results[0]))]
    plt.figure(figsize=(8,8)) 
    titles=['ID Pre','ID Rec','OOD Pre','OOD Rec']    
    for i,result in enumerate(results):
        plt.plot(x, result, linewidth=3, markersize=8)  # 绘制折线图，添加数据点，设置点的大小
    plt.grid()
    plt.legend(titles,fontsize=10)      
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_group_acc_over_epoch(group_acc,legend=['Many','Medium','Few'],title="Group Acc",step=1,save_path='',warmup_epoch=0):
    assert len(group_acc)>0 and len(group_acc[0])==len(legend)
    group_acc=np.array(group_acc)
    many=group_acc[:,0]
    medium=group_acc[:,1]
    few=group_acc[:,2]
    x=[i*step for i in range(1,len(group_acc)+1)]
 
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.plot(x, many, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, medium, marker='o', markersize=3)
    plt.plot(x, few, marker='o', markersize=3) 
    # if warmup_epoch!=0:
    #     plt.axvline(2, color='r')
    plt.legend(legend)
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_loss_over_epoch(losses,title,save_path=''):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    x=[i for i in range(1,len(losses)+1)]
    plt.plot(x, losses, markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_acc_over_epoch(acc,title="Average Accuracy",save_path=''):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    x=[i for i in range(1,len(acc)+1)]
    plt.plot(x, acc, markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    if save_path!='':
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

  
def plot_ood_dectector_FPR95(results,xlegend=[],save_path="",xlabel="thresh",ylabel='FPR95'):
    """
        inputs:
            三维：[IF,thresh,TPF/FPR]
        第三维用不同形状的线代替 
    """
    
    return 


def plot_ood_dectector_TPR(results,line_names,x=[],title='',save_path="",xlabel="thresh",ylabel='TPR'):
    """
        inputs:
            三维：[IF,thresh,TPF/FPR]
        第三维用不同形状的线代替 
    """
    assert len(results)==len(name)
    plt.title('{}'.format(title))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x and x[0] is not str:
        plt.xticks(x)
    markers=['o','x','*','^','+']
    linestyles=[':','--','-',"''",]
    for i,item in enumerate(results):  # if 
        for j,it in enumerate(item):
            plt.plot(x, it, marker=markers[i],linestyle=':', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.legend(line_names)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_feat_tsne(feat,y,title,num_classes,save_path=""):
    # 生成颜色
    fontsize=20
    colors = generate_colors(num_classes) 
    c_p = []
    if torch.is_tensor(y): yy = y.numpy()
    else: yy = y
    if -1 in yy:
        colors[-1]=generate_ood_color()
    for i in yy: 
        c_p.append(colors[i])
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(12,12)) 
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10, c=c_p,alpha=0.5)
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    # plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
    plt.close()
    return  


def plot_dl_du_all_feat_tsne(feat,y,title,num_classes,save_path="",dl_len=None,du_len=None):
    # 生成颜色
    fontsize=20
    if dl_len==None and du_len==None:
        dl_len=feat.size(0)
    colors = generate_colors(num_classes) 
    c_p = []
    if torch.is_tensor(y): yy = y.numpy()
    else: yy = y
    if -1 in yy:
        colors.append(generate_ood_color())
    for i in yy: 
        c_p.append(colors[i])
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(12,12)) 
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=10, c=c_p[:],alpha=0.5)
    # plt.scatter(X_tsne[:dl_len, 0], X_tsne[:dl_len, 1],s=10, c=c_p[:dl_len],alpha=0.5,marker='o')
    # plt.scatter(X_tsne[dl_len:, 0], X_tsne[dl_len:, 1],s=10, c=c_p[dl_len:],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    
    # dl
    plt.figure(figsize=(12,12)) 
    plt.scatter(X_tsne[:dl_len, 0], X_tsne[:dl_len, 1],s=10, c=c_p[:dl_len],alpha=0.5,marker='o') 
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_dl_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    
    # du
    plt.figure(figsize=(12,12))  
    plt.scatter(X_tsne[dl_len:dl_len+du_len, 0], X_tsne[dl_len:dl_len+du_len, 1],s=10, c=c_p[dl_len:dl_len+du_len],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_du_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
     # du
    plt.figure(figsize=(12,12))  
    plt.scatter(X_tsne[dl_len+du_len:, 0], X_tsne[dl_len+du_len:, 1],s=10, c=c_p[dl_len+du_len:],alpha=0.5,marker='*')
    
    labels=[mpatches.Patch(color=colors[i],label=i)for i in range(len(colors))]
    plt.legend(handles=labels,loc='upper right',fontsize=fontsize) 
    plt.title(title,fontsize=fontsize)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="":
        plt.savefig(os.path.join(save_path,"allfeat_test_tsne_best.jpg"), dpi=300, bbox_inches='tight')  
    plt.close()
    return  


def plot_ablation_feat_tsne(gts,preds,feats,num_classes=0,title='Test data feature',alg='',
                subtitles=[],save_path=''):
    assert subtitles!=[] and len(subtitles)==4
    
    fontsize=24
    
    colors = generate_colors(num_classes,return_deep_group=True) 
    colors_r,colors_w=colors[:num_classes],colors[num_classes:] 
    n=len(gts)//len(subtitles) 
    fig = plt.figure(figsize=(10, 10))
    
    # font_zh.set_size(fontsize) 
    for i in range(len(subtitles)): 
        # gt=gts[i]
        # pred=preds[i]
        # feat=feats[i]
        # c_item=[]
        # if torch.is_tensor(gt): gt = gt.numpy() 
        # X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
        
        ax = fig.add_subplot(2,2,i+1)  
        logname=os.path.join(save_path,'{}.csv'.format(subtitles[i]))
        df=pd.read_csv(logname,sep=' ',delimiter=',',names=['x', 'y', 'gt','pred','c'])
        df=np.array(df)
        for j in range(1,len(df)): 
            x,y,g,p,c=df[j] 
            if g==p:
                plt.scatter(x, y,s=10, c=c,marker='o') 
            else:            
                plt.scatter(x,y,s=20, c=c,marker='+')#,alpha=0.5 
        # with open(logname, 'w') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',') 
        #     for j in range(len(gt)):
        #         if gt[j]==pred[j]:
        #             cp=colors_r[gt[j]]
        #         else:
        #             cp=colors_w[gt[j]]
        #         logwriter.writerow([X_tsne[j, 0], X_tsne[j, 1],gt[j],pred[j].item(),cp])
        # for j in range(len(X_tsne)):
        #     if gt[j]==pred[j]:
        #         plt.scatter(X_tsne[j][0], X_tsne[j][ 1],s=10, c=c_p[j],marker='o',alpha=0.5) 
        #     else:            
        #         plt.scatter(X_tsne[j][ 0], X_tsne[j][ 1],s=10, c=c_p[j],marker='+',alpha=0.5) 
             
        
        # ax.scatter(X_tsne[:,0], X_tsne[:,1],s=10, c=c_item,alpha=0.5) 
        plt.xticks([])
        plt.yticks([])
        
        # ax.set_title(subtitles[i], y=-0.18, fontsize=fontsize) 
        font_t.set_size(fontsize+4)         
        ax.set_title(subtitles[i], y=-0.18, fontproperties=font_t) 
        # elif i==1:
        #     ax = fig.add_subplot(222)
        #     labels=[mpatches.Patch(color=colors_r[i],label="Correct-{}".format(i+1))for i in range(len(colors_r))]+\
        #     [mpatches.Patch(color=colors_w[i],label="Wrong-{}".format(i+1))for i in range(len(colors_w))]
            
        #     font_zh.set_size(fontsize-4)  
        #     # plt.legend(handles=labels,loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0., prop=font_zh)  ##设置ax4中legend的位置，将其放在图外
    
    plt.subplots_adjust(wspace =0.05,)#调整子图间距
            
    
    # plt.show()
    if save_path!="":
        filename="test_data_ablation_feat_tsne.jpg" 
        plt.savefig(os.path.join(save_path,filename), dpi=300, bbox_inches='tight')  
        print('save path:{}'.format(os.path.join(save_path,filename)))
    plt.close()
    
    return
 

def plot_problem_feat_tsne(gts,preds,feats,num_classes=0,titles='Test data feature',alg='', 
                if_legend=False,filename='',save_path=''):
    fontsize=22   
    # font_zh.set_size(fontsize) 
    colors = generate_colors(num_classes,return_deep_group=True) 
    colors_r,colors_w=colors[:num_classes],colors[num_classes:]
    # plt.figure(figsize=(8,8)) 
    plt.figure(figsize=(12,18)) 
    for i in range(len(gts)):
        gt=gts[i]
        pred=preds[i]
        feat=feats[i]
        title=titles[i]
        plt.subplot(3,2,i+1)
        
        c_p = [] 
        # gt=[]
        # pred=[]
        # X_tsne=[]
        
        if torch.is_tensor(gt): gt = gt.numpy()
        else: gt = gt 
        X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
        # 保存 X_tsne[j, 0] [j,1] gt[j] pred [j] 
        # logname=os.path.join(save_path[:save_path.rfind('/')+1]+'{}.csv'.format(title))
        # with open(logname, 'r') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #    # logwriter.writerow(['x', 'y', 'gt','pred']) 
        #     for j in range(len(gt)):
        #         logwriter.writerow([X_tsne[j, 0], X_tsne[j, 1],gt[j],pred[j].item()])
        # csv_reader = csv.reader(open(logname))
        # df=pd.read_csv(logname,sep=' ',delimiter=',',names=['x', 'y', 'gt','pred'])
        # df=np.array(df)
        # for j in range(1,len(df)): 
        #     x,y,g,p=df[j]
        #     x,y,g,p=float(x),float(y),int(g),int(p)   
        #     if g==p:
        #         plt.scatter(x, y,s=5, c=colors_r[g],marker='o')#,alpha=0.5 
        #     else:            
        #         plt.scatter(x,y,s=15, c=colors_w[g],marker='+')#,alpha=0.5 
        for j in range(len(X_tsne)):
            if gt[j]==pred[j]:
                plt.scatter(X_tsne[j][0], X_tsne[j][ 1],s=15, c=colors_r[gt[j]],marker='o',alpha=0.5) 
            else:            
                plt.scatter(X_tsne[j][0], X_tsne[j][ 1],s=20, c=colors_w[gt[j]],marker='+',alpha=0.5) 
                
        # font_zh.set_size(fontsize)    
        plt.xticks([])
        plt.yticks([])
        font_t.set_size(fontsize) 
        plt.title(title, fontproperties=font_t,y=-0.1) #
        # plt.title(title, fontsize=fontsize,y=-0.1) #
        
    plt.subplots_adjust(wspace =0.03,hspace=0.12)#调整子图间距
    if save_path!="": 
        plt.savefig( save_path , dpi=300, bbox_inches='tight')  
    plt.close()
    
    return

def plot_tail_feat_tsne(gt,pred,feat,num_classes=0,title='',alg='',
                filename='',save_path=''):
    fontsize=16
    colors = generate_colors(num_classes,return_deep_group=True) 
    colors_r,colors_w=colors[:num_classes],colors[num_classes:]
    c_p = []
    if torch.is_tensor(gt): gt = gt.numpy()
    else: gt = gt
    
    tails=[7,8,9]
    old_gt=gt
    ii=0
    for i in range(len(old_gt)): 
        if old_gt[i] in tails:
            gt[ii]=old_gt[i]
            pred[ii]=pred[i]
            feat[ii]=feat[i]
            ii+=1        
    gt=gt[:ii]
    pred=pred[:ii]
    feat=feat[:ii]
    for i in range(len(gt)): 
        c_p.append(colors_r[gt[i]]) if gt[i]==pred[i] else c_p.append(colors_w[gt[i]]) 
    X_tsne = TSNE(n_components=2,random_state=33,early_exaggeration=30).fit_transform(feat.numpy()) 
    # mix
    plt.figure(figsize=(8,8)) 
    for i in range(len(X_tsne)): 
        if gt[i]==pred[i]:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],s=10, c=c_p[i],marker='o',alpha=0.5) 
        else:            
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1],s=10, c=c_p[i],marker='x',alpha=0.5) 
    labels=[mpatches.Patch(color=colors_r[i],label="Correct-{}".format(i+1))for i in tails]+\
    [mpatches.Patch(color=colors_w[i],label="Wrong-{}".format(i+1))for i in tails]
    plt.legend(handles=labels,loc='upper left',fontsize=10)  
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # plt.show()
    if save_path!="": 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
    plt.close()
    
    return


def plot_accs_zhexian(group_accs,branches_name,title,x_legend,save_path="",xlabel="",ylabel='Accuracy (%)',color=[]):
    assert len(branches_name)==len(group_accs)
    assert (len(branches_name)==len(color))if color!=[] else True
    plt.title('{}'.format(title))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if x_legend.any() is not str:
        plt.xticks(x_legend)
    # if color==[]:
    #     color=generate_colors(len(branches_name))
    # for i,group_acc in enumerate(group_accs):
    #     plt.plot(x_legend, group_acc, marker='o', markersize=3,c=color[i])  # 绘制折线图，添加数据点，设置点的大小
     
    markers=['o','X','*','^','D','s']
    linestyles=['-','dotted','--',':','-.','dashdot']
    for i,group_acc in enumerate(group_accs):
        plt.plot(x_legend, group_acc, marker=markers[i], linestyle=linestyles[i],linewidth=3, markersize=8)  # 绘制折线图，添加数据点，设置点的大小
    plt.grid()
    plt.legend(branches_name,fontsize=10) 
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    
def plot_feat_cha_zhexian(gt_mean_cha,gt_var_cha,pred_mean_cha,pred_var_cha,num_classes=10,alg='',
                    filename='',save_path=''):
    fontsize=12
    fig = plt.figure(figsize=(6, 4))  
    x=range(num_classes)
    plt.xticks(x)
    plt.plot(x, gt_mean_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(x, gt_var_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(x, pred_mean_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(x, pred_var_cha, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.legend(['gt_mean_diff','gt_var_diff','pred_mean_diff','pred_var_diff'],fontsize=fontsize)
    plt.legend([r"$\Vert {\mu}^{train}_c-\mu^{test}_c \Vert$",r"$\Vert \sigma^{train}_c-\sigma^{test}_c \Vert$"],fontsize=fontsize)
    
    if save_path!="":
        filename="test_data_diff_{}.jpg".format(alg) if filename=='' else filename
        plt.savefig(os.path.join(save_path,filename), dpi=300, bbox_inches='tight')
    plt.close() 
    
    return  
def plot_zhexian(x,y,xlabel='',ylabel='',save_path=''):
    fig= plt.figure(figsize=(6,2))  
    ax=plt.subplot(1,1,1)
    plt.plot(x, y,marker='o')
    fontsize=12
    
    # plt.xticks(x,fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    font_h.set_size(fontsize)
    plt.xticks(x,fontproperties=font_h)
    plt.yticks(fontproperties=font_h)
    font_h.set_size(fontsize)
    ff= {  # 用 dict 单独指定 title 样式
    'family': 'Helvetica',
    'weight': 'normal',
    'size': fontsize+2,
    'usetex' : True,
    }  
    plt.xlabel(xlabel,ff)
    # plt.ylabel(ylabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontproperties=font_h)
    # ax.set_title(' ',fontsize=fontsize+2, y=-0.35,fontproperties=font_zh )  
         
    plt.grid()
    if save_path!="":
        plt.savefig(os.path.join(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    return
def plot_feat_diff(means,algs,num_classes=10,save_path=''):
    fontsize=12
    fig = plt.figure(figsize=(6, 4))  
    x=range(num_classes)
    plt.xticks(x,fontsize=fontsize)
    for i in range(len(means)):
        plt.plot(x, means[i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.legend(algs,fontsize=fontsize)
    plt.fill_between([0, 2.5],0,0.4, facecolor='#ff7f0e', alpha=0.3) 
    plt.fill_between([2.5, 5.5],0,0.4, facecolor='#1f77b4', alpha=0.3) 
    plt.fill_between([5.5, 9],0,0.4, facecolor='#ee82ee', alpha=0.3) 
    
    if save_path!="":
        plt.savefig(os.path.join(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    return
   
def plot_task_bar(labeled_dist,unlabeled_dist,save_path):
    x_legend_=[i for i in range(len(labeled_dist))]
    cubic_interploation_model=interp1d(x_legend_,labeled_dist,kind="cubic")
    x_legend=np.linspace(x_legend_[0],x_legend_[-1],500)
    labeled_y=cubic_interploation_model(x_legend)
    
    
    cubic_interploation_model=interp1d(x_legend_,unlabeled_dist,kind="cubic")
    unlabeled_y=cubic_interploation_model(x_legend)
    
    fig = plt.figure(figsize=(6, 4))
    # myfonts = "Times New Roman"    
    # plt.rcParams['font.family'] = "sans-serif"
    # plt.rcParams['font.sans-serif'] = myfonts
    # plt.rc('font',family='Times New Roman')
    #使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)  
    #将绘图区对象添加到画布中
    fig.add_axes(ax)
    
    ax.axis[:].set_visible(False)#通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis["x"] = ax.new_floating_axis(0,0)#ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"].set_axisline_style("->", size = 1.0)#给x坐标轴加上箭头
    #添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1,0)
    ax.axis["y"].set_axisline_style("-|>", size = 1.0)
    #设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right") 
    #给x坐标轴加上箭头
    ax.axis["x"].set_axisline_style("->", size = 1.0)
    ax.axis["y"].set_axisline_style("->", size = 1.0)
    ax.axis["x"].label.set_visible(True)
    ax.axis["x"].label.set_text("Class ID")
    ax.axis["x"].label.set_pad(-20)
    ax.axis["y"].label.set_visible(True)
    ax.axis["y"].label.set_text("Number")
    ax.axis["y"].label.set_pad(-20)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    plt.xticks([]) 
    plt.yticks([]) 
    # plt.xlabel("Class ID")
    # plt.ylabel("Number")
    y0=np.zeros_like(labeled_y)     
    plt.fill_between(x_legend, unlabeled_y, y0, where=(unlabeled_y > y0),facecolor='#1f77b4',alpha=0.3,label='unlabeled ID data') 
    plt.fill_between(x_legend, labeled_y, y0, where=(labeled_y > y0),facecolor='#1f77b4',label='labeled ID data')
    x_ood=np.array([i+len(unlabeled_dist)-1 for i in range(5)])
    y_ood=np.array([max(unlabeled_y)]*5)
    y2=np.zeros_like(x_ood) 
    plt.fill_between(x_ood, y_ood, y2, where=(y_ood > y2), facecolor='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    plt.legend()
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_task_together(labeled_dist,unlabeled_dist,group_accs,branches_name,title,x_legend,save_path="",xlabel="",ylabel='Accuracy (%)',color=[]):
    tick_fontsize=14
    label_fontsize=16
    legend_fontsize=12
    title_fontsize=18
    dy=-1.07
    labels=[]
    lines=[]
    fig = plt.figure(figsize=(10, 4)) 
    grid=plt.GridSpec(3,2,wspace=0.2,hspace=0.6)
    ax=plt.subplot(grid[0:2,0])   
    
    # 子图1
    
    # ax = fig.add_subplot(2,2,1)
    # rect1 = [0.14, 0.35, 0.77, 0.6] # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    # rect2 = [0.14, 0.05, 0.77, 0.2]
    # plt.axes(rect1) 
    x_legend_=[i+1 for i in range(len(labeled_dist))] 
    x_ood=np.array([i+1+len(unlabeled_dist) for i in range(10)])
    y_ood=np.array([420, 390, 372, 389, 366, 400, 327, 381, 394, 396])
    # 设置ticks字体大小
    font_h.set_size(tick_fontsize)
    plt.xticks(np.array([i for i in range(1,21)]),fontproperties=font_h) 
    plt.yticks([],fontproperties=font_h)   
    
    # plt.xticks(np.array([i for i in range(1,21)]),fontsize=tick_fontsize) 
    # plt.yticks([])   
    # 设置ticks字体大小 
    # font_zh.set_size(label_fontsize)
    # plt.xlabel(r"Class ID",fontproperties=font_zh)
    # plt.ylabel(r"Number",fontproperties=font_zh)    
    # axline,axlabel=ax.get_legend_handle_labels()
    # lines.extend(axline)
    # labels.extend(axlabel)
    font_t.set_size(title_fontsize)
    ax.set_title(r'(a)', y=dy,fontproperties=font_t )
    y0=np.zeros_like(labeled_dist)     
    plt.bar(x_legend_, labeled_dist, color='#1f77b4',label='labeled ID data')
    plt.bar(x_legend_, unlabeled_dist,  bottom=labeled_dist,color='#1f77b4',alpha=0.3,label='unlabeled ID data' ) 
    plt.bar(1,0 , color='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    font_h.set_size(legend_fontsize)
    plt.legend(prop=font_h) 
    
    # plt.legend(fontsize=legend_fontsize) 
    # 设置ticks字体大小
    font_h.set_size(tick_fontsize)
    plt.xticks(np.array([i for i in range(1,11)]),fontproperties=font_h) 
    
    # plt.xticks(np.array([i for i in range(1,11)]),fontsize=tick_fontsize) 
    plt.yticks([])  
    # 子图2 
    # ax = fig.add_subplot(2,2,3)
    ax=plt.subplot(grid[2,0])
    # plt.axes(rect2) 
    y2=np.zeros_like(x_ood)  
    plt.bar(x_ood, y_ood, color='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    # font_zh.set_size(title_fontsize)
    # ax.set_title(r'(a)', y=dy,fontproperties=font_zh) 
    
    # ax.set_title(r'(a)', y=dy,fontsize=title_fontsize) 
    # axline,axlabel=ax.get_legend_handle_labels()
    # lines.extend(axline)
    # labels.extend(axlabel)
    font_h.set_size(label_fontsize)
    plt.xlabel(r"Class ID",fontproperties=font_h)
    # plt.xlabel(r"Class ID",fontsize=label_fontsize)
    
    # fig.text(0.085, 0.5, 'Number',fontsize=label_fontsize, va='center', rotation='vertical')
    
    fig.text(0.085, 0.5, 'Number of Samples',fontproperties=font_h, va='center', rotation='vertical')
    
    # plt.ylabel(r"Number",fontproperties=font_zh) 
    font_h.set_size(tick_fontsize)
    plt.xticks(np.array([i for i in range(11,21)]),fontproperties=font_h) 
    # plt.xticks(np.array([i for i in range(11,21)]),fontsize=tick_fontsize) 
    plt.yticks([])       
    # font_zh.set_size(legend_fontsize)
    # plt.legend(prop=font_zh) 
    # fig.legend(lines,labels,loc='upper right')
    # ax=plt.subplot(1, 2, 2)    
    ax=plt.subplot(grid[0:3,1])
    font_t.set_size(title_fontsize)
    ax.set_title(r'(b)', y=-0.28,fontproperties=font_t) 
    # ax.set_title(r'(b)', y=-0.28,fontsize=title_fontsize) 
    
    # ff= {  # 用 dict 单独指定 title 样式
    # 'family': 'Times New Roman',
    # 'weight': 'normal',
    # 'size': label_fontsize,
    # 'usetex' : True,
    # }  
    ff= {  # 用 dict 单独指定 title 样式
    'family': 'Helvetica',
    'weight': 'normal',
    'size': label_fontsize,
    'usetex' : True,
    }  
    # plt.rcParams['text.usetex'] = True
    plt.xlabel(xlabel,ff) 
    
    font_h.set_size(label_fontsize)
    # plt.xlabel(xlabel,fontsize=label_fontsize) 
    # plt.xlabel(xlabel,fontproperties=font_zh) 
    plt.ylabel(ylabel,fontproperties=font_h)
    # plt.ylabel(ylabel,fontsize=label_fontsize)
    
    font_h.set_size(tick_fontsize)
    if x_legend.any() is not str:
        plt.xticks(x_legend,fontproperties=font_h) 
    # plt.xticks(x_legend,fontsize=tick_fontsize) 
    # plt.yticks(fontsize=tick_fontsize) 
    plt.yticks(fontproperties=font_h)  
    markers=['o','X','*','^','D','s']
    linestyles=['-','dotted','--',':','-.','dashdot']
    for i,group_acc in enumerate(group_accs):
        plt.plot(x_legend, group_acc, marker=markers[i], linestyle=linestyles[i],linewidth=3, markersize=8)  # 绘制折线图，添加数据点，设置点的大小
    plt.grid()
    
    font_h.set_size(legend_fontsize)
    plt.legend(branches_name,loc='lower left',prop=font_h)
    # plt.legend(branches_name,loc='lower left',fontsize=legend_fontsize)
    
    # plt.subplots_adjust(hspace =0.8,)#调整子图间距
    # plt.legend(branches_name,loc=2, fontsize=fontsize-2,bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_r_group_together(group_accs,branches_name,title,x_legend,save_path="",xlabel="",ylabel='Accuracy (%)',color=[]):
    tick_fontsize=12
    label_fontsize=16
    legend_fontsize=12
    title_fontsize=18
    yy=-0.54
    fig = plt.figure(figsize=(8,5)) 
    ff= {  # 用 dict 单独指定 title 样式
    'family': 'Helvetica',
    'weight': 'normal', 
    'size': label_fontsize,
    'usetex' : True,
    }  
    
    # plt.rcParams['font.family'] = 'DeJavu Serif'
    # plt.rcParams['font.sans-serif'] = ['Helvetica']
    # plt.rcParams['text.usetex'] = True
    markers=['o','X','*','^','D','s']
    linestyles=['-','dotted','--',':','-.','dashdot']
    subtitles=['(a) Head','(b) Medium', '(c) Tail']
    for i in range(1,len(group_accs)+1):
        ax=plt.subplot(2,2,i)
        font_t.set_size(title_fontsize)
        ax.set_title(subtitles[i-1], y=yy,fontproperties=font_t )  
        # ax.set_title(subtitles[i-1], y=yy,fontsize=title_fontsize )  
        font_h.set_size(label_fontsize)
        # plt.xlabel(r"${R}$",ff)  
        plt.xlabel(r"${R}$",ff)  
        plt.ylabel("Accuracy (%)",fontproperties=font_h)
        # plt.ylabel("Accuracy",fontsize=label_fontsize)
        font_h.set_size(tick_fontsize)
        if x_legend.any() is not str:
            # plt.xticks(x_legend,fontsize=tick_fontsize)
            plt.xticks(x_legend,fontproperties=font_h)
        plt.yticks(fontproperties=font_h) 
        # plt.yticks(fontsize=tick_fontsize) 
        markers=['o','X','*','^','D','s']
        linestyles=['-','dotted','--',':','-.','dashdot']
        for j,group_acc in enumerate(group_accs[i-1]):
            plt.plot(x_legend, group_acc, marker=markers[j], linestyle=linestyles[j],linewidth=3, markersize=8)  # 绘制折线图，添加数据点，设置点的大小
        plt.grid() 
        font_h.set_size(legend_fontsize)
        if i==3:
            # plt.legend(branches_name,loc='lower left',prop=font_zh)  ##设置ax4中legend的位置，将其放在图外
            plt.legend(branches_name,loc=2, bbox_to_anchor=(1.47,1.01),borderaxespad = 0., prop=font_h)  ##设置ax4中legend的位置，将其放在图外
    
            # plt.legend(branches_name,loc=2, bbox_to_anchor=(1.43,1.01),borderaxespad = 0., fontsize=legend_fontsize) 
    plt.subplots_adjust(hspace=0.6,wspace=0.25)#,wspace=0.3 
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_task_stack_bar(labeled_dist,unlabeled_dist,save_path,title=""):
    x_legend_=[i for i in range(len(labeled_dist))] 
    
    fig = plt.figure(figsize=(6, 4))  
    
    plt.xticks([]) 
    plt.yticks([]) 
    plt.xlabel("Class ID")
    plt.ylabel("Number")
    plt.title(title)
    y0=np.zeros_like(labeled_dist)     
    plt.bar(x_legend_, labeled_dist, color='#1f77b4',label='labeled ID data')
    plt.bar(x_legend_, unlabeled_dist,  bottom=labeled_dist,color='#1f77b4',alpha=0.3,label='unlabeled ID data') 
    x_ood=np.array([i+len(unlabeled_dist) for i in range(5)])
    y_ood=np.array([max(unlabeled_dist)]*5)
    y2=np.zeros_like(x_ood)  
    # plt.fill_between(x_ood, y_ood, y2, where=(y_ood > y2), facecolor='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    plt.bar(x_ood, y_ood, color='#ff7f0e', alpha=0.3,label='unlabeled OOD data') 
    plt.legend()
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 

def plot_ood_detect_feat(unlabeled_feat,unlabeled_y,pred_id,pred_ood):
    # 分对的ID用绿色三角形表示，分对的OOD用绿色圆形表示
    # 分错的ID用不同颜色的×形表示，分错的OOD用红色的圆形表示
    # 都为0的用灰色表示，表示没用到这部分数据
    no_detect=(1-pred_id)*(1-pred_ood)
    no_detect_index=torch.nonzero(no_detect,as_tuple=False).squeeze(1)
    ones=torch.ones_like(unlabeled_y)
    zeros=torch.zeros_like(unlabeled_y)
    gt_id =torch.where(unlabeled_y >= 0,ones,zeros)
    id_correct_index= torch.nonzero(gt_id==pred_id,as_tuple=False).squeeze(1)
    ood_correct_index= torch.nonzero((1-gt_id)==pred_ood,as_tuple=False).squeeze(1)
    id_wrong_index= torch.nonzero(gt_id==pred_ood,as_tuple=False).squeeze(1)
    ood_wrong_index= torch.nonzero((1-gt_id)==pred_id,as_tuple=False).squeeze(1)
    assert no_detect.sum()+id_correct_index.sum+ood_correct_index.sum()+id_wrong_index.sum()+ood_wrong_index.sum()==unlabeled_y.size(0)
    
    return 
def plot_bar(y,save_path=None):
    x=[i for i in range(len(y))]
    plt.bar(x, y)
    plt.xticks([]) 
    plt.yticks([]) 
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 


def plot_multi_bars(datas,labels, legend=None,xlabel="",ylabel="",title="",save_path="", tick_step=1, group_gap=0.2, bar_gap=0):
    '''
    legend : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    ''' 
    # x为每组柱子x轴的基准位置
    x = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # 绘制柱子
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width) 
    plt.title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels) 
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    if save_path!='': 
        plt.savefig(save_path,dpi=300, bbox_inches='tight')
    plt.close() 
    return 