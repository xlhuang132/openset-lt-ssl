import numpy as np
def reduce_num_list(num_list,total): 
    per=total/sum(num_list)
    for i in range(len(num_list)):
        num_list[i]=int(num_list[i]*per)
    return num_list

def get_num_per_cls(n_labels_head,num_classes, imb_factor, imb_type ):
        img_max = n_labels_head
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(num_classes):
                num = img_max * (imb_factor**(cls_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * num_classes) 
        return img_num_per_cls 

def train_val_split(labels, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    # 根据不平衡设定来切分数据
    num_classes=cfg.DATASET.NUM_CLASSES
    num_labeled_head=cfg.DATASET.DL.NUM_LABELED_HEAD
    imb_factor_l=1./cfg.DATASET.DL.IMB_FACTOR_L 
    num_unlabeled_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD 
    imb_factor_ul=1./cfg.DATASET.DU.ID.IMB_FACTOR_UL 
    imb_type=cfg.DATASET.IMB_TYPE
    ood_r=cfg.DATASET.DU.OOD.RATIO if cfg.DATASET.DU.OOD.ENABLE else 0
    total_ood_num=0
    # DL 数据
    l_num_per_cls_list=get_num_per_cls(num_labeled_head,num_classes,imb_factor_l,imb_type)
    print('** DL data distribution:{}'.format(l_num_per_cls_list))
    ul_num_per_cls_list=get_num_per_cls(num_unlabeled_head,num_classes,imb_factor_ul,imb_type)
    total_ood_num=int(ood_r*sum(ul_num_per_cls_list))
    if ood_r==0:pass
    else:        
        total_id_num=sum(ul_num_per_cls_list)-total_ood_num
        ul_num_per_cls_list=reduce_num_list(ul_num_per_cls_list,total_id_num)
        
    print('** DU ID data distribution:{}'.format(ul_num_per_cls_list))
    print('** DU OOD data num:{}'.format(total_ood_num))
    if cfg.DATASET.NAME=='cifar10':
        val_num=500
    elif cfg.DATASET.NAME=='cifar100':
        val_num=50
    else:
        raise "The dataset is not valid!"
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        assert l_num_per_cls_list[i]+ul_num_per_cls_list[i]+val_num<=len(idxs)
        train_labeled_idxs.extend(idxs[:l_num_per_cls_list[i]])
        train_unlabeled_idxs.extend(idxs[l_num_per_cls_list[i]:l_num_per_cls_list[i]+ul_num_per_cls_list[i]])
        val_idxs.extend(idxs[-val_num:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs,l_num_per_cls_list,total_ood_num