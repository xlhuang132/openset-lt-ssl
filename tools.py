 
import numpy as np
import pickle as p 
from PIL import Image 
import torch 
import argparse
from config.defaults import update_config,_C as cfg
from trainer.build_trainer import build_trainer 
import random
import os 
import torch.backends.cudnn as cudnn   

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        # 以字典的形式取出数据
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        print(Y.shape)
        return X, Y

def save_img():
    imgX, imgY = load_CIFAR_batch("/home/aa/xlhuang/Openset-LT-SSL/data/cifar10/cifar-10-batches-py/data_batch_1")
    for i in range(imgX.shape[0]):
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB",(i0,i1,i2))
        name = "img" + str(i)+".png"
        if not os.path.exists("./data/"+str(imgY[i])):
            os.mkdir("./data/"+str(imgY[i]))
        img.save("./data/"+str(imgY[i])+"/"+name,"png")
    print("save successfully!") 
    
def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/baseline_cifar10.yaml",
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


# seed=random.randint(1,1000)
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
IF=cfg.DATASET.IFS# 10,50,100
ood_r=cfg.DATASET.OODRS # basline不用mixup的话不用考虑r 0.0,0.25, 0.5, 0.75,1.0 randomsampler+classreversedsampler没有用到mixup

for if_ in IF:  # if
    # 同分布
    for r in ood_r:  
        cfg.defrost()
        cfg.DATASET.DL.IMB_FACTOR_L=if_
        cfg.DATASET.DU.ID.IMB_FACTOR_UL=if_
        cfg.SEED=seed
        cfg.DATASET.DU.OOD.RATIO=r
        print("*************{} IF {}  R {} begin *************".format(cfg.DATASET.NAME,if_,r))
        cfg.freeze() 
        trainer=build_trainer(cfg)
        trainer.train()
        print("*************{} IF {}  R {} end *************".format(cfg.DATASET.NAME,if_,r))
        
        
    
   