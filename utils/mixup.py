import numpy as np
import torch
import math
from models.ood_detector import perturb_input 
from torchvision import transforms
def gasuss_noise(img):
    ''' 
        添加高斯噪声
        image:原始图像
        mean : 均值 
        var : 方差,越大，噪声越大
    '''
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(2.0, 2.0))
    img_blur = blur(img)
    return img_blur



def mix_up(l_images,aux_images,model,alpha=0.5,reversed=False,cfg=None):
    len_aux=aux_images.size(0)
    len_l=l_images.size(0)
    if len_aux==0:
        # 添加扰动        
        # eps=cfg.ALGORITHM.OOD_DETECTOR.MAGNITUDE
        # temperature=cfg.ALGORITHM.OOD_DETECTOR.TEMPERATURE
        # with torch.enable_grad():
        #     x_perturbs = perturb_input(
        #             model, l_images, eps,temperature
        #         )
        # return x_perturbs
        # return gasuss_noise(l_images)
        return l_images
    elif len_aux>len_l:
        aux_images=aux_images[:len_l]
    elif len_aux<len_l:
        extend_num= math.ceil(len_l/len_aux)
        tmp=[aux_images]*extend_num
        aux_images=torch.cat(tmp,dim=0)[:len_l]
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    rand_idx = torch.randperm(l_images.size(0)) 
    mixed_images = lam * l_images + (1.0 - lam) * aux_images[rand_idx]
    return mixed_images
