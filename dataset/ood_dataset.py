
from torch.utils.data import Dataset
import os

import torchvision 
 

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
def pad(x, border=4):
    return np.pad(x, [(0,0), (border, border), (border, border), (0, 0)])

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

class OOD_Dataset(Dataset):
    def __init__(self, OOD_path,transform=None, target_transform=None,ood_num=None):
        super(OOD_Dataset, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 )
        assert os.path.exists(OOD_path) 
        self.data = np.load(OOD_path+'.npy')
        if ood_num:
            self.ood_num=min(ood_num,len(self.data))
        else:self.ood_num=len(self.data)
         
        self.data=self.data[:self.ood_num]
        self.data = transpose(normalize(pad(self.data)))
        self.targets = np.array( [-1]*len(self.data))
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target