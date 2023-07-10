import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from PIL import Image


class DUCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    
    def __init__(self, root,train=True,
                 transform=None,target_transform=None, download=False):
        super(DUCIFAR10, self).__init__(root, train=train, transform=transform,target_transform=target_transform, download=download)  

    def __getitem__(self, index):
        """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
class DUCIFAR100(DUCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
