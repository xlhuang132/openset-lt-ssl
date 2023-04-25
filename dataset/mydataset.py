from PIL import Image
from torch.utils.data import Dataset
from .base import BaseNumpyDataset
import random
import numpy as np
#val，l-train dataset
class MyDataset(BaseNumpyDataset):
    """
    Interface provided for customized data sets
    names_file：a txt file, each line in the form of "image_path label"
    transform: transform pipline for mydatasets
    """
    def __init__(self, names_file, num_classes=10,transform=None): 
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []
        self.num_classes=num_classes
        file = open(self.names_file)
        
        for f in file:
            self.names_list.append(f)
            self.size += 1

        self.num_per_cls_list =  self._load_num_samples_per_class()
        self.total_num=self.size
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].strip().split(' ')[0]
        image = Image.open(image_path)
        if(image.mode == 'L'):
            image = image.convert('RGB')
        label = int(self.names_list[idx].split(' ')[1])

        if self.transform:
            image = self.transform(image)

        return image, label,idx

    # label-to-class quantity
    def _load_num_samples_per_class(self):
        labels = []
        for idx in range(len(self.names_list)):
            label = int(self.names_list[idx].split(' ')[1])
            labels.append(label)
        labels=np.array(labels)
        classes = range(-1,self.num_classes)
        classwise_num_samples = [0]*(len(classes)-1)
        for i in classes:
            if i==-1:
                self.ood_num=len(np.where(labels == i)[0])
                continue
            classwise_num_samples[i] = len(np.where(labels == i)[0])
 
        return np.array(classwise_num_samples)
    
    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i