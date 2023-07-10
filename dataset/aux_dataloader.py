
from datasets.random_sampler import RandomSampler
from torch.utils.data import DataLoader
import torchvision

from torchvision import transforms

def get_aux_dataloader(cfg,data_root=None,data_type=None,total_samples=None,batch_size=None):
    assert data_root!=None and data_type!=None and total_samples!=None and batch_size!=None
    
    mean = [0.4914, 0.4822, 0.4465] if cfg.DATASET.DATASET.startswith('cifar') else [.5, .5, .5]
    std = [0.2023, 0.1994, 0.2010] if cfg.DATASET.DATASET.startswith('cifar') else [.5, .5, .5]
        
    aux_transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if data_type=="SVHN":
        aux_dataset=torchvision.datasets.SVHN(data_root,split='train',transform=aux_transform,download=True)
    elif data_type=="CIFAR100":
        aux_dataset=torchvision.datasets.CIFAR100(data_root,train=True,transform=aux_transform,download=False)
    elif data_type=="CIFAR10":
        aux_dataset=torchvision.datasets.CIFAR10(data_root,train=True,transform=aux_transform,download=False)
    aux_sampler=RandomSampler(data_source=aux_dataset,total_samples= total_samples)
    aux_loader=DataLoader(
                aux_dataset,
                batch_size=batch_size, 
                pin_memory=False,  
                drop_last=False,
                sampler=aux_sampler
            )
    
    return aux_loader