import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_transform(train=True):
    """定义数据预处理：训练集做数据增强，验证集只做标准化"""
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),      
            transforms.ToTensor(),                   
            transforms.Normalize((0.4914, 0.4822, 0.4465),   
                                 (0.2023, 0.1994, 0.2010))   
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    return transform

def get_dataloader(train=True):
    """返回训练集或验证集的 DataLoader"""
    transform = get_transform(train)
    
    dataset = datasets.CIFAR10(root=config.data_root, train=train,
                               download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=train, num_workers=config.num_workers)
    return dataloader