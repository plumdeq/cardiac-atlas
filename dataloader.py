# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Data loader for emeramyloid project

"""
# Third-party imports 
import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

# Cross-library imports
import config


conf = config.DevConf()


class AmyloidDataloader(object):
    """
    Prepares train and test datalaoders

    Args:
        batch_size (int, default=4): data loader batch size
        shuffle (bool, default=True): shuffle data samples
        num_workers (int, default=4): parallel loaders

    """
    def __init__(self, batch_size=4, shuffle=True, num_workers=4):
        # Default parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # Transforms and dataloaders
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])
                    ]),
            'test': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                    ]),
            'val': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
                    ]),
            }

        self.image_datasets = {
            x: datasets.ImageFolder(conf.data_paths[x], self.data_transforms[x])
            for x in ['train', 'test', 'val']
            }

        self.dataloders = {
                x: DataLoader(self.image_datasets[x], batch_size=self.batch_size, 
                          shuffle=self.shuffle, num_workers=self.num_workers) 
            for x in ['train', 'test', 'val']
            } 

        self.dataset_sizes = {
            x: len(self.image_datasets[x]) 
            for x in ['train', 'test', 'val'] 
            } 
        
        self.class_names = self.image_datasets['train'].classes
