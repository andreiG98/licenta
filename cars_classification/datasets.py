import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10

import numpy as np

# Make transforms and use data loaders
# We'll use these a lot, so make them variables
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

def get_train_valid_loader(config, augment=True, random_seed=8890, shuffle=True, valid_size=0.1, pin_memory=True):
    
    if config['cifar10']:
        print("Initializing datasets and dataloaders CIFAR10 train...")
        
        data_transforms = {}
        data_transforms['val'] = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_nums, std_nums)])
        if augment:
            data_transforms['train'] = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_nums, std_nums)])
        else:
            data_transforms['train'] = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_nums, std_nums)]),
                                    
        # Use the image folder function to create datasets
        chosen_datasets = {phase: CIFAR10(root='./data', 
                                  train=True,
                                  download=True, 
                                  transform=data_transforms[phase])
                            for phase in ['train', 'val']}

    else:
        print("Initializing datasets and dataloaders for train and validation...")
        
        # Set the directory for the data
        data_dir = 'data_modified'
    
        img_size = config['imgsize']
        data_transforms = {}
        data_transforms['val'] = transforms.Compose([
                                    transforms.Resize(size=img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_nums, std_nums)])
        if augment:
            data_transforms['train'] = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize(size=img_size),
                                            transforms.RandomRotation(degrees=15),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_nums, std_nums)])
        else:
            data_transforms['train'] = transforms.Compose([
                                            transforms.Resize(size=img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean_nums, std_nums)]),
                                            
        # Use the image folder function to create datasets
        chosen_datasets = {phase: ImageFolder(os.path.join(data_dir, 'train'), data_transforms[phase])
                                for phase in ['train', 'val']}
                    
    num_train = len(chosen_datasets['train'])
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    sampler = {x: SubsetRandomSampler(idx)
                for x, idx in zip(['train', 'val'], [train_idx, val_idx])}
                
    num_workers = 8
    # Make iterables with the dataloaders
    dataloaders_dict = {x: DataLoader(chosen_datasets[x], 
                                    batch_size=config['batch_size'], 
                                    sampler=sampler[x],
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
                        for x in ['train', 'val']}

    return dataloaders_dict
    
def get_test_loader(config, pin_memory=True):
    # if config['cifar10']:
    #     print("Initializing datasets and dataloaders CIFAR10 test...")
        
    #     data_transforms = transforms.Compose([
    #                             transforms.ToTensor(),
    #                             transforms.Normalize(mean_nums, std_nums)])
        
    #     testset = CIFAR10(root='./data',
    #                         train=False,
    #                         download=True,
    #                         transform=transform)
    #     num_workers = 8
    #     dataloaders = DataLoader(testset, 
    #                             batch_size=config['batch_size'],
    #                             shuffle=False, 
    #                             num_workers=num_workers,
    #                             pin_memory=pin_memory)

    # else:
    print("Initializing dataset and dataloader for test...")
    
    # Set the directory for the data
    data_dir = 'data_modified'

    img_size = config['imgsize']
    data_transforms = transforms.Compose([
                                transforms.Resize(size=img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean_nums, std_nums)])
                                        
    # Use the image folder function to create datasets
    chosen_datasets = ImageFolder(os.path.join(data_dir, 'test'), data_transforms)
                
    num_workers = 8
    # Make iterables with the dataloaders
    dataloaders = DataLoader(chosen_datasets, 
                            batch_size=1, 
                            num_workers=num_workers,
                            pin_memory=pin_memory)

    return dataloaders