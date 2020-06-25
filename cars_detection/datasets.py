from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import transforms as T

import os
import numpy as np

import utils
from car_model_dataset_coco import CarModelDatasetCOCO
from car_model_dataset_voc import CarModelDatasetVOC
        
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

def get_transform(augment):
    transforms = []
    transforms.append(T.ToTensor())
    if augment:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Normalize(mean_nums, std_nums))
        
    return T.Compose(transforms)
    
# Set the directory for the data
train_data_dir = 'data/train_modified'
test_data_dir = 'data/test_modified'

train_voc = 'data/train_correct.csv'
test_voc = 'data/test_correct.csv'
class_list = 'data/class_names_id.csv'

num_workers = 32

def get_train_valid_loader_voc(config, augment=True, random_seed=8890, shuffle=True, valid_size=0.1, pin_memory=True):
    print("Initializing datasets and dataloaders VOC for train and validation...")
    
    train_dataset = CarModelDatasetVOC(root=train_data_dir,
                                        annotation=train_voc,
                                        class_list=class_list,
                                        img_size=config['imgsize'],
                                        car_only=config['train_only_car'],
                                        transforms=get_transform(True))
    val_dataset = CarModelDatasetVOC(root=train_data_dir,
                                      annotation=train_voc,
                                      class_list=class_list,
                                      img_size=config['imgsize'],
                                      car_only=config['train_only_car'],
                                      transforms=get_transform(False))
                
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
                
    # Make iterables with the dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['batch_size'], 
                              sampler=train_sampler, 
                              num_workers=num_workers, 
                              collate_fn=utils.collate_fn, 
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config['batch_size'], 
                            sampler=val_sampler, 
                            num_workers=num_workers, 
                            collate_fn=utils.collate_fn, 
                            pin_memory=pin_memory)
                        
    return train_loader, val_loader
    
def get_test_loader_voc(config, pin_memory=True):
    print("Initializing dataset and dataloader VOC for test...")

    test_dataset = CarModelDatasetVOC(root=test_data_dir,
                                    annotation=test_voc,
                                    class_list=class_list,
                                    img_size=config['imgsize'],
                                    car_only=config['train_only_car'],
                                    transforms=get_transform(False))
                
    # Make iterables with the dataloaders
    test_loader = DataLoader(test_dataset, 
                            batch_size=1,
                            num_workers=num_workers,
                            collate_fn=utils.collate_fn,
                            pin_memory=pin_memory)
                        
    return test_loader

def get_train_valid_loader_coco(config, augment=True, random_seed=8890, shuffle=True, valid_size=0.1, pin_memory=True):
    print("Initializing datasets and dataloaders COCO for train and validation...")
    
    train_coco = 'data/train_correct.json'

    train_dataset = CarModelDatasetCOCO(root=train_data_dir,
                                    annotation=train_coco,
                                    img_size=config['imgsize'],
                                    car_only=config['train_only_car'],
                                    transforms=get_transform(True))
    val_dataset = CarModelDatasetCOCO(root=train_data_dir,
                                  annotation=train_coco,
                                  img_size=config['imgsize'],
                                  car_only=config['train_only_car'],
                                  transforms=get_transform(False))
                
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)
                
    # Make iterables with the dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['batch_size'], 
                              sampler=train_sampler, 
                              num_workers=num_workers, 
                              collate_fn=utils.collate_fn, 
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config['batch_size'], 
                            sampler=val_sampler, 
                            num_workers=num_workers, 
                            collate_fn=utils.collate_fn, 
                            pin_memory=pin_memory)
                        
    return train_loader, val_loader
    
def get_test_loader_coco(config, pin_memory=True):
    print("Initializing dataset and dataloader COCO for test...")
    
    test_coco = 'data/test_correct.json'

    test_dataset = CarModelDatasetCOCO(root=test_data_dir,
                                        annotation=test_coco,
                                        img_size=config['imgsize'],
                                        car_only=config['train_only_car'],
                                        transforms=get_transform(False))
                
    # Make iterables with the dataloaders
    test_loader = DataLoader(test_dataset, 
                            batch_size=1,
                            num_workers=num_workers,
                            collate_fn=utils.collate_fn,
                            pin_memory=pin_memory)
                        
    return test_loader