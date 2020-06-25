from datasets import get_train_valid_loader_voc, get_train_valid_loader_coco, get_transform
from model import construct_model_detection
from engine import train_one_epoch, evaluate

import utils

import torch
import torchvision
import argparse
import json
import os
import time
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import transforms as T 

def get_exp_dir(config):
    if config['train_only_car']:
        exp_dir = f'logs/train_only_car/{config["arch"]}_{config["imgsize"][0]}_{config["epochs"]}'
    else:
        exp_dir = f'logs/{config["arch"]}_{config["imgsize"][0]}_{config["epochs"]}'

    if config['finetune']:
        exp_dir += '_finetune'
    
    if config['feature_extract']:
        exp_dir += '_feature_extract'
    
    if config['multi_anchors']:
        exp_dir += '_multi_anchors'

    os.makedirs(exp_dir, exist_ok=True)

    exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    files = set(map(int, exps))
    if len(files):
        exp_id = min(set(range(1, max(files) + 2)) - files)
    else:
        exp_id = 1

    exp_dir = os.path.join(exp_dir, str(exp_id))
    os.makedirs(exp_dir, exist_ok=True)

    json.dump(config, open(exp_dir + '/config.json', 'w'))

    return exp_dir
    
def load_weight(model, optimizer, lr_scheduler, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd['model'])
    optimizer.load_state_dict(sd['optimizer'])
    lr_scheduler.load_state_dict(sd['lr_scheduler'])
    epoch = sd['epoch']

    print('Loaded model from epoch %d\n' % (epoch))


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = {
        'dataset_type': args.dataset_type,
        'batch_size': args.batch_size,
        'optimizer': args.optim,
        'print_freq': args.print_freq,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'imgsize': (args.imgsize, args.imgsize),
        'arch': args.arch,
        'version': args.version,
        'finetune': args.finetune,
        'path': args.path,
        'train_only_car': args.train_only_car,
        'feature_extract': args.feature_extract,
        'multi_anchors': args.multi_anchors,
    }
    
    if config['dataset_type'] == 'VOC':
        train_loader, val_loader = get_train_valid_loader_voc(config, valid_size=0)
    else:
        train_loader, val_loader = get_train_valid_loader_coco(config, valid_size=0)
      
    print('Valid size = 0')
    if config['dataset_type'] == 'COCO':
        class_names = train_loader.dataset.cats
        num_classes = len(class_names)
    elif config['dataset_type'] == 'VOC':
        if config['train_only_car']:
            num_classes = 2 # background or car
        else:
            class_names = train_loader.dataset.labels.keys()
            num_classes = len(class_names)
            
    print('Num classes: {}'.format(num_classes))
        
    start_epoch = 0
    
    unnorm_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255]
    unnorm_std = [1/0.229, 1/0.224, 1/0.255]
    see_examples = 5
    for i, (imgs, targets) in enumerate(train_loader):
        if i >= see_examples - 1:
          break
      
        normalize = T.Normalize(unnorm_mean, unnorm_std)
        img, target = imgs[0], targets[0]
        img, _ = normalize(img, target)
        img = np.transpose(img, (1, 2, 0))
        
        plt.imshow(img)
        clear_output(wait=True)
        x1, y1, x2, y2 = targets[0]['boxes'][0]
        rect = patches.Rectangle((x1,y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.show()
    
        time.sleep(1)
    
    model = construct_model_detection(config, num_classes)
    model = model.to(device)
    print(model)
    
    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params,
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
                          
    if config['optimizer'].lower == 'adam':
        optimizer = optim.Adam(params,
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor = 0.1,
                                                        patience = 5,
                                                        mode = 'min')
            
    if config['finetune']:
        load_weight(model, optimizer, lr_scheduler, config['path'], device)

    best_loss = math.inf
    
    exp_dir = get_exp_dir(config)
    
    PATH_to_log_dir = exp_dir + '/runs/'
    # Declare Tensorboard writer
    writer = SummaryWriter(PATH_to_log_dir)
    print('Tensorboard is recording into folder: ' + PATH_to_log_dir + '\n')
    
    
    for epoch in range(start_epoch, config['epochs']):
        # train for one epoch, printing every 10 iterations
        epoch_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, print_freq=config['print_freq'])
        
        try:
            lr_scheduler.step()
        except:
            lr_scheduler.step(epoch_loss)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch},
                    os.path.join(exp_dir, 'best.pth'))
                    
        # if config['dataset_type'] == 'COCO' and (epoch % 10 == 0 or epoch == config['epochs'] - 1):    
        #     valres = evaluate(model, val_loader, device)
        #     # print(valres)
    
    return PATH_to_log_dir