import math
import sys
import time
import torch
import numpy as np

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # lr_scheduler = None
    
    # if epoch == 0:
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'image_path'} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    loss = str(metric_logger.meters['loss'])
    loss = loss.split(' ')[1].split('(')[1].split(')')[0]
    loss = float(loss)
    
    loss_classifier = str(metric_logger.meters['loss_classifier'])
    loss_classifier = loss_classifier.split(' ')[1].split('(')[1].split(')')[0]
    loss_classifier = float(loss_classifier)
    
    loss_box_reg = str(metric_logger.meters['loss_box_reg'])
    loss_box_reg = loss_box_reg.split(' ')[1].split('(')[1].split(')')[0]
    loss_box_reg = float(loss_box_reg)
    
    loss_objectness = str(metric_logger.meters['loss_objectness'])
    loss_objectness = loss_objectness.split(' ')[1].split('(')[1].split(')')[0]
    loss_objectness = float(loss_objectness)
    
    loss_rpn_box_reg = str(metric_logger.meters['loss_rpn_box_reg'])
    loss_rpn_box_reg = loss_rpn_box_reg.split(' ')[1].split('(')[1].split(')')[0]
    loss_rpn_box_reg = float(loss_rpn_box_reg)
    
    writer.add_scalar('Train/Loss', loss, epoch)
    writer.add_scalar('Train/Loss Classifier', loss_classifier, epoch)
    writer.add_scalar('Train/Loss Box Regression', loss_box_reg, epoch)
    writer.add_scalar('Train/Loss Objectness', loss_objectness, epoch)
    writer.add_scalar('Train/Loss RPN Box Regression', loss_rpn_box_reg, epoch)
    writer.flush()    

    return loss_value


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, model_cl=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print('Targets after')
        print(targets)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        
        # bbox = output['']
        print('Outputs')
        print(outputs)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    return coco_evaluator