import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.scripts.se_resnet import se_resnet50
from models.scripts.se_linear_resnet import se_linear_resnet34, se_linear_resnet50
from models.scripts.cbam_resnet import cbam_resnet50
from models.scripts.cbam_linear_resnet import cbam_linear_resnet34, cbam_linear_resnet50
from models.scripts.network_gradcam import NetworkGradCam

def construct_model_detection(config, num_classes=2):
    if config['arch'] == 'resnet50_fpn':
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return model
        
    if config['arch'] == 'vgg19_bn':
        features = torchvision.models.vgg19_bn(pretrained=True).features
    elif config['arch'] == 'resnet18':
        base = torchvision.models.resnet18(pretrained=True)
        # isolate the feature blocks
        features = nn.Sequential(base.conv1,
                                 base.bn1,
                                 base.relu,
                                 base.maxpool,
                                 base.layer1, 
                                 base.layer2, 
                                 base.layer3, 
                                 base.layer4)
    elif config['arch'] == 'resnet50':
        base = torchvision.models.resnet50(pretrained=True)
        # isolate the feature blocks
        features = nn.Sequential(base.conv1,
                                 base.bn1,
                                 base.relu,
                                 base.maxpool,
                                 base.layer1, 
                                 base.layer2, 
                                 base.layer3, 
                                 base.layer4)
    else:
        print("Invalid model name, exiting...")
        exit()


    backbone = features

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For vgg19_bn, it's 512
    #                                For resnet50, it's 2048
    # so we need to add it here
    if hasattr(base, 'fc'):
        num_features = base.fc.in_features
    else: # mobile net v2 / vgg19
        num_features = base.classifier[-1].in_features

    backbone.out_channels = num_features

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.25, 0.5, 1.0, 1.5, 2.0, 2.5),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

    return model

def construct_model_classification(config, num_classes):
    if config['arch'] == 'resnext50':
        feature_extractor = torchvision.models.resnext50_32x4d(pretrained=True)
    elif config['arch'] == 'resnet18':
        feature_extractor = torchvision.models.resnet18(pretrained=True)
    elif config['arch'] == 'resnet34':
        feature_extractor = torchvision.models.resnet34(pretrained=True)
    elif config['arch'] == 'resnet50':
        feature_extractor = torchvision.models.resnet50(pretrained=True)
    elif config['arch'] == 'se_resnet50':
        feature_extractor = se_resnet50(pretrained=True)
    elif config['arch'] == 'se_linear_resnet34':
        feature_extractor = se_linear_resnet34(pretrained=True)
    elif config['arch'] == 'se_linear_resnet50':
        feature_extractor = se_linear_resnet50(pretrained=True)
    elif config['arch'] == 'cbam_resnet50':
        feature_extractor = cbam_resnet50(pretrained=True)
    elif config['arch'] == 'cbam_linear_resnet34':
        feature_extractor = cbam_linear_resnet34(pretrained=True)
    elif config['arch'] == 'cbam_linear_resnet50':
        feature_extractor = cbam_linear_resnet50(pretrained=True)
    elif config['arch'] == 'vgg19':
        feature_extractor = torchvision.models.vgg19(pretrained=True)
    elif config['arch'] == 'vgg19_bn':
        feature_extractor = torchvision.models.vgg19_bn(pretrained=True)
    elif config['arch'] == 'se_vgg19_bn':
        feature_extractor = se_vgg19_bn(pretrained=True)
    else:
        print("Invalid model name, exiting...")
        exit()

    
    if config['feature_extract']:
        for param in feature_extractor.parameters():
            param.requires_grad = False
            
    if config['grad_cam']:
        model = NetworkGradCam(feature_extractor, num_classes, config['feature_extract'])
    
    return model