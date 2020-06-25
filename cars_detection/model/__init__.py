import torchvision
import torch.nn as nn
import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.se_resnet import se_resnet50
from model.se_linear_resnet import se_linear_resnet50, se_linear_resnet34
from model.cbam_resnet import cbam_resnet50
from model.cbam_linear_resnet import cbam_linear_resnet50, cbam_linear_resnet34


def construct_model_detection(config, num_classes):
    # if config['arch'] == 'resnext50':
    #     base = torchvision.models.resnext50_32x4d(pretrained=True)
    # elif config['arch'] == 'resnet34':
    #     base = torchvision.models.resnet34(pretrained=True)
    
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
    elif config['arch'] == 'mobilenetv2':
        features = torchvision.models.mobilenet_v2(pretrained=False).features
    else:
        print("Invalid model name, exiting...")
        exit()

    if config['feature_extract']:
        for name, param in features.named_parameters():
            param.requires_grad = False

    backbone = features

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For vgg19_bn, it's 512
    #                                For resnet18, it's 512
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
    
    if config['train_only_car']:
        print('Car only anchor')
        anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.25, 0.5, 1.0, 1.5, 2.0, 2.5),))
    else:
        print('Models anchor')
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
                                        
    if config['multi_anchors']:
        print('Multi anchors')
        anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),))
                            
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
    
class NetworkGradCam(nn.Module):
    def __init__(self, feature_extractor, num_classes, feature_extract):
        super().__init__()

        # gradient placeholder
        self.gradient = None

        if hasattr(feature_extractor, 'fc'): # resnets
            # isolate the feature blocks
            self.features = nn.Sequential(feature_extractor.conv1,
                                          feature_extractor.bn1,
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                          feature_extractor.layer1, 
                                          feature_extractor.layer2, 
                                          feature_extractor.layer3, 
                                          feature_extractor.layer4)
            # average pooling layer
            self.avgpool = feature_extractor.avgpool

            in_features = feature_extractor.fc.in_features
            # classifier
            self.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        else: # vgg
            self.features = feature_extractor.features
            if feature_extract:
                self.features.eval()

            # average pooling layer
            self.avgpool = feature_extractor.avgpool

            in_features_first = feature_extractor.classifier[0].in_features
            in_features_last = feature_extractor.classifier[-1].in_features
            self.classifier = nn.Sequential(
                    nn.Linear(in_features=in_features_first, out_features=in_features_last, bias=True),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=in_features_last, out_features=in_features_last, bias=True),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(in_features=in_features_last, out_features=num_classes, bias=True)
                )
            
    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):
        # extract the features
        x = self.features(x)
        
        # # register the hook
        # h = x.register_hook(self.activations_hook)
        
        # complete the forward pass
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
        
def construct_model_classification(config, num_classes):
    print(config['arch'])
    if config['arch'] == 'resnext50':
        feature_extractor = torchvision.models.resnext50_32x4d(pretrained=True)
    elif config['arch'] == 'resnet34':
        feature_extractor = torchvision.models.resnet34(pretrained=True)
    elif config['arch'] == 'resnet50':
        feature_extractor = torchvision.models.resnet50(pretrained=True)
    elif config['arch'] == 'vgg19':
        feature_extractor = torchvision.models.vgg19(pretrained=True)
    elif config['arch'] == 'vgg19_bn':
        feature_extractor = torchvision.models.vgg19_bn(pretrained=True)
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
    elif config['arch'] == 'se_vgg19_bn':
        feature_extractor = se_vgg19_bn(pretrained=True)
    else:
        print("Invalid model name, exiting...")
        exit()

    if config['grad_cam']:
        model = NetworkGradCam(feature_extractor, num_classes, config['feature_extract'])
    
    # model = feature_extractor

    return model