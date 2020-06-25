import torch.nn as nn
import torch
from torchvision.models import ResNet
# from model.se_layer import SELayer
from model.conv import conv1x1, conv3x3

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = channels // reduction
        self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        b, c, _, _ = x.size()
        # squeeze
        x = self.avg_pool(x).view(b, c)
        # excitation
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(b, c, 1, 1)
        
        return module_input * x.expand_as(module_input)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, groups=groups, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
def se_linear_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model with SEBasicBlock.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    # model.avgpool = nn.AdaptiveAvgPool2d(7)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth', progress=True)
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    # if pretrained:
    #     # model.load_state_dict(load_state_dict_from_url(
    #     #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    #     model.load_state_dict(torch.load("./logs/seresnet50-60a8950a85b2b.pkl"), strict=False)
        
    return model
        
def se_linear_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with SEBottleneck.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    # model.avgpool = nn.AdaptiveAvgPool2d(7)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    # if pretrained:
    #     # model.load_state_dict(load_state_dict_from_url(
    #     #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    #     model.load_state_dict(torch.load("./logs/seresnet50-60a8950a85b2b.pkl"), strict=False)
        
    return model