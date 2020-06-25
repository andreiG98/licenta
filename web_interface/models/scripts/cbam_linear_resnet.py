from models.scripts.conv import conv1x1, conv3x3

import torch
import torch.nn as nn
from torchvision.models import ResNet

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class MLP(nn.Module):
    """
    Multilayer perceptron block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        
        mid_channels = channels // reduction_ratio
        self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        
        return x


class ChannelAttention(nn.Module):
    """
    CBAM channel attention block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(channels=channels, reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out)
        
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out)
        
        att = avg_out + max_out
        att = self.sigmoid(att)
        
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        
        return x


class SpatialAttention(nn.Module):
    """
    CBAM spatial channel block.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(att)
        
        att = self.sigmoid(att)
        x = x * att
        
        return x


class CBAMBlock(nn.Module):
    """
    CBAM attention block for CBAM-ResNet.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.ch_att = ChannelAttention(channels=channels, reduction_ratio=reduction_ratio)
        self.sp_att = SpatialAttention()

    def forward(self, x):
        x = self.ch_att(x)
        x = self.sp_att(x)
        
        return x

class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.atte = CBAMBlock(planes * self.expansion)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.atte(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, groups=groups, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.atte = CBAMBlock(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.atte(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def cbam_linear_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model with CBAMBasicBlock.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CBAMBasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth', progress=True)
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
        
    return model
        
def cbam_linear_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model with CBAMBottleneck.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
        
    return model
