import math
import torch
from torch import nn
from torch.nn import functional as F


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, version = 'b0'):
        super(efficientN, self).__init__()
        from efficientnet_pytorch import EfficientNet

        # load pretrained EfficientNet B3
        self.model_ft = EfficientNet.from_pretrained(f'efficientnet-{version}')
        # self.model_ft = EfficientNet.from_name(f'efficientnet-{version}')

        for child in self.model_ft.children():

          for param in child.parameters():
            param.requires_grad = False

        # re-init last conv layer and last fc layer to fit with dataset
        in_channels = self.model_ft._conv_head.in_channels
        out_channels = self.model_ft._conv_head.out_channels
        num_ftrs = self.model_ft._fc.in_features

        self.model_ft._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.model_ft._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.010000000000000009, eps = 0.001)
        self.model_ft._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model_ft(x)
