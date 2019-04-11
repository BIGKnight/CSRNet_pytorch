import torch.nn as nn
from torchvision import models
import torch.nn.functional as functional
import time
import torch
from dilated_conv2d.dilated_conv2d_wrapper import BasicDilatedConv2D

# class BasicDilatedConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
#         super(BasicDilatedConv2d, self).__init__()
#         self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32))
#         self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
#         self.padding = padding
#         self.dilation = dilation
#         nn.init.xavier_uniform_(self.weight, gain=1)
        
#     def forward(self, x):
#         return torch.nn.functional.conv2d(x, self.weight, self.bias, padding=self.padding, dilation=self.dilation)

class CSRNet_HDC(nn.Module):
    def __init__(self):
        super(CSRNet_HDC, self).__init__()
        self.seen = 0
        self.backend_feat = [(512, 3), (512, 2), (512, 1), (256, 1), (128, 2), (64, 3)]
        self.front_end = nn.Sequential(*(list(list(models.vgg16(True).children())[0].children())[0:23]))
        self.back_end = make_layers(self.backend_feat, in_channels=512)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
#         for m in self.back_end.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        img_shape = x.shape
        front_end = self.front_end(x)
        back_end = self.back_end(front_end)
        output = self.output_layer(back_end)
        # x= functional.interpolate(x, img_shape[2:], mode="bilinear", align_corners=True)
        return output


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v, atrous in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = BasicDilatedConv2D(in_channels, v, kernel_size=3, padding=atrous, dilation=atrous)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
