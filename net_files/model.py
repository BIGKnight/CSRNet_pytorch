import torch.nn as nn
from torchvision import models
import torch.nn.functional as functional


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.front_end = nn.Sequential((list(list(models.vgg16(True).children())[0].children())[0:23]))
        self.back_end = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        img_shape = x.shape
        x = self.front_end(x)
        x = self.back_end(x)
        x = self.output_layer(x)
        # x= functional.interpolate(x, img_shape[2:], mode="bilinear", align_corners=True)
        return x

    def _initialize_weights(self):
        for m in self.back_end.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
