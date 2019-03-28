import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.mul(torch.div(torch.mean((estimated_density_map - gt_map) ** 2, dim=(0, 1, 2, 3)), 2.), 10000)


class AEBatch(nn.Module):
    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.abs(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)))


class SEBatch(nn.Module):
    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.pow(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)), 2)
