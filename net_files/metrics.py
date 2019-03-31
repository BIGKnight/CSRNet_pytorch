import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as functional


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.MSELoss = nn.MSELoss(size_average=False)

    def forward(self, estimated_density_map, gt_map):
        return self.MSELoss(estimated_density_map, gt_map)

class PatchGaussianLoss(nn.Module):
    def __init__(self, sigma):
        super(PatchGaussianLoss, self).__init__()
        self.gaussian_radius = sigma * 3
        self.gaussian_map = torch.FloatTensor(np.multiply(
            cv2.getGaussianKernel(self.gaussian_radius * 2 + 1, sigma),
            cv2.getGaussianKernel(self.gaussian_radius * 2 + 1, sigma).T
        )).view(1, 1, self.gaussian_radius * 2 + 1, self.gaussian_radius * 2 + 1).cuda()
        self.MSELoss = nn.MSELoss(size_average=False)

    def forward(self, estimated_density_map, gt_map):
        x_pred = functional.conv2d(estimated_density_map, self.gaussian_map, bias=None, stride=1, padding=self.gaussian_radius)
        y_gt = functional.conv2d(gt_map, self.gaussian_map, bias=None, stride=1, padding=self.gaussian_radius)
        return self.MSELoss(estimated_density_map, gt_map)


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
