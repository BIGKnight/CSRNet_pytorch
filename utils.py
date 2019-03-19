import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class GroundTruthProcess(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GroundTruthProcess, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.FloatTensor(torch.ones(out_channels, in_channels, kernel_size, kernel_size)).cuda()

    def forward(self, x):
        result = F.conv2d(x, self.kernel, bias=None, stride=self.kernel_size, padding=0)
        return result


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()


class HSI_Calculator(nn.Module):
    def __init__(self):
        super(HSI_Calculator, self).__init__()

    def forward(self, image):
        image = transforms.ToTensor()(image)
        I = torch.mean(image)
        Sum = image.sum(0)
        Min = 3 * image.min(0)[0]
        S = (1 - Min.div(Sum.clamp(1e-6))).mean()
        numerator = (2 * image[0] - image[1] - image[2]) / 2
        denominator = ((image[0] - image[1]) ** 2 + (image[0] - image[2]) * (image[1] - image[2])).sqrt()
        theta = (numerator.div(denominator.clamp(1e-6))).clamp(-1 + 1e-6, 1 - 1e-6).acos()
        logistic_matrix = (image[1] - image[2]).ceil()
        H = (theta * logistic_matrix + (1 - logistic_matrix) * (360 - theta)).mean() / 360
        return H, S, I
