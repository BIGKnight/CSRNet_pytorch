from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import time
from utils import HSI_Calculator
import torch
import math


class EvalDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 validate_num,
                 mode
                 ):
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.calcu = HSI_Calculator()
        self.mode = mode
        for i in range(self.validate_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            height = img.size[1]
            width = img.size[0]
#             resize_height = height
#             resize_width = width
#             if resize_height <= 400:
#                 tmp = resize_height
#                 resize_height = 400
#                 resize_width = (resize_height / tmp) * resize_width

#             if resize_width <= 400:
#                 tmp = resize_width
#                 resize_width = 400
#                 resize_height = (resize_width / tmp) * resize_height
            
#             resize_height = math.ceil(resize_height / 200) * 200
#             resize_width = math.ceil(resize_width / 200) * 200

            resize_height = math.ceil(height / 8) * 8
            resize_width = math.ceil(width / 8) * 8
            
            img = transforms.Resize([resize_height, resize_width])(img)
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img, gt_map = self.imgs[index]
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            imgs = []
            patch_h_num = (img.shape[1] - 400) // 200 + 1
            patch_w_num = (img.shape[2] - 400) // 200 + 1
            for i in range(patch_h_num):
                x = i * 200
                for j in range(patch_w_num):
                    y = j * 200
                    imgs.append(img[:, x:x+400, y:y+400])
            imgs = torch.stack(imgs)
            return index + 1, imgs, gt_map, img_shape[1], img_shape[2]

        elif self.mode == 'whole':
            img, gt_map = self.imgs[index]
#             H,S,I = self.calcu(img)
#             if I.numpy() < 0.28:
#                 img = F.adjust_brightness(img, 0.37 / I.numpy())
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            return index + 1, img.cuda(), gt_map.cuda()

    def __len__(self):
        return self.validate_num
