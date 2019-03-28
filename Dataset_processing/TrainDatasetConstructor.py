from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as data
import random
import time
from utils import HSI_Calculator
import scipy.io as scio
import math


class TrainDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 train_num,
                 mode
                 ):
        self.train_num = train_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.permulation = np.random.permutation(self.train_num)
        self.calcu = HSI_Calculator()
        self.mode = mode
        for i in range(self.train_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            height = img.size[1]
            width = img.size[0]
            img = transforms.Resize([math.ceil(height / 128) * 128, (math.ceil(width / 128) * 128)])(img)
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):
        if self.mode == 'crop':
            start = time.time()
            img, gt_map = self.imgs[self.permulation[index]]
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            flip_random = random.random()
            if flip_random > 0.5:
                img = F.hflip(img)
                gt_map = F.hflip(gt_map)
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            img_shape = img.shape  # C, H, W
            random_h = random.randint(0, (3 * img_shape[1] // 4) - 1)
            random_w = random.randint(0, (3 * img_shape[2] // 4) - 1)
            patch_height = img_shape[1] // 4
            patch_width = img_shape[2] // 4
            img = img[:, random_h:random_h + patch_height, random_w:random_w + patch_width]
            gt_map = gt_map[:, random_h // 8:random_h // 8 + patch_height // 8, random_w // 8:random_w // 8 + patch_width // 8]
            end = time.time()
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img).cuda()
            gt_map = gt_map.cuda()
            return self.permulation[index] + 1, img, gt_map, (end - start)

        elif self.mode == 'whole':
            start = time.time()
            img, gt_map = self.imgs[self.permulation[index]]
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            flip_random = random.random()
            if flip_random > 0.5:
                img = F.hflip(img)
                gt_map = F.hflip(gt_map)
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            end = time.time()
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            return self.permulation[index] + 1, img, gt_map, (end - start)

    def __len__(self):
        return self.train_num

    def shuffle(self):
        self.permulation = np.random.permutation(self.train_num)
        return self
    

