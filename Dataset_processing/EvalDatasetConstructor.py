from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import time
from utils import HSI_Calculator
import torch


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
        start = time.time()
        self.mode = mode
        for i in range(self.validate_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])
        end = time.time()

    def __getitem__(self, index):
        if self.mode == 'crop':
            start = time.time()
            img, gt_map = self.imgs[index]
            # H, S, I = self.calcu(img)
            # I = I.numpy()
            # if I < 0.432:
            #     img = F.adjust_brightness(img, 0.432 / I)
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W

            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            padding_h = img_shape[1] % 8
            padding_w = img_shape[2] % 8
            pads = [padding_w // 2,
                    padding_w - padding_w // 2,
                    padding_h // 2,
                    padding_h - padding_h // 2]
            img = functional.pad(
                img,
                pads,
                value=0.
            )  # left, right, up, down

            patch_height = (img_shape[1] + padding_h) // 4
            patch_width = (img_shape[2] + padding_w) // 4
            imgs = []
            for i in range(7):
                for j in range(7):
                    start_h = (patch_height // 2) * i
                    start_w = (patch_width // 2) * j
                    # print(img.shape, start_h, start_w, patch_height, patch_width)
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
            imgs = torch.stack(imgs)
            end = time.time()
            return index + 1, imgs, gt_map, (end - start)

        elif self.mode == 'whole':
            start = time.time()
            img, gt_map = self.imgs[index]
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            end = time.time()
            return index + 1, img, gt_map, (end - start)

    def __len__(self):
        return self.validate_num
