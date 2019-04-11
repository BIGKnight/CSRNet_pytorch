import cv2
import numpy as np
import scipy
import scipy.io as scio
from PIL import Image
import time
import math


def get_density_map_gaussian(H, W, ratio_h, ratio_w,  points, adaptive_kernel=False, fixed_value=15):
    h = H // 8 
    w = W // 8
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    # if adaptive_kernel:
    #     # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
    #     leafsize = 2048
    #     tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    #     distances = tree.query(points, k=4)[0]

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
        # if num_gt > 1:
        #     if adaptive_kernel:
        #         sigma = int(np.sum(distances[idx][1:4]) // 3 * 0.3)
        #     else:
        #         sigma = fixed_value
        # else:
        #     sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = fixed_value
        sigma = max(1, sigma)

        gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map


# 22, 37
if __name__ == "__main__":
    image_dir_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/images"
    ground_truth_dir_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/ground_truth"
    output_gt_dir = "/home/zzn/Documents/Datasets/part_A_final/train_data/gt_map_crop"
    for i in range(300):
        img_path = image_dir_path + "/IMG_" + str(i + 1) + ".jpg"
        gt_path = ground_truth_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
        img = Image.open(img_path)
        height = img.size[1]
        width = img.size[0]
        points = scio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        
        resize_height = height
        resize_width = width
        if resize_height <= 400:
            tmp = resize_height
            resize_height = 400
            resize_width = (resize_height / tmp) * resize_width
            
        if resize_width <= 400:
            tmp = resize_width
            resize_width = 400
            resize_height = (resize_width / tmp) * resize_height
            
        resize_height = math.ceil(resize_height / 200) * 200
        resize_width = math.ceil(resize_width / 200) * 200
        
        ratio_h = (resize_height / (8 * height))
        ratio_w = (resize_width / (8 * width))
        # print(height, width, ratio_h, ratio_w)
        gt = get_density_map_gaussian(resize_height, resize_width, ratio_h, ratio_w, points, False, 1)
        gt = np.reshape(gt, [resize_height // 8, resize_width // 8])  # transpose into w, h
        np.save(output_gt_dir + "/GT_IMG_" + str(i + 1), gt)
        print("complete!")