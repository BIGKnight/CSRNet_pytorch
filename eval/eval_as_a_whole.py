import random
import math
import numpy as np
import sys
from PIL import Image
import time
import torch
from utils import show


def eval_model(config, eval_loader, modules, if_show_sample=False):
    net = modules['model'].eval()
    ae_batch = modules['ae']
    se_batch = modules['se']

    MAE_ = []
    MSE_ = []
    rand_number = random.randint(0, config['eval_num'] - 1)
    counter = 0
    time_cost = 0
    for eval_img_index, eval_img, eval_gt in eval_loader:
        eval_patchs = eval_img
        eval_gt_shape = eval_gt.shape
        start = time.time()
        with torch.no_grad():
            eval_prediction = net(eval_patchs)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        end = time.time()
        time_cost += (end - start)
        batch_ae = ae_batch(eval_prediction, eval_gt).data.cpu().numpy()
        batch_se = se_batch(eval_prediction, eval_gt).data.cpu().numpy()

        validate_pred_map = np.squeeze(eval_prediction.permute(0, 2, 3, 1).data.cpu().numpy())
        validate_gt_map = np.squeeze(eval_gt.permute(0, 2, 3, 1).data.cpu().numpy())
        gt_counts = np.sum(validate_gt_map)
        pred_counts = np.sum(validate_pred_map)
        if rand_number == counter and if_show_sample:
            origin_image = Image.open("/home/zzn/part_" + config['SHANGHAITECH'] + "_final/test_data/images/IMG_" + str(eval_img_index.numpy()[0]) + ".jpg")
            show(origin_image, validate_gt_map, validate_pred_map, eval_img_index.numpy()[0])
            sys.stdout.write('The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))
        MAE_.append(batch_ae)
        MSE_.append(batch_se)
        counter += 1

    # calculate the validate loss, validate MAE and validate RMSE
    MAE_ = np.reshape(MAE_, [-1])
    MSE_ = np.reshape(MSE_, [-1])
    validate_MAE = np.mean(MAE_)
    validate_RMSE = np.sqrt(np.mean(MSE_))

    return validate_MAE, validate_RMSE, time_cost
