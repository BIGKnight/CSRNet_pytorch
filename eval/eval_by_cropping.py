import random
import math
import numpy as np
import sys
from PIL import Image
import torch
from utils import show
import time


def eval_model(config, eval_loader, modules, if_show_sample=False):
    net = modules['model'].eval()
    ae_batch = modules['ae']
    se_batch = modules['se']

    MAE_ = []
    MSE_ = []
    time_cost = 0
    rand_number = random.randint(0, config['eval_num'] - 1)
    counter = 0
    for eval_img_index, eval_img, eval_gt, img_height, img_width in eval_loader:
        start = time.time()
        eval_patchs = torch.squeeze(eval_img)
        height = img_height.numpy()[0]
        width = img_width.numpy()[0]
        patch_h_num = (height - 400) // 200 + 1
        patch_w_num = (width - 400) // 200 + 1
        with torch.no_grad():
            eval_gt_shape = eval_gt.shape
            eval_prediction = net(eval_patchs)
            torch.cuda.empty_cache()
        prediction_map = torch.zeros(eval_gt_shape).cuda()
        eval_patchs_shape = eval_prediction.shape
        # print(eval_patchs_shape, eval_gt_shape)
        for i in range(patch_h_num):
            for j in range(patch_w_num):
#                 start_h = i * 50
#                 start_w = math.floor(eval_patchs_shape[3] / 4)
                valid_h = 50
                valid_w = 50
                h_pred = i * 25
                w_pred = j * 25
                prediction_map[:, :, h_pred:h_pred + valid_h, w_pred:w_pred + valid_w] += eval_prediction[i * patch_w_num + j:i * patch_w_num + j + 1, :, :, :]
        
        internal_margin_h = (eval_gt_shape[2] - 50) 
        internal_margin_w = (eval_gt_shape[3] - 50) 
        if internal_margin_h > 0:
            prediction_map[:, :, 25:25 + internal_margin_h, :] /= 2
        if internal_margin_w > 0:
            prediction_map[:, :, :, 25:25 + internal_margin_w] /= 2
                
                
        torch.cuda.synchronize()
        end = time.time()
        time_cost += (end - start)
        
        batch_ae = ae_batch(prediction_map, eval_gt).data.cpu().numpy()
        batch_se = se_batch(prediction_map, eval_gt).data.cpu().numpy()

        validate_pred_map = np.squeeze(prediction_map.permute(0, 2, 3, 1).data.cpu().numpy())
        validate_gt_map = np.squeeze(eval_gt.permute(0, 2, 3, 1).data.cpu().numpy())
        gt_counts = np.sum(validate_gt_map)
        pred_counts = np.sum(validate_pred_map)
        # random show 1 sample
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

