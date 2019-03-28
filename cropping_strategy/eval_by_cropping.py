import random
import math
import numpy as np
import sys
from PIL import Image
from utils import *


def eval_model(config, eval_loader, modules, if_show_sample=False):
    net = modules['model'].eval()
    criterion = modules['loss']
    ae_batch = modules['ae']
    se_batch = modules['se']

    loss_ = []
    MAE_ = []
    MSE_ = []
    rand_number = random.randint(0, config['eval_num'] - 1)
    counter = 0
    for eval_img_index, eval_img, eval_gt, eval_data_ptc in eval_loader:
        eval_patchs = torch.squeeze(eval_img)
        eval_gt_shape = eval_gt.shape
        eval_prediction = net(eval_patchs)

        prediction_map = torch.zeros(eval_gt_shape).cuda()
        eval_patchs_shape = eval_prediction.shape
        # print(eval_patchs_shape, eval_gt_shape)
        for i in range(7):
            for j in range(7):
                start_h = math.floor(eval_patchs_shape[2] / 4)
                start_w = math.floor(eval_patchs_shape[3] / 4)
                valid_h = eval_patchs_shape[2] // 2
                valid_w = eval_patchs_shape[3] // 2
                h_pred = math.floor(3 * eval_patchs_shape[2] / 4) + (eval_patchs_shape[2] // 2) * (i - 1)
                w_pred = math.floor(3 * eval_patchs_shape[3] / 4) + (eval_patchs_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_patchs_shape[2] / 4)
                    start_h = 0
                    h_pred = 0
                elif i == 6:
                    valid_h = math.ceil(3 * eval_patchs_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_patchs_shape[3] / 4)
                    start_w = 0
                    w_pred = 0
                elif j == 6:
                    valid_w = math.ceil(3 * eval_patchs_shape[3] / 4)

                prediction_map[:, :, h_pred:h_pred + valid_h, w_pred:w_pred + valid_w] += eval_prediction[
                                                                                          i * 7 + j:i * 7 + j + 1, :,
                                                                                          start_h:start_h + valid_h,
                                                                                          start_w:start_w + valid_w]
        eval_loss = criterion(prediction_map, eval_gt).data.cpu().numpy()
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

        loss_.append(eval_loss)
        MAE_.append(batch_ae)
        MSE_.append(batch_se)
        counter += 1

    # calculate the validate loss, validate MAE and validate RMSE
    loss_ = np.reshape(loss_, [-1])
    MAE_ = np.reshape(MAE_, [-1])
    MSE_ = np.reshape(MSE_, [-1])
    validate_loss = np.mean(loss_)
    validate_MAE = np.mean(MAE_)
    validate_RMSE = np.sqrt(np.mean(MSE_))

    return validate_MAE, validate_loss, validate_RMSE

