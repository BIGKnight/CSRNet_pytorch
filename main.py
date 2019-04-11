import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from utils import *
import math
from net.model import CSRNet
from eval.eval_as_a_whole import *
import torchvision.transforms as transforms
from Dataset_processing.TrainDatasetConstructor import TrainDatasetConstructor
from Dataset_processing.EvalDatasetConstructor import EvalDatasetConstructor
from net.metrics import *
from PIL import Image
import time
torch.backends.cudnn.benchmark=True
# config
config = {
'SHANGHAITECH': 'A',
'min_RATE':10000000,
'min_MAE':10240000,
'min_MSE':10240000,
'eval_num':182,
'train_num':300,
'learning_rate': 1e-4,
'train_batch_size': 1,
'epoch': 1000,
'eval_per_step': 3000,
'mode':'whole'
}
img_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/images"
gt_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/gt_map"
img_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/images"
gt_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/gt_map"
model_save_path = "/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_baseline.pkl"
f = open("/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/logs/log_baseline.txt", "w")
# data_load
train_dataset = TrainDatasetConstructor(img_dir, gt_dir, config['train_num'], mode=config['mode'])
eval_dataset = EvalDatasetConstructor(img_dir_t, gt_dir_t, config['eval_num'], mode=config['mode'])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'])
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = CSRNet().cuda()
# net = torch.load("/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_a_whole_differ_loss_12:08_0329.pkl")
# set optimizer and estimator

optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'])
criterion = Loss().cuda()
ae_batch = AEBatch().cuda()
se_batch = SEBatch().cuda()
modules = {'model':net, 'loss':criterion, 'ae':ae_batch, 'se':se_batch}

step = 0
for epoch_index in range(config['epoch']):
    dataset = train_dataset.shuffle()
    loss_list = []
    time_per_epoch = 0
    
    if epoch_index == 200:
        config['eval_per_step'] = 1200
    if epoch_index == 300:
        config['eval_per_step'] = 300
    
    for train_img_index, train_img, train_gt in train_loader:
        if step % config['eval_per_step'] == 0:
            validate_MAE, validate_loss, validate_RMSE, time_cost = eval_model(config, eval_loader, modules, False)
            f.write('In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}, time cost = {} s\n'.format(step, epoch_index + 1, validate_loss, validate_MAE, validate_RMSE, time_cost))
            f.flush()
            
            # save model
            if config['min_MAE'] > validate_MAE:
                config['min_MAE'] = validate_MAE
                torch.save(net, model_save_path)
            
#             # return train model
 
        net.train()
        optimizer.zero_grad()
        # B
        x = train_img
        y = train_gt
        start = time.time()
        prediction = net(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        step += 1
        torch.cuda.synchronize()
        end2 = time.time()
        time_per_epoch += end2 - start
    f.write('epoch {}: time cost {}\r\n'.format(epoch_index, time_per_epoch))
    f.flush()
f.close()    
