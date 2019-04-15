import sys
import numpy as np
import random
from utils import *
import math
from model import CSRNet_HDC
from eval.eval_as_a_whole import eval_model
from Dataset_processing.TrainDatasetConstructor import TrainDatasetConstructor
from Dataset_processing.EvalDatasetConstructor import EvalDatasetConstructor
from net.metrics import Loss, AEBatch, SEBatch
from PIL import Image
import time
# torch.backends.cudnn.benchmark=True
# this order will test different cudnn algorithm for choosing the fastest one, 
# which sometimes give rise to extra gpu memory cost because it need to 
# test different algorithm simultaneously
# %matplotlib inline

# config
config = {
'SHANGHAITECH': 'A',
'min_RATE':10000000,
'min_MAE':10240000,
'min_MSE':10240000,
'eval_num':182,
'train_num':300,
'learning_rate': 1e-7,
'train_batch_size': 1,
'epoch': 1000,
'eval_per_step': 300,
'mode':'whole'
}
img_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/images"
gt_dir = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/train_data/gt_map"
img_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/images"
gt_dir_t = "/home/zzn/Documents/Datasets/part_" + config['SHANGHAITECH'] + "_final/test_data/gt_map"
model_save_path = "/home/zzn/PycharmProjects/CSRNet_pytorch-master/checkpoints/model_2x2_sgd.pkl"
f = open("/home/zzn/PycharmProjects/CSRNet_pytorch-master/logs/log_2x2_sgd.txt", "w")


# data_load
train_dataset = TrainDatasetConstructor(img_dir, gt_dir, config['train_num'], mode=config['mode'], if_random_hsi=True, if_flip=True)
eval_dataset = EvalDatasetConstructor(img_dir_t, gt_dir_t, config['eval_num'], mode=config['mode'])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'])
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = CSRNet_HDC().cuda()
# net = torch.load("/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_a_whole_differ_loss_12:08_0329.pkl")
# set optimizer and estimator

optimizer = torch.optim.SGD(net.parameters(), config['learning_rate'], 0.95, 5e-4)
# optimizer = torch.optim.Adam(net.parameters(), 1e-6, weight_decay=5e-4)
criterion = Loss().cuda()
ae_batch = AEBatch().cuda()
se_batch = SEBatch().cuda()
modules = {'model':net, 'loss':criterion, 'ae':ae_batch, 'se':se_batch}

step = 0
# torch.cuda.empty_cache()
for epoch_index in range(config['epoch']):
    dataset = train_dataset.shuffle()
    loss_list = []
    time_per_epoch = 0
    
    for train_img_index, train_img, train_gt in train_loader:
        if step % config['eval_per_step'] == 0:
            validate_MAE, validate_RMSE, time_cost = eval_model(config, eval_loader, modules, False)
            f.write('In step {}, epoch {}, MAE = {}, MSE = {}, time cost = {}.\n'.format(step, epoch_index + 1, validate_MAE, validate_RMSE, time_cost))
            f.flush()
            
            #save model
            if config['min_MAE'] > validate_MAE:
                config['min_MAE'] = validate_MAE
                torch.save(net, model_save_path)
#             # return train model
        net.train()
        optimizer.zero_grad()
        # B
        x = train_img
        y = train_gt
        prediction = net(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        step += 1
