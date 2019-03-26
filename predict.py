import sys
import numpy as np
import random
from utils import *
from net_files.model import CSRNet
from Dataset_processing.TrainDatasetConstructor import TrainDatasetConstructor
from Dataset_processing.EvalDatasetConstructor import EvalDatasetConstructor
from net_files.metrics import *
from PIL import Image
MAE = 10240000
SHANGHAITECH = "A"
# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/gt_map"

img_dir_t = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/images"
gt_dir_t = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/gt_map"

train_dataset = TrainDatasetConstructor(img_dir, gt_dir, 300, mode='whole')
eval_dataset = EvalDatasetConstructor(img_dir_t, gt_dir_t, 10, mode='whole')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = CSRNet().to(cuda_device)
# set optimizer and estimator
criterion = Loss().to(cuda_device)
optimizer = torch.optim.Adam(net.parameters(), 1e-5)
ae_batch = AEBatch().to(cuda_device)
se_batch = SEBatch().to(cuda_device)
step = 0
for epoch_index in range(10000):
    train_dataset = train_dataset.shuffle()
    for train_img_index, train_img, train_gt, data_ptc in train_loader:
        # eval per 100 batch
        if step % 400 == 0:
            net.eval()
            loss_ = []
            MAE_ = []
            MSE_ = []
            rand_number = random.randint(0, 9)
            counter = 0
            for eval_img_index, eval_img, eval_gt, eval_data_ptc in eval_loader:
                eval_prediction = net(eval_img)
                eval_loss = criterion(eval_prediction, eval_gt).data.cpu().numpy()
                batch_ae = ae_batch(eval_prediction, eval_gt).data.cpu().numpy()
                batch_se = se_batch(eval_prediction, eval_gt).data.cpu().numpy()
                validate_pred_map = np.squeeze(eval_prediction.permute(0, 2, 3, 1).data.cpu().numpy())
                validate_gt_map = np.squeeze(eval_gt.permute(0, 2, 3, 1).data.cpu().numpy())
                gt_counts = np.sum(validate_gt_map)
                pred_counts = np.sum(validate_pred_map)
                # random show 1 sample
                if rand_number == counter:
                    origin_image = Image.open("/home/zzn/part_" + SHANGHAITECH + "_final/test_data/images/IMG_" + str(
                        eval_img_index.numpy()[0]) + ".jpg")
                    show(origin_image, validate_gt_map, validate_pred_map, eval_img_index.numpy()[0])
                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

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

            sys.stdout.write('In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, epoch_index + 1,
                                                                                             validate_loss,
                                                                                             validate_MAE,
                                                                                             validate_RMSE))
            sys.stdout.flush()
            # save model
            if MAE > validate_MAE:
                MAE = validate_MAE
                torch.save(net, "/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_mae_a.pkl")

        # return train model

        net.train()
        optimizer.zero_grad()
        # B
        x = train_img
        y = train_gt
        prediction = net(x)
        loss = criterion(prediction, train_gt)
        loss.backward()
        optimizer.step()
        step += 1
