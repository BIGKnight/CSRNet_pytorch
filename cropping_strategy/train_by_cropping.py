from net_files.model import CSRNet
from cropping_strategy.eval_by_cropping import *
from Dataset_processing.TrainDatasetConstructor import TrainDatasetConstructor
from Dataset_processing.EvalDatasetConstructor import EvalDatasetConstructor
from net_files.metrics import *
# config
config = {
'SHANGHAITECH': 'A',
'min_RATE':10000000,
'min_MAE':10240000,
'min_MSE':10240000,
'eval_num':10,
'train_num':300,
'learning_rate': 1e-5,
'train_batch_size': 1,
'epoch': 100000,
'eval_per_step': 400,
'mode':'crop'}

img_dir = "/home/zzn/part_" + config['SHANGHAITECH'] + "_final/train_data/images"
gt_dir = "/home/zzn/part_" + config['SHANGHAITECH'] + "_final/train_data/gt_map"
img_dir_t = "/home/zzn/part_" + config['SHANGHAITECH'] + "_final/test_data/images"
gt_dir_t = "/home/zzn/part_" + config['SHANGHAITECH'] + "_final/test_data/gt_map"
model_save_path = "/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/model_a_crop.pkl"
# data_load
train_dataset = TrainDatasetConstructor(img_dir, gt_dir, config['train_num'], mode=config['mode'])
eval_dataset = EvalDatasetConstructor(img_dir_t, gt_dir_t, config['eval_num'], mode=config['mode'])
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'])
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = CSRNet().to(cuda_device)
# set optimizer and estimator
criterion = Loss().to(cuda_device)
optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'])
ae_batch = AEBatch().to(cuda_device)
se_batch = SEBatch().to(cuda_device)
modules = {'model':net, 'loss':criterion, 'ae':ae_batch, 'se':se_batch}
step = 0
for epoch_index in range(config['epoch']):
    dataset = train_dataset.shuffle()
    for train_img_index, train_img, train_gt, data_ptc in train_loader:
        if step % config['eval_per_step'] == 0:
            validate_MAE, validate_loss, validate_RMSE = eval_model(config, eval_loader, modules, True)
            sys.stdout.write(
                'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, epoch_index + 1, validate_loss,
                                                                                  validate_MAE, validate_RMSE))
            sys.stdout.flush()

            # save model
            if config['min_MAE'] > validate_MAE:
                config['min_MAE'] = validate_MAE
                torch.save(net, model_save_path)

            # return train model

        net.train()
        optimizer.zero_grad()
        # B
        x = train_img
        y = train_gt
        prediction = net(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        step += 1
        sys.stdout.write('In step {}\r'.format(step))
        sys.stdout.flush()
