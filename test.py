from utils import *
from Dataset_processing.TrainDatasetConstructor import DatasetConstructor
from net_files import metrics
import time
SHANGHAITECH = "B"

# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/gt_map"
dataset = DatasetConstructor(img_dir, gt_dir, 316, 10, False)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

mae_metrics = []
mse_metrics = []
net = torch.load("/home/zzn/Downloads/CSRNet_pytorch-master/checkpoints/the_best_model_onShanghaiB.pkl").to(cuda_device)
gt_process_model = GroundTruthProcess(1, 1, 8).to(cuda_device)
net.eval()


ae_batch = metrics.AEBatch().to(cuda_device)
se_batch = metrics.SEBatch().to(cuda_device)

start = time.time()

for real_index, test_img, test_gt, test_time_cost in test_loader:
    print(test_time_cost.numpy())
    image_shape = test_img.shape
    patch_height = int(image_shape[3])
    patch_width = int(image_shape[4])
    # B
    #   start = time.time()
    eval_x = test_img.view(49, 3, patch_height, patch_width)
    #     eval_y = test_gt.view(1, 1, patch_height * 4, patch_width * 4)

    #     eval_groundtruth = gt_process_model(eval_y)
    patch_height = int(patch_height / 8)
    patch_width = int(patch_width / 8)
    start = time.time()
    eval_prediction = net(eval_x)
    end = time.time()
    print(end - start)
