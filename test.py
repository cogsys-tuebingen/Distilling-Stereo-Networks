from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
# import skimage
# import skimage.io
import wandb
from matplotlib import pyplot
import cv2

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--example_img_freq', default=20, type=int, help='how often error maps and disparitys are created')
parser.add_argument('--kd_loss_config', help='specifies which losses are used', type=str)

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)


kd_loss_config_strings = [str(item) for item in args.kd_loss_config.split(',')]
kd_loss_config = {'feat_config': kd_loss_config_strings[0],
               'soft_config': kd_loss_config_strings[1],
               'tpw_config':kd_loss_config_strings[2]
               }
# model, optimizer
model = __models__[args.model](args.maxdisp, kd_loss_config)


def test_inf():
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 200
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP

    crop_width = 960
    crop_height = 512
    for i in range(40):
        left = torch.rand(1, 3, crop_height, crop_width).cuda()
        right = torch.rand(1, 3, crop_height, crop_width).cuda()
        print(left.shape, i)
        with torch.no_grad():
            _ = model(left, right)

    # MEASURE PERFORMANCE
    for batch_idx in range(repetitions):
        left = torch.rand(1,3,crop_height,crop_width).cuda()
        right = torch.rand(1,3,crop_height,crop_width).cuda()
        print(left.shape, batch_idx)
        with torch.no_grad():
            starter.record()

            _ = model(left, right)
            ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        if batch_idx == repetitions:
            break
        timings[batch_idx] = curr_time
    mean_syn = np.sum(timings[10:]) / (repetitions - 10)
    std_syn = np.std(timings[10:])
    print({'Sync_inftime_mean': mean_syn,
           'Sync_inftime_std': std_syn})
    print('Sync_inftime_mean: ', mean_syn)
    print('Sync_inftime_std: ', std_syn)
# model.cuda()
# test_inf()

model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

print("Model size:")
print_model_size(model)


wandb.init(project="GwcDist")


def test():

    Loss_list = []
    EPE_list = []
    D1_list = []
    Threshold3_list = []
    inference_time_list = []

    os.makedirs('./predictions', exist_ok=True)
    os.makedirs('./prediction_errors', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):

        start_time = time.time()
        disp_ests, losses, EPEs, D1s, Threshold3s = test_sample(sample)
        inference_time = time.time() - start_time

        print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader),
                                                inference_time))

        losses = tensor2numpy(losses)
        EPEs = tensor2numpy(EPEs)
        D1s = tensor2numpy(D1s)
        Threshold3s = tensor2numpy(Threshold3s)

        Loss_list.append(losses)
        EPE_list.append(EPEs)
        D1_list.append(D1s * 100)
        Threshold3_list.append(Threshold3s * 100)
        inference_time_list.append(inference_time)

        if len(Loss_list) % args.example_img_freq == 0:

            disp_est_tn = disp_ests[-1]
            disp_est_np = tensor2numpy(disp_est_tn)
            error_map = disp_error_image_func(disp_est_tn, sample["disparity"])
            error_map = tensor2numpy(error_map.permute(0, 2, 3, 1))
            top_pad_np = tensor2numpy(sample["top_pad"])
            right_pad_np = tensor2numpy(sample["right_pad"])
            left_filenames = sample["left_filename"]

            for disp_est, top_pad, right_pad, fn, er_disp, EPE, D1, Threshold3 in zip(disp_est_np, top_pad_np,
                                                                                      right_pad_np, left_filenames,
                                                                                      error_map, EPEs, D1s, Threshold3s):
                assert len(disp_est.shape) == 2
                print(disp_est)
                if args.dataset == 'kitti':
                    disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
                else:
                    disp_est = np.array(disp_est, dtype=np.float32)
                name = fn.split('/')
                if args.dataset == 'drivingstereo':
                    fn = os.path.join("predictions/", '_'.join(name[1:]))
                    fnerror = os.path.join("prediction_errors/", '_'.join(name[1:]))
                else:
                    fn = os.path.join("predictions/", '_'.join(name[2:]))
                    fnerror = os.path.join("prediction_errors/", '_'.join(name[2:]))

                print("saving to", fn, disp_est.shape)
                disp_est_uint = np.round(disp_est)
                print(disp_est_uint.shape)
                pyplot.imsave(fn, disp_est_uint, cmap='jet')

                disp_est_uint = cv2.imread(fn)
                disp_est_uint = cv2.putText(disp_est_uint, 'EPE: {:.2f}, D1: {:.2f}'.format
                (np.round(EPE, 2), np.round(D1 * 100, 2)),
                                            (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                cv2.imwrite(fn, disp_est_uint)

                pyplot.imsave(fnerror, er_disp)

                er_disp = cv2.imread(fnerror)
                er_disp = cv2.putText(er_disp, 'EPE: {:.2f}, D1: {:.2f}'.format
                (np.round(EPE, 2), np.round(D1 * 100, 2)),
                                      (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                cv2.imwrite(fnerror, er_disp)

                wandb.log({'loss': losses[0],
                           'EPE': EPEs[0],
                           'D1': D1s[0] * 100,
                           'Threshold3': Threshold3s[0] * 100,
                           'inference time': inference_time,
                           fn: [wandb.Image(fn), wandb.Image(fnerror), wandb.Image(sample["left"])]
                           })
        else:
            wandb.log({'loss': losses[0],
                       'EPE': EPEs[0],
                       'D1': D1s[0] * 100,
                       'Threshold3': Threshold3s[0] * 100,
                       'inference time': inference_time,
                       })


    wandb.log({'avg_loss': np.mean(np.array(Loss_list)),
               'avg_EPE': np.mean(np.array(EPE_list)),
               'avg_D1': np.mean(np.array(D1_list)),
               'avg_Threshold3': np.mean(np.array(Threshold3_list)),
               'avg_inference_time': np.mean(np.array(inference_time_list))
               })

    print('mean Loss:', np.mean(np.array(Loss_list)))
    print('mean EPE:', np.mean(np.array(EPE_list)))
    print('mean D1:', np.mean(np.array(D1_list)))
    print('mean Threshold3:', np.mean(np.array(Threshold3_list)))
    print('avg_inference_time', np.mean(np.array(inference_time_list)))


# test one sample
@make_nograd_func
def test_sample(sample):

    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())

    ### Compute metrics ###
    disp_gt = sample['disparity'].cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    losses = [F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True) for disp_est in disp_ests]
    EPEs = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    D1s = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    Threshold3s = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    return disp_ests, losses, EPEs, D1s, Threshold3s


if __name__ == '__main__':
    test()