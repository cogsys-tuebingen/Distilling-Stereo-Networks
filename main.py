from __future__ import print_function, division
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import gc
import wandb
from matplotlib import pyplot
import cv2
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from utils.metrics import log_sizes



import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet3D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--vallist', required=True, help='validation list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=4, help='valdation batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=100, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--val_img_freq', type=int, required=True, help='frequency of creating example images')
parser.add_argument('--wandb', type=str, required=True, help='wandb project')
parser.add_argument('--compute_complexity', action='store_true', help='output complexity and model size')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)
os.makedirs('./predictions', exist_ok=True)
os.makedirs('./prediction_errors', exist_ok=True)


# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader

StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
val_dataset = StereoDataset(args.datapath, args.vallist, False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
ValImgLoader = DataLoader(val_dataset, args.val_batch_size, shuffle=False, num_workers=4, drop_last=False)

if args.dataset == "kitti":
    inf_filenames = "./filenames/kitti15_full.txt"
else:
    inf_filenames = "./filenames/sceneflow_small_test_450.txt"

inf_dataset = StereoDataset(args.datapath, inf_filenames, False)
InfImgLoader = DataLoader(inf_dataset, 1, shuffle=False, num_workers=8, drop_last=False)


# model, optimizer
model = __models__[args.model](args.maxdisp)

# wandb log and print sizes
wandb.init(project=args.wandb, entity='lightstereomatching', settings=wandb.Settings(_disable_stats=True, start_method='fork'))

wandb.run.name = args.model

if args.compute_complexity:
    log_sizes(model)

model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    best_checkpoint_EPE = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            start_time = time.time()
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            train_step_time = time.time() - start_time
            scalar_outputs["train_step_time"] = train_step_time
            if do_summary:
                wandb.log({'train_loss': scalar_outputs['loss'],
                           'train_EPE': scalar_outputs['EPE'][-1],
                           'train_D1': scalar_outputs['D1'][-1],
                           'train_Thres1': scalar_outputs['Thres1'][-1],
                           'train_Thres2': scalar_outputs['Thres2'][-1],
                           'train_Thres3': scalar_outputs['Thres3'][-1],
                           'train_step': global_step,
                           'train_step_time': scalar_outputs["train_step_time"]
                           })
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       train_step_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # validation
        avg_val_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            val_run_step = len(ValImgLoader) * batch_idx
            create_img = val_run_step % args.val_img_freq == 0
            loss, scalar_outputs, image_outputs = val_sample(sample, current_epoch=epoch_idx, create_image=create_img)
            wandb.log({'val_loss': scalar_outputs['loss'],
                       'val_EPE': scalar_outputs['EPE'][-1],
                       'val_D1': scalar_outputs['D1'][-1],
                       'val_Thres1': scalar_outputs['Thres1'][-1],
                       'val_Thres2': scalar_outputs['Thres2'][-1],
                       'val_Thres3': scalar_outputs['Thres3'][-1],
                       'val_step': global_step,
                       'val_run_step': val_run_step,
                       'inf_time': scalar_outputs['inf_time']
                       })
            save_scalars(logger, 'val', scalar_outputs, global_step)
            save_images(logger, 'val', image_outputs, global_step)
            avg_val_scalars.update(scalar_outputs)
            print('Epoch {}/{}, Iter {}/{}, val loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(ValImgLoader), loss,
                                                                                    scalar_outputs['inf_time']))
            del scalar_outputs, image_outputs
        avg_val_scalars = avg_val_scalars.mean()
        wandb.log({'avg_val_loss': avg_val_scalars['loss'],
                   'avg_val_EPE': avg_val_scalars['EPE'][0],
                   'avg_val_D1': avg_val_scalars['D1'][0],
                   'avg_val_Thres1': avg_val_scalars['Thres1'][0],
                   'avg_val_Thres2': avg_val_scalars['Thres2'][0],
                   'avg_val_Thres3': avg_val_scalars['Thres3'][0],
                   'avg_inf_time': avg_val_scalars['inf_time'],
                   'epoch': epoch_idx 
                   })
        save_scalars(logger, 'fullval', avg_val_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_val_scalars", avg_val_scalars)

        # saving new best checkpoint
        if avg_val_scalars['EPE'][0] < best_checkpoint_EPE:
            best_checkpoint_EPE = avg_val_scalars['EPE'][0]
            print("Overwriting best checkpoint")
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/best.ckpt".format(args.logdir))

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        test_inf()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            #image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# val one sample
@make_nograd_func
def val_sample(sample, current_epoch, create_image=False):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    scalar_outputs = {}

    start_time = time.time()
    with torch.no_grad():
        disp_ests = model(imgL, imgR)

    scalar_outputs["inf_time"] = time.time() - start_time

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs["loss"] = loss
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if not create_image:
        return tensor2float(loss), tensor2float(scalar_outputs), image_outputs
    else:
        current_epoch = str(current_epoch)
        os.makedirs('./predictions/' + current_epoch, exist_ok=True)
        os.makedirs('./prediction_errors/' + current_epoch, exist_ok=True)

        EPEs = tensor2numpy(scalar_outputs["EPE"])
        D1s = tensor2numpy(scalar_outputs["D1"])
        disp_est_tn = disp_ests[-1]
        disp_est_np = tensor2numpy(disp_est_tn)
        error_map = disp_error_image_func(disp_est_tn, sample["disparity"])
        error_map = tensor2numpy(error_map.permute(0, 2, 3, 1))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        for disp_est, top_pad, right_pad, fn, er_disp, EPE, D1 in zip(disp_est_np, top_pad_np,
                                                                                  right_pad_np, left_filenames,
                                                                                  error_map, EPEs, D1s):
            if args.dataset == 'kitti':
                disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            else:
                disp_est = np.array(disp_est, dtype=np.float32)
            name = fn.split('/')
            if args.dataset == 'drivingstereo':
                fn = os.path.join("predictions/" + current_epoch, '_'.join(name[1:]))
                fnerror = os.path.join("prediction_errors/" + current_epoch, '_'.join(name[1:]))
            else:
                fn = os.path.join("predictions/" + current_epoch, '_'.join(name[2:]))
                fnerror = os.path.join("prediction_errors/" + current_epoch, '_'.join(name[2:]))

            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est)
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
            #print(sample["left"][0].shape)
            images = {fn: [wandb.Image(fn), wandb.Image(fnerror), wandb.Image(sample["left"][0])]}

            wandb.log(images)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def test_inf():
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 200
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    # for batch_idx, sample in enumerate(ValImgLoader):
    #     print(sample['left'].shape)
    #     _ = model(sample['left'].cuda(), sample['right'].cuda())
    #     if batch_idx > 20:
    #         break
    #MEASURE PERFORMANCE
    for batch_idx, sample in enumerate(InfImgLoader):
        sample['left'].cuda()
        sample['right'].cuda()
        with torch.no_grad():
            starter.record()
            start = time.time()
            _ = model(sample['left'], sample['right'])
            ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        if batch_idx == repetitions:
            break
        timings[batch_idx] = curr_time
    mean_syn = np.sum(timings[10:]) / (repetitions - 10)
    std_syn = np.std(timings[10:])
    wandb.log({'Sync_inftime_mean': mean_syn,
               'Sync_inftime_std': std_syn})
    print('Sync_inftime_mean: ', mean_syn)
    print('Sync_inftime_std: ', std_syn)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_inf()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    train()

