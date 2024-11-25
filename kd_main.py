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
from models import __models__
from utils import __loss_types__
from utils import *
from utils.kd_loss_utils import *
from utils.metrics import *

# leak investigation
# import tracemalloc
# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet3D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--kittitest_datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--vallist', required=True, help='valing list')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=4, help='valing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--feat_epochs', type=str, help='the epochs to decay feature weight: the downscale rate')
parser.add_argument('--soft_epochs', type=str, help='the epochs to decay soft weight: the downscale rate')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', required=False, help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=100, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--kittitest_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--val_img_freq', type=int, required=True, help='frequency of creating example images')
parser.add_argument('--wandb', type=str, required=True, help='wandb project')

# kd args
parser.add_argument('--kd_loss_config', help='specifies which losses are used', type=str)
parser.add_argument('--loss_weights', type=str)
parser.add_argument('--soft_weights', type=str)
parser.add_argument('--feat_weights', type=str)
parser.add_argument('--gt_pw_loss', type=str, required=True, help='type of criterion for groundtruth pixel-wise loss')
parser.add_argument('--teacher_pw_loss', type=str, required=True, help='type of criterion for teacher pixel-wise distillation loss')
parser.add_argument('--soft_loss', type=str, help='type of criterion for teacher soft distillation loss')
parser.add_argument('--feat_loss', type=str, help='type of criterion for teacher feature distillation loss')
parser.add_argument('--teacher', default='gwcnet-g', help='select a teacher model structure', choices=__models__.keys())
parser.add_argument('--teacherckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--compute_complexity', action='store_true', help='output complexity and model size')
parser.add_argument('--kitti12_testing', action='store_true', help='output complexity and model size')

parser.add_argument('--tea_pw_loss', action='store_true', help='also use teacher pixelwise loss')
parser.add_argument('--pw_lambda', type=float, default=0, help='base pixelwise teacher loss weight')


# parse arguments, set seeds
args = parser.parse_args()
print(args)
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

if args.kitti12_testing:
    StereoDataset = __datasets__['kitti']
    kitti12test_list = './filenames/kitti12_full.txt'
    kitti12_dataset = StereoDataset(args.kittitest_datapath, kitti12test_list, False)
    KITTI12ImgLoader = DataLoader(kitti12_dataset, args.val_batch_size, shuffle=False, num_workers=4, drop_last=False)

wandb.init(project=args.wandb, entity='lightstereomatching', config=args, settings=wandb.Settings(_disable_stats=True, start_method='fork'))

# model, optimizer
# if args.teacher == "gwcnet-lac" or args.loss_type == "gwc_vol_x4":
#     transform_feats = True
# else:
#     transform_feats = False


# specify args
kd_loss_config_strings = [str(item) for item in args.kd_loss_config.split(',')]
kd_loss_config = {'feat_config': kd_loss_config_strings[0],
               'soft_config': kd_loss_config_strings[1],
               'tpw_config':kd_loss_config_strings[2]
               }
loss_weights = [float(item) for item in args.loss_weights.split(',')]
feat_weights= [float(item) for item in args.feat_weights.split(',')]
soft_weights= [float(item) for item in args.soft_weights.split(',')]

feat_downscale_epochs, feat_downscale_rate = split_weight_epochs(args.feat_epochs)
soft_downscale_epochs, soft_downscale_rate = split_weight_epochs(args.soft_epochs)


model = __models__[args.model](args.maxdisp)


# wandb log and print sizes of student
if args.compute_complexity:
    log_sizes(model)

model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
print(model)
# load teacher
if args.teacher == "gwcnet-lac":
    raise NotImplementedError
    # affinity_settings = {}
    # affinity_settings['win_w'] = 3
    # affinity_settings['win_h'] = 3
    # affinity_settings['dilation'] = [1, 2, 4, 8]
    #
    # teacher = __models__[args.teacher](args.maxdisp, args.loss_type, struct_fea_c=4, fuse_mode='separate', affinity_settings=affinity_settings,
    #                                    udc=True, refine='csr')
    #
    # teacher = nn.DataParallel(teacher)
    #
    # # load teacher parameters
    # print("loading model {}".format(args.teacherckpt))
    # ckpt = torch.load(args.teacherckpt)
    # teacher.load_state_dict(ckpt)

elif args.teacher == "acvnet":
    teacher = __models__[args.teacher](args.maxdisp, kd_loss_config)
    teacher = nn.DataParallel(teacher)

    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.teacherckpt))
    state_dict = torch.load(args.teacherckpt)
    model_dict = teacher.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    teacher.load_state_dict(model_dict)

else:
    raise NotImplementedError
    # teacher = __models__[args.teacher](args.maxdisp, args.loss_type)
    # teacher = nn.DataParallel(teacher)
    #
    # # load teacher parameters
    # print("loading model {}".format(args.teacherckpt))
    # state_dict = torch.load(args.teacherckpt)
    # teacher.load_state_dict(state_dict['model'])

teacher.cuda()
teacher.eval()

# load parameters
start_epoch = 0
if args.resume:
    #print("XXXXXXXXXXXXXXXXXXX")
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
    best_checkpoint_D1 = 100
    scalar_output_list = []

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        #adjust feat weights
        if epoch_idx in feat_downscale_epochs:
            loss_weights[1] /= feat_downscale_rate
            print("feature weight set to ", loss_weights[1])

        #adjust soft weights
        if epoch_idx in soft_downscale_epochs:
            loss_weights[2] /= soft_downscale_rate
            print("soft weight set to ", loss_weights[2])

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            start_time = time.time()
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            train_step_time = time.time() - start_time
            scalar_outputs["train_step_time"] = train_step_time
            if do_summary:
                wandb.log({'train_vanilla_loss': scalar_outputs['vanilla_loss'],
                           'train_feat_loss': scalar_outputs['feat_loss'],
                           'train_soft_loss': scalar_outputs['soft_loss'],
                           'train_pw_loss': scalar_outputs['pw_loss'],
                           'train_total_loss': scalar_outputs['total_loss'],
                           'train_EPE': scalar_outputs['EPE'][-1],
                           'train_D1': scalar_outputs['D1'][-1],
                           'train_Thres1': scalar_outputs['Thres1'][-1],
                           'train_Thres2': scalar_outputs['Thres2'][-1],
                           'train_Thres3': scalar_outputs['Thres3'][-1],
                           'train_step': global_step,
                           'train_step_time': scalar_outputs["train_step_time"]
                           })
                # save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)

            scalar_output_list.append(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       train_step_time))

        # log epoch averages

        avg_scalar_outputs = dict_mean(scalar_output_list, {'vanilla_loss', 'feat_loss', 'soft_loss', 'pw_loss', 'total_loss', 'train_step_time'})


        # losses = {'l1_loss', 'kd_loss', 'total_loss'}
        # train_losses_dict = {your_key: avg_scalar_outputs[your_key] for your_key in losses}


        wandb.log({'avg_train_vanilla_loss': avg_scalar_outputs['vanilla_loss'],
                    'avg_train_feat_loss': avg_scalar_outputs['feat_loss'],
                    'avg_train_soft_loss': avg_scalar_outputs['soft_loss'],
                    'avg_train_pw_loss': avg_scalar_outputs['pw_loss'],
                    'avg_train_total_loss': avg_scalar_outputs['total_loss'],
                    'avg_train_time': avg_scalar_outputs['train_step_time']
                   # 'avg_train_EPE': avg_scalar_outputs['EPE'][-1],
                   # 'avg_train_D1': avg_scalar_outputs['D1'][-1],
                   # 'avg_train_Thres1': avg_scalar_outputs['Thres1'][-1],
                   # 'avg_train_Thres2': avg_scalar_outputs['Thres2'][-1],
                   # 'avg_train_Thres3': avg_scalar_outputs['Thres3'][-1],
                   })

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # validtion
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
            # save_scalars(logger, 'val', scalar_outputs, global_step)
            # save_images(logger, 'val', image_outputs, global_step)
            avg_val_scalars.update(scalar_outputs)
            #print(avg_val_scalars)
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
        #save_scalars(logger, 'fullval', avg_val_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_val_scalars", avg_val_scalars)


        if args.kitti12_testing and ((epoch_idx + 1) % args.kittitest_freq == 0):
            # running kitti 0-shot test
            avg_KITTI12_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(KITTI12ImgLoader):
                global_step = len(KITTI12ImgLoader) * epoch_idx + batch_idx
                val_run_step = len(KITTI12ImgLoader) * batch_idx
                create_img = val_run_step % 30 == 0
                loss, scalar_outputs, image_outputs = val_sample(sample, current_epoch=epoch_idx,
                                                                 create_image=create_img)
                wandb.log({'KITTI12_loss': scalar_outputs['loss'],
                           'KITTI12_EPE': scalar_outputs['EPE'][-1],
                           'KITTI12_D1': scalar_outputs['D1'][-1],
                           'KITTI12_Thres1': scalar_outputs['Thres1'][-1],
                           'KITTI12_Thres2': scalar_outputs['Thres2'][-1],
                           'KITTI12_Thres3': scalar_outputs['Thres3'][-1],
                           'KITTI12_step': global_step,
                           'KITTI12_run_step': val_run_step,
                           'KITTI12_inf_time': scalar_outputs['inf_time']
                           })
                # save_scalars(logger, 'KITTI12', scalar_outputs, global_step)
                # save_images(logger, 'KITTI12', image_outputs, global_step)
                avg_KITTI12_scalars.update(scalar_outputs)
                # print(avg_val_scalars)
                print('Epoch {}/{}, Iter {}/{}, KITTI12 loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                        batch_idx,
                                                                                        len(KITTI12ImgLoader), loss,
                                                                                        scalar_outputs['inf_time']))
                del scalar_outputs, image_outputs
            avg_KITTI12_scalars = avg_KITTI12_scalars.mean()
            wandb.log({'avg_KITTI12_loss': avg_KITTI12_scalars['loss'],
                       'avg_KITTI12_EPE': avg_KITTI12_scalars['EPE'][0],
                       'avg_KITTI12_D1': avg_KITTI12_scalars['D1'][0],
                       'avg_KITTI12_Thres1': avg_KITTI12_scalars['Thres1'][0],
                       'avg_KITTI12_Thres2': avg_KITTI12_scalars['Thres2'][0],
                       'avg_KITTI12_Thres3': avg_KITTI12_scalars['Thres3'][0],
                       'avg_KITTI12_time': avg_KITTI12_scalars['inf_time'],
                       'epoch': epoch_idx
                       })
            # save_scalars(logger, 'fullKITTI12', avg_KITTI12_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_KITTI12_scalars", avg_KITTI12_scalars)
            del avg_KITTI12_scalars



        # saving new best checkpoint wrt EPE
        if avg_val_scalars['EPE'][0] < best_checkpoint_EPE:
            best_checkpoint_EPE = avg_val_scalars['EPE'][0]
            wandb.log({'best_ckpt_EPE': best_checkpoint_EPE})
            print("Overwriting best EPE checkpoint")
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}bestEPE.ckpt".format(args.logdir))


        if avg_val_scalars['D1'][0] < best_checkpoint_D1:
            best_checkpoint_D1 = avg_val_scalars['D1'][0]
            wandb.log({'best_ckpt_D1': best_checkpoint_D1})
            print("Overwriting best D1 checkpoint")
            # saves best D1 checkpoint if it isnt the best EPE checkpoint also
            if best_checkpoint_EPE != avg_val_scalars['EPE'][0]:
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}bestD1.ckpt".format(args.logdir))

        del avg_scalar_outputs
        del avg_val_scalars

        gc.collect()

        #tracker.print_diff()
        #print(mem_top())



# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()


    pw_loss = 0.
    soft_loss = 0.
    feat_loss = 0.

    optimizer.zero_grad()

    disp_ests, stud_feat_out, stud_soft_out = model(imgL, imgR, being_taught=True)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    vanilla_loss = model_loss(disp_ests, disp_gt, mask, __loss_types__[args.gt_pw_loss])

    if not all(value == 'none' for value in kd_loss_config.values()):
        with torch.no_grad():
            tea_disp_ests, tea_feat_out, tea_soft_out = teacher(imgL, imgR, teaching=True)

    # loss config [Use teacher pixelwise, if > 0, Use feat loss (and which features to return), if > 0, Use soft loss (and which many predictions to return)]


    # feature distillation loss
    if kd_loss_config['feat_config'] != 'none':
        feat_loss = kd_model_loss(stud_feat_out, tea_feat_out, feat_weights, __loss_types__[args.feat_loss])

    # soft distillation loss
    if kd_loss_config['soft_config'] != 'none':
        soft_loss = kd_model_loss(stud_soft_out, tea_soft_out, soft_weights, __loss_types__[args.soft_loss])

    # teacher pixel-wise loss
    if 'default' in kd_loss_config['tpw_config']:
        stu_finaL_preds = disp_ests[-1]
        tea_final_preds = tea_disp_ests[-1]
        pw_loss = __loss_types__[args.teacher_pw_loss](stu_finaL_preds[mask], tea_final_preds[mask])

    vanilla_loss = loss_weights[0] * vanilla_loss
    feat_loss = loss_weights[1] * feat_loss
    soft_loss = loss_weights[2] * soft_loss
    pw_loss = loss_weights[3] * pw_loss

    total_loss = vanilla_loss + feat_loss + soft_loss + pw_loss

    scalar_outputs = {"vanilla_loss": vanilla_loss,
              "pw_loss" : pw_loss,
              "feat_loss": feat_loss,
              "soft_loss": soft_loss,
              "total_loss": total_loss}

    print(scalar_outputs)

    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            #image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    total_loss.backward()
    optimizer.step()

    return tensor2float(total_loss), tensor2float(scalar_outputs), image_outputs


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
    loss = model_loss(disp_ests, disp_gt, mask, __loss_types__[args.gt_pw_loss])

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


if __name__ == '__main__':
    train()
