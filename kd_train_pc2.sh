#!/usr/bin/env bash
set -x
DATAPATH='/data/GWC_data/SceneFlow/'
DATAPATH2='/data/GWC_data/KITTI/'

python kd_main.py --model gwcnet-xsfe-2hg-slied16 \
               --datapath $DATAPATH \
               --kittitest_datapath $DATAPATH2 \
               --dataset sceneflow \
               --trainlist ./filenames/sceneflow_test.txt \
               --vallist ./filenames/sceneflow_compile_test.txt \
               --epochs 5 \
               --lrepochs "1:10" \
               --batch_size 2 \
               --val_batch_size 2 \
               --logdir './logdir/' \
               --wandb "pc testing" \
               --val_img_freq 1 \
               --teacher acvnet \
               --teacherckpt '../acvn_sf2.ckpt' \
               --kd_loss_config 'none,none,none' \
               --loss_weights '1,1,1,1' \
               --feat_weights '1,1,1,1,5' \
               --soft_weights '1' \
               --gt_pw_loss 'SmoothL1' \
               --teacher_pw_loss 'LogL1' \
               --soft_loss 'soft_kl' \
               --feat_loss 'hint' \
               --summary_freq 1 \
               --kitti12_testing \
               --kittitest_freq 1
               #--loadckpt './gwc_sf.ckpt'
#'../gwclac_sf.pth' \

               #--loadckpt '../xsfe_2hg16_sf.ckpt' \
