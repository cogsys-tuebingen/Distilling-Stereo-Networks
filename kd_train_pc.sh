#!/usr/bin/env bash
set -x
DATAPATH='/data/GWC_data/SceneFlow/'
DATAPATH2='/data/GWC_data/KITTI/'

python kd_main.py --model gwcnet-xsfe-2hg-slied16 \
               --datapath $DATAPATH \
               --kittitest_datapath $DATAPATH2 \
               --dataset sceneflow \
               --trainlist ./filenames/sceneflow_compile_test.txt \
               --vallist ./filenames/sceneflow_compile_test.txt \
               --epochs 10 \
               --lrepochs "1:10" \
               --batch_size 2 \
               --val_batch_size 2 \
               --logdir './logdir/' \
               --wandb "pc testing" \
               --val_img_freq 1 \
               --teacher acvnet \
               --teacherckpt '../acvn_sf2.ckpt' \
               --kd_loss_config 'gwc_vol + early_feats,cost1 + cost2,default' \
               --loss_weights '1,1,0.2,0.1' \
               --feat_weights '1, 1, 1, 1, 5' \
               --soft_weights '0.5, 1' \
               --gt_pw_loss 'LogL1_v2' \
               --teacher_pw_loss 'SmoothL1' \
               --soft_loss 'soft_kl' \
               --feat_loss 'StereoSimilarity' \
               --feat_epochs '1,2,4:10' \
               --soft_epochs '1,2,4:10' \
               --summary_freq 1 \
#               --kitti12_testing \
#               --kittitest_freq 1
               #--loadckpt './gwc_sf.ckpt'
#'../gwclac_sf.pth' \

               #--loadckpt '../xsfe_2hg16_sf.ckpt' \
