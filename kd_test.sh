#!/usr/bin/env bash
set -e  # Exit on any error
set -x  # Print each command

# === Paths ===
DATAPATH="/data/rahim/data/Sceneflow/test"
TESTLIST="./filenames/sceneflow_test.txt"
CKPT="./checkpoints/bestEPE.ckpt"

# === Run the test ===
python test.py \
    --model gwcnet-xsfe-2hg-slied16 \
    --dataset sceneflow \
    --datapath "$DATAPATH" \
    --testlist "$TESTLIST" \
    --loadckpt "$CKPT" \
    --kd_loss_config gwc_vol+early_feats,cost2,default \
    --maxdisp 192

