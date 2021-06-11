#!/bin/bash

# Note: to be run from inside scripts_local folder. This script should be called from inside docker_local
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# example variables
model_name="masking_multihot"
loss_type="multi_hot"
step=5400
img_size="632 512"
gpu=7
save_preds_to="../results/${model_name}.csv"

python main.py --evaluate \
               --model_name $model_name \
               --loss_type $loss_type \
               --step $step \
               --img_size $img_size \
               --gpu_id $gpu \
               --save_preds_to $save_preds_to
