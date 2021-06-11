#!/bin/bash

# Note: to be run from inside scripts folder. This script should be called from inside docker_local
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# variables
model_name="masking_multihot"
loss_type="multi_hot"
img_size="632 512"
gpu=7

echo "Training model" $model_name on GPU: $gpu" => started"
python main.py --train \
               --model_name $model_name \
               --loss_type $loss_type \
               --img_size $img_size \
               --gpu_id $gpu
