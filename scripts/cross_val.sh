#!/bin/bash

# Note: to be run from inside scripts folder. This script should be called from inside docker_local
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# variables
base_name="masking_onehot"
loss_type="one_hot"
img_size="316 256"
gpu=7

for i in 1 2 3 4 5
do
    model_name="${base_name}_cv_${i}"
#    gpu=$(($i-1))
    echo "Training model" $model_name on GPU: $gpu" => started"

    python main.py --train \
                   --model_name $model_name \
                   --loss_type $loss_type \
                   --cv $i \
                   --img_size $img_size \
                   --gpu_id $gpu
done
