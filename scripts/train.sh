#!/bin/bash

# Note: to be run from inside scripts folder. This script should be called from inside docker_local
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# variables
base_name="masking_multihot"
loss_type="multi_hot"
img_size="316 256"
n_repeats=1  # how many models to train

run_train() {
    echo "Training model" $model_name on GPU: $gpu" => started"
    python main.py --model_name $model_name \
                --loss_type $loss_type \
                --img_size $img_size \
                --gpu_id $gpu
}

for (( i=1; i<=$n_repeats; i++ ))
do
    gpu=$(($i-1))  # specify GPU
    model_name="${base_name}_run_${i}"
    run_train
done
