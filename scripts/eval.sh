#!/bin/bash

# TODO: to be completed
# Note: to be run from inside scripts_local folder. This script should be called from inside docker_local
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# variables to be used for all
basename="masking_multihot"
loss_type="multi_hot"
step=5400
img_size="316 256"
gpu=7


run_eval () {
    # uses global vars defied above
    python experiments.py --run evaluate_model \
                          --model_name $model_name \
                          --img_size $img_size \
                          --loss_type $loss_type \
                          --step $step \
                          --gpu_id $gpu
}

# evaluate single model
for i in 1 2 3 4 5
do
    model_name="${basename}_run_${i}"
    run_eval
done
