#!/bin/bash

# TO BE COMPLETED

# varaibles to be used for all
basename="softmax_baseline_3-1_final"
loss_type="one_hot"
step=5400

data_folder=""
img_size="632 512"
imread_mode=2

val_csv=""
checkpoints_path=""
n_workers=4
gpu=7


run_eval () {
    # uses global vars defied above
    python experiments.py --run evaluate_model \
                          --model_name $model_name \
                          --val_csv $val_csv \
                          --data_folder $data_folder \
                          --imread_mode $imread_mode \
                          --img_size $img_size \
                          --checkpoints_path $checkpoints_path \
                          --loss_type $loss_type \
                          --step $step \
                          --n_workers $n_workers \
                          --gpu_id $gpu
}

# evaluate single model
for i in 1 2 3 4 5
do
    model_name="${basename}_run_${i}"
    run_eval
done
