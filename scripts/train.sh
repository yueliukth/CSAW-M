#!/bin/bash

# TO BE COMPLETED

# variables
pretraining="image_net"
base_name="softmax_baseline_3-1_final"
loss_type="one_hot"
augments="h_flip v_flip rot_10 color_jitter"

data_folder=""
img_size="632 512"
imread_mode=2

train_csv=""
checkpoints_path=""
line_parse_type=1
csv_sep_type=1
b_size=64
lr=1e-6
n_workers=4
n_epochs=50
eval_step=50  # only used for saving checkpoints if no val_csv is provided

# gpu=7
n_repeats=5  # how many runs on training we want to have

run_train() {
    echo "Training model" $model_name on GPU: $gpu" => started"
    python main.py --model_name $model_name \
                --loss_type $loss_type \
                --pretraining $pretraining \
                --train_csv $train_csv \
                --data_folder $data_folder \
                --imread_mode $imread_mode \
                --img_size $img_size \
                --augments $augments \
                --line_parse_type $line_parse_type \
                --csv_sep_type $csv_sep_type \
                --b_size $b_size \
                --lr $lr \
                --n_workers $n_workers \
                --checkpoints_path $checkpoints_path \
                --n_epochs $n_epochs \
                --eval_step $eval_step \
                --gpu_id $gpu \
                &> $out_file &  # note: this is non-blocking
}

if (( $n_repeats == 1 )); then
    model_name="${base_name}"
    out_file="../logs/${model_name}.txt"
    run_train
else
    for (( i=1; i<=$n_repeats; i++ ))
    do
        gpu=$(($i-1))  # note: this is non-blocking
        model_name="${base_name}_run_${i}"
        out_file="../logs/${model_name}.txt"
        run_train
    done
fi

