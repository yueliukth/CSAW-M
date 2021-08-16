#!/bin/bash

# evaluation of models on the private test set
# Note: to be run from inside scripts folder.
cd ../src || { echo "Could not change directory, aborting..."; exit 1; }

# variables to be used for all
for i in 1 2 3 4 5
do
    # variables to be used for all
    one_or_multi="multi"
    model_name="csaw-m_${one_or_multi}_hot_final_run_${i}"
    loss_type="${one_or_multi}_hot"
    step=5400
    gpu=7

    #test_csv="/raid/sorkhei/CSAW-M-review/labels/CSAW-M_test.csv"
    test_csv='none'
    test_folder="/raid/sorkhei/CSAW-M-review/preprocessed/private"
    line_parse_type=0
    checkpoints_path="/raid/sorkhei/CSAW-M-review_checkpoints"
    save_preds_to="/raid/sorkhei/CSAW-M-review_preds_2/${model_name}_private.csv"

    python main.py --evaluate  \
                   --model_name $model_name \
                   --loss_type $loss_type \
                   --step $step \
                   --gpu_id $gpu \
                   --line_parse_type $line_parse_type \
                   --test_csv $test_csv --test_folder $test_folder --checkpoints_path $checkpoints_path \
                   --only_get_preds \
                   --save_preds_to $save_preds_to
done

