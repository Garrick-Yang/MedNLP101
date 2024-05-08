#!/usr/bin/env bash

# 获取当前脚本所在的目录
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换工作路径到脚本所在的目录
cd "$script_dir"

printf "当前工作路径: %s\n" "$script_dir"

DATA_DIR="../../data/CBLUEDatasets"
TASK_NAME="ee"
MODEL_TYPE="bert"
MODEL_DIR="../../data/model_data"
MODEL_NAME="chinese-bert-wwm-ext"
OUTPUT_DIR="../../data/output"
RESULT_OUTPUT_DIR="../../data/result_output"

MAX_LENGTH=128

echo "Start running"

if [ $# == 0 ]; then
    python ../../cblue_baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_train \
        --max_length=${MAX_LENGTH} \
        --train_batch_size=16 \
        --eval_batch_size=16 \
        --learning_rate=3e-5 \
        --epochs=5 \
        --warmup_proportion=0.1 \
        --earlystop_patience=100 \
        --max_grad_norm=0.0 \
        --logging_steps=200 \
        --save_steps=200 \
        --seed=2021
elif [ $1 == "predict" ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_predict \
        --max_length=${MAX_LENGTH} \
        --eval_batch_size=32 \
        --seed=2021
fi