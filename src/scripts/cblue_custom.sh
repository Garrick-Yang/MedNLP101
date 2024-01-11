DATA_DIR="CBLUEDatasets"                       # 数据集总目录

TASK_NAME="qqr"                                # 具体任务
MODEL_TYPE="bert"                              # 预训练模型类型
MODEL_DIR="data/model_data"                    # 预训练模型保存路径
MODEL_NAME="chinese-bert-wwm"                  # 预训练模型名称
OUTPUT_DIR="data/output"                       # 模型保存目录
RESULT_OUTPUT_DIR="data/result_output"         # 提交结果保存目录

MAX_LENGTH=128

python ../baselines/run_classifier.py \
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
    --epochs=3 \
    --warmup_proportion=0.1 \
    --earlystop_patience=3 \
    --logging_steps=250 \
    --save_steps=250 \
    --seed=2021