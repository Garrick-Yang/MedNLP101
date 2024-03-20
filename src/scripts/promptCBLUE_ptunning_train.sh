PRE_SEQ_LEN=128
LR=2e-2
your_data_path="./data/PromptCBLUE_Datasets/toy_examples/"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./output/PromptCBLUE"  # 填入用来存储模型的路径
model_name_or_path="/mnt/nvme_share/yangjz/Python/LLMs/chatglm-6b"    # LLM底座模型路径，或者是huggingface hub上的模型名称


CUDA_VISIBLE_DEVICES=0 python src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \

#CUDA_VISIBLE_DEVICES=0: 测试合并。
#--do_train: 表示进行训练。
#--train_file $your_data_path/train.json: 指定训练数据集的路径，你需要替换$your_data_path为你的数据文件所在的路径。
#--validation_file $your_data_path/dev.json: 指定验证数据集的路径，同样需要替换$your_data_path。
#--prompt_column input: 指定数据中用作模型输入的列名。
#--response_column target: 指定数据中用作模型输出（目标回答）的列名。
#--overwrite_cache: 如果设置了此选项，将覆盖处理后的数据缓存。
#--model_name_or_path $model_name_or_path: 指定预训练模型的名称或路径，需要替换$model_name_or_path为实际的模型名称或路径。
#--output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR: 指定输出目录的路径，包括训练过程中保存的模型和日志，需要替换$your_checkpopint_path、$PRE_SEQ_LEN和$LR为实际的值。
#--overwrite_output_dir: 如果输出目录已经存在，将其覆盖。
#--max_source_length 700: 设置输入序列的最大长度为700个token。
#--max_target_length 196: 设置目标（回答）序列的最大长度为196个token。
#--per_device_train_batch_size 1: 设置每个GPU上的训练批量大小为1。
#--per_device_eval_batch_size 1: 设置每个GPU上的评估批量大小为1。
#--gradient_accumulation_steps 2: 梯度累积的步数，有助于增加有效的批量大小，减少内存消耗。
#--max_steps 1000: 设置训练的最大步数为1000步。
#--logging_steps 10: 每训练10步记录一次日志。
#--save_steps 10: 每训练10步保存一次模型。
#--learning_rate $LR: 设置学习率，需要替换$LR为具体的学习率值。
#--pre_seq_len $PRE_SEQ_LEN: 设置预序列长度，即输入给模型的提示信息的长度，需要替换$PRE_SEQ_LEN为具体的值。