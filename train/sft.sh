data_path=$1
model_path=$2
output_dir=$3
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

deepspeed --num_gpus=8 train.py \
    --deepspeed ds_configs/zero2.json \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True