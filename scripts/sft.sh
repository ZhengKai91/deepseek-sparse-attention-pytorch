#!/bin/bash
RECIPE_NAME=DSA
TRAINING_LENGTH=32768 
WANDB_NAME=${RECIPE_NAME}_${TRAINING_LENGTH}

export CUDA_VISIBLE_DEVICES=1

torchrun  --nproc_per_node=1 \
        -m training.train  \
        --model_name_or_path "Qwen/Qwen3-4B-Instruct-2507" \
        --dataset_name_or_path Leooyii/Slimpajama_downsample_32k_1B \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --eval_strategy no \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1     \
        --tf32 True \
        --report_to "wandb" \
        --use_wandb False \
        --wandb_name ${WANDB_NAME} 