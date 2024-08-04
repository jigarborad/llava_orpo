#!/bin/bash
# The following parameter were found to have the best performance on llava bench out of differen beta, aver  average and learning rate
# Model Name: llava-lora-orpo-beta-0.1-lr-5e-5
# Beta: 0.1
# Learning Rate: 5e-5
DATA_PATH=playground/data/RLAIF-V-Dataset/RLAIF-V-Dataset_data.json
IMAGE_FOLDER=playground/data/RLAIF-V-Dataset/ # TODO: Mkae sure to change this to the correct path
run_name=llava-v1.6-mistral-7b-orpo-lora
ouput_dir=./checkpoints/${run_name}
# Notice that I am loading the latest model checkopint 
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
    --version v1 \
    --task ORPO \
    --orpo_beta 0.1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${ouput_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5\
    --is_multimodal True \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${run_name} \