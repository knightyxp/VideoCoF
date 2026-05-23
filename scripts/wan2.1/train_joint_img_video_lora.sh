#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-1.3B}
export DATASET_NAME=${DATASET_NAME:-}
export DATASET_META_NAME=${DATASET_META_NAME:-data/json/video_joint_edit_4tasks.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OUTPUT_DIR=${OUTPUT_DIR:-experiments/wan2.1_1.3b_joint_img_video_lora}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-500}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}

accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/1.3b_lora_zero_stage2_1node.json \
  --num_processes "$NPROC_PER_NODE" \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision bf16 \
  scripts/wan2.1/train_joint_img_video_lora.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$DATASET_NAME" \
  --train_data_meta "$DATASET_META_NAME" \
  --use_image_video_edit_dataset \
  --rank 128 \
  --video_sample_n_frames 66 \
  --source_frames 33 \
  --edit_frames 33 \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --checkpointing_steps "$CHECKPOINTING_STEPS" \
  --learning_rate 1e-4 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --adam_weight_decay 3e-2 \
  --adam_epsilon 1e-10 \
  --vae_mini_batch "$TRAIN_BATCH_SIZE" \
  --max_grad_norm 0.05 \
  --random_hw_adapt \
  --enable_bucket \
  --uniform_sampling \
  --video_edit_loss_on_edited_frames_only \
  --use_deepspeed
