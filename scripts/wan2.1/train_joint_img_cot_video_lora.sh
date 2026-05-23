#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export MODEL_NAME=${MODEL_NAME:-models/Wan2.1-T2V-14B}
export DATASET_NAME=${DATASET_NAME:-data/VideoCoF-50k}
export DATASET_META_NAME=${DATASET_META_NAME:-data/VideoCoF-50k/train.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OUTPUT_DIR=${OUTPUT_DIR:-experiments/videocof_wan2.1_14b_lora}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-1000}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

if [[ ! -d "$MODEL_NAME" ]]; then
  echo "MODEL_NAME does not exist: $MODEL_NAME" >&2
  exit 1
fi

if [[ ! -d "$DATASET_NAME" ]]; then
  echo "DATASET_NAME does not exist: $DATASET_NAME" >&2
  exit 1
fi

if [[ ! -f "$DATASET_META_NAME" ]]; then
  echo "DATASET_META_NAME does not exist: $DATASET_META_NAME" >&2
  exit 1
fi

train_args=(
  scripts/wan2.1/train_joint_img_video_lora.py
  --config_path="config/wan2.1/wan_civitai.yaml"
  --pretrained_model_name_or_path="$MODEL_NAME"
  --train_data_dir="$DATASET_NAME"
  --train_data_meta="$DATASET_META_NAME"
  --use_cot_dataset
  --enable_gradual_ground
  --rank=128
  --network_alpha=64
  --video_sample_n_frames=66
  --source_frames=33
  --edit_frames=33
  --reasoning_frames=1
  --train_batch_size=1
  --gradient_accumulation_steps=1
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS"
  --num_train_epochs="$NUM_TRAIN_EPOCHS"
  --checkpointing_steps="$CHECKPOINTING_STEPS"
  --learning_rate=1e-04
  --seed=42
  --video_sample_stride=2
  --output_dir="$OUTPUT_DIR"
  --gradient_checkpointing
  --mixed_precision="bf16"
  --adam_weight_decay=3e-2
  --adam_epsilon=1e-10
  --vae_mini_batch=1
  --max_grad_norm=0.05
  --random_hw_adapt
  --enable_bucket
  --uniform_sampling
  --video_edit_loss_on_edited_frames_only
  --use_deepspeed
)

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  train_args+=(--max_train_steps="$MAX_TRAIN_STEPS")
fi

accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/14b_lora_zero2_bf16_config.json \
  --num_processes "$NPROC_PER_NODE" \
  --num_machines 1 \
  --dynamo_backend no \
  --mixed_precision="bf16" \
  "${train_args[@]}"
