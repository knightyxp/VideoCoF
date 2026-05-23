#!/bin/bash -l
#SBATCH --job-name=videocof_cot_14b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=./slurmlog/slurm-%j.out
#SBATCH --error=./slurmlog/slurm-%j.err

set -euo pipefail

# Edit these defaults or override them in the submission environment.
export CONDA_ROOT=${CONDA_ROOT:-/path/to/conda}
export CONDA_ENV=${CONDA_ENV:-videocof}
export REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}
export MODEL_NAME=${MODEL_NAME:-/path/to/Wan2.1-T2V-14B}
export DATASET_NAME=${DATASET_NAME:-}
export DATASET_META_NAME=${DATASET_META_NAME:-data/VideoCoF-50k/train.json}
export OUTPUT_DIR=${OUTPUT_DIR:-experiments/videocof_14b_cot_lora_rank256_alpha128}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.triton_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export PATH=$CONDA_ROOT/bin:$PATH
module load cuda/12.8.1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES"
echo "GPUS_PER_NODE=$SLURM_GPUS_PER_TASK"
echo

cd "$REPO_DIR"
mkdir -p slurmlog

srun bash -lc "
  export PATH=$CONDA_ROOT/bin:\$PATH
  source $CONDA_ROOT/etc/profile.d/conda.sh
  conda activate $CONDA_ENV || { echo 'Worker activate failed'; exit 1; }

  echo 'WORKER \$SLURM_PROCID on ' \$(hostname)
  echo 'which python:' \$(which python)
  echo 'python -V:' \$(python -V)
  echo

  torchrun \
    --nnodes \$SLURM_JOB_NUM_NODES \
    --nproc_per_node \$SLURM_GPUS_PER_TASK \
    --rdzv_backend c10d \
    --rdzv_endpoint \$MASTER_ADDR:\$MASTER_PORT \
    --rdzv_id \$SLURM_JOB_ID \
    scripts/wan2.1/train_joint_img_video_lora.py \
        --config_path config/wan2.1/wan_civitai.yaml \
        --pretrained_model_name_or_path \"\$MODEL_NAME\" \
        --train_data_dir \"\$DATASET_NAME\" \
        --train_data_meta \"\$DATASET_META_NAME\" \
        --use_cot_dataset \
        --enable_gradual_ground \
        --rank 256 \
        --network_alpha 128 \
        --video_sample_n_frames 66 \
        --source_frames 33 \
        --edit_frames 33 \
        --reasoning_frames 4 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 2 \
        --num_train_epochs 2 \
        --checkpointing_steps 1000 \
        --learning_rate 1e-04 \
        --seed 42 \
        --video_sample_stride 2 \
        --output_dir \"\$OUTPUT_DIR\" \
        --gradient_checkpointing \
        --mixed_precision bf16 \
        --adam_weight_decay 3e-2 \
        --adam_epsilon 1e-10 \
        --vae_mini_batch 1 \
        --max_grad_norm 0.05 \
        --random_hw_adapt \
        --enable_bucket \
        --uniform_sampling \
        --video_edit_loss_on_edited_frames_only \
        --use_deepspeed \
        --deepspeed_config config/14b_lora_zero2_bf16_config.json
"

echo "JOB COMPLETED"
