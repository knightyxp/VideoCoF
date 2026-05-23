#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

MODEL_NAME=${MODEL_NAME:-Wan2.1-T2V-14B}
TEST_JSON=${TEST_JSON:-data/test_json/20_test.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/videocof_cot_parallel_test}
LORA_PATH=${LORA_PATH:-videocof_weight/videocof.safetensors}
ACCELERATION_LORA_PATH=${ACCELERATION_LORA_PATH:-videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors}

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NUM_FRAMES=${NUM_FRAMES:-33}
SOURCE_FRAMES=${SOURCE_FRAMES:-33}
REASONING_FRAMES=${REASONING_FRAMES:-4}
SEED=${SEED:-0}
USE_DMD=${USE_DMD:-1}
ENABLE_ACCELERATION_LORA=${ENABLE_ACCELERATION_LORA:-1}

if [[ "$USE_DMD" == "1" ]]; then
  PREDICT_SCRIPT=examples/wan2.1/predict_v2v_dmd_cot_json.py
else
  PREDICT_SCRIPT=examples/wan2.1/predict_v2v_cot_json.py
fi

cmd=(
  torchrun
  --nproc_per_node="$NPROC_PER_NODE"
  "$PREDICT_SCRIPT"
  --model_name "$MODEL_NAME"
  --test_json "$TEST_JSON"
  --output_dir "$OUTPUT_DIR"
  --seed "$SEED"
  --num_frames "$NUM_FRAMES"
  --source_frames "$SOURCE_FRAMES"
  --reasoning_frames "$REASONING_FRAMES"
  --repeat_rope
)

if [[ -n "$LORA_PATH" ]]; then
  cmd+=(--lora_path "$LORA_PATH")
fi

if [[ "$USE_DMD" == "1" && "$ENABLE_ACCELERATION_LORA" == "1" ]]; then
  cmd+=(--enable_acceleration_lora)
  if [[ -n "$ACCELERATION_LORA_PATH" ]]; then
    cmd+=(--acceleration_lora_path "$ACCELERATION_LORA_PATH")
  fi
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
