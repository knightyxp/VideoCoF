#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON=${INPUT_JSON:-data/test_json/4tasks_rem_add_swap_local-style_test.json}
OUTPUT_JSON=${OUTPUT_JSON:-score/gpt_evaluation.json}
VIDEO_ROOT=${VIDEO_ROOT:-results/videocof_eval}
EDITED_VIDEO_ROOT=${EDITED_VIDEO_ROOT:-$VIDEO_ROOT}
API_KEY=${OPENAI_API_KEY:?Set OPENAI_API_KEY}
API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
MODEL=${MODEL:-gpt-4o}
NUM_FRAMES=${NUM_FRAMES:-3}
NUM_WORKERS=${NUM_WORKERS:-4}

python metric/gpt_evaluation.py \
  --input_json "$INPUT_JSON" \
  --output_json "$OUTPUT_JSON" \
  --video_root "$VIDEO_ROOT" \
  --edited_video_root "$EDITED_VIDEO_ROOT" \
  --api_key "$API_KEY" \
  --model "$MODEL" \
  --api_base "$API_BASE" \
  --num_frames "$NUM_FRAMES" \
  --num_workers "$NUM_WORKERS" \
  --original_from_compare_left_half \
  --print_stream
