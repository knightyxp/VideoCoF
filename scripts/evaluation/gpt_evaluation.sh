#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON=${INPUT_JSON:-results/bench_manifest.json}
OUTPUT_JSON=${OUTPUT_JSON:-score/gpt_evaluation.json}
VIDEO_ROOT=${VIDEO_ROOT:-results/videocof_eval}
EDITED_VIDEO_ROOT=${EDITED_VIDEO_ROOT:-$VIDEO_ROOT}
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
API_BASE=${OPENAI_BASE_URL:-https://api.openai.com/v1}
MODEL=${OPENAI_MODEL:-gpt-4o-2024-05-13}
NUM_FRAMES=${NUM_FRAMES:-3}
NUM_WORKERS=${NUM_WORKERS:-4}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
MAX_RETRIES=${MAX_RETRIES:-0}

python metric/gpt_evaluation.py \
  --input_json "$INPUT_JSON" \
  --output_json "$OUTPUT_JSON" \
  --video_root "$VIDEO_ROOT" \
  --edited_video_root "$EDITED_VIDEO_ROOT" \
  --model "$MODEL" \
  --api_base "$API_BASE" \
  --num_frames "$NUM_FRAMES" \
  --num_workers "$NUM_WORKERS" \
  --request_timeout "$REQUEST_TIMEOUT" \
  --max_retries "$MAX_RETRIES" \
  --original_from_compare_left_half \
  --print_stream "$@"
