#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON=${INPUT_JSON:-data/test_json/4tasks_rem_add_swap_local-style_test.json}
OUTPUT_JSON=${OUTPUT_JSON:-score/perceptual_score.json}
VIDEO_ROOT=${VIDEO_ROOT:-results/videocof_eval}
EDITED_VIDEO_ROOT=${EDITED_VIDEO_ROOT:-$VIDEO_ROOT}
NUM_FRAMES=${NUM_FRAMES:-33}
DINO_MODEL_NAME=${DINO_MODEL_NAME:-dinov2_vits14}

python metric/compute_clip_score.py \
  --input_json "$INPUT_JSON" \
  --video_root "$VIDEO_ROOT" \
  --edited_video_root "$EDITED_VIDEO_ROOT" \
  --num_frames "$NUM_FRAMES" \
  --dino_model_name "$DINO_MODEL_NAME" \
  --output_json "$OUTPUT_JSON"
