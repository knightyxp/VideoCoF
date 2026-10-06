#!/usr/bin/env bash
set -euo pipefail

INPUT_JSON=${INPUT_JSON:?Set INPUT_JSON to your prepared evaluation JSON or benchmark JSONL}
OUTPUT_JSON=${OUTPUT_JSON:-score/perceptual_score.json}
VIDEO_ROOT=${VIDEO_ROOT:-results/videocof_eval}
EDITED_VIDEO_ROOT=${EDITED_VIDEO_ROOT:-$VIDEO_ROOT}
NUM_FRAMES=${NUM_FRAMES:-33}
DINO_MODEL_NAME=${DINO_MODEL_NAME:-dinov2_vits14}
CLIP_MODEL=${CLIP_MODEL:-ViT-B/32}
DINO_REPO=${DINO_REPO:-facebookresearch/dinov2}
EDITED_VIDEO_PATTERN=${EDITED_VIDEO_PATTERN:-'gen_{task_type}_{sample_id}.mp4'}
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXTRA_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--device "$DEVICE")
fi
if [[ -n "${CLIP_DOWNLOAD_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--clip_download_root "$CLIP_DOWNLOAD_ROOT")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/../../metric/compute_clip_score.py" \
  --input_json "$INPUT_JSON" \
  --video_root "$VIDEO_ROOT" \
  --edited_video_root "$EDITED_VIDEO_ROOT" \
  --edited_video_pattern "$EDITED_VIDEO_PATTERN" \
  --num_frames "$NUM_FRAMES" \
  --dino_model_name "$DINO_MODEL_NAME" \
  --dino_repo "$DINO_REPO" \
  --clip_model "$CLIP_MODEL" \
  --output_json "$OUTPUT_JSON" \
  "${EXTRA_ARGS[@]}" "$@"
