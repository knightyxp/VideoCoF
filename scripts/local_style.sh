export CUDA_VISIBLE_DEVICES=2

torchrun --nproc_per_node=1 inference.py \
  --video_path assets/dough.mp4 \
  --prompt "Make the dough on the cutting board crusty and golden as if freshly baked." \
  --output_dir results/local_style \
  --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
  --seed 0 \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope \
  --videocof_path videocof_weight/videocof.safetensors


