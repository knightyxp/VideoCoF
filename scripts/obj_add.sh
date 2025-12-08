export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 inference.py \
  --video_path assets/woman_ballon.mp4 \
  --prompt "Add the woman in a floral dress pointing at the balloon on the left." \
  --output_dir results/obj_add \
  --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
  --seed 0 \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope \
  --videocof_path videocof_weight/videocof.safetensors