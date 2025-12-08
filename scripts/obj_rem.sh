export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 inference.py \
  --video_path assets/two_man.mp4 \
  --prompt "Remove the young man with short black hair wearing black shirt on the left." \
  --output_dir results/obj_rem \
  --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
  --seed 0 \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope \
  --lora_path videocof_weight/videocof.safetensors