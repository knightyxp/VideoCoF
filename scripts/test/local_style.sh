export CUDA_VISIBLE_DEVICES=0

# sample_id: 001
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/bowl.mp4 \
  --prompt "Make the largest cup on the right white and smooth." \
  --output_dir results/local_style_1 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope

# sample_id: 002
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/ketchup.mp4 \
  --prompt "Make the ketchup bottle to the right of the BBQ sauce bottle violet color." \
  --output_dir results/local_style_2 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope

# sample_id: 003
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/fruit.mp4 \
  --prompt "Make the pomegranate at the right side of the basket lavender color." \
  --output_dir results/local_style_3 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope
