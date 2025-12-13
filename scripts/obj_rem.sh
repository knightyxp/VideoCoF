export CUDA_VISIBLE_DEVICES=0

# sample_id: 001
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/two_man.mp4 \
  --prompt "Remove the young man with short black hair wearing black shirt on the left." \
  --output_dir results/obj_rem_1 \
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
  --video_path assets/three_people.mp4 \
  --prompt "Remove the man with short dark hair wearing a gray suit on the right" \
  --output_dir results/obj_rem_2 \
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
  --video_path assets/office.mp4 \
  --prompt "Remove the beige CRT computer setup." \
  --output_dir results/obj_rem_3 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope
