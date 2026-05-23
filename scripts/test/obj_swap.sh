export CUDA_VISIBLE_DEVICES=0

# sample_id: 001
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/sign.mp4 \
  --prompt "Replace the yellow \"SCHOOL\" sign with a red hospital sign, featuring a white hospital emblem on the top and the word \"HOSPITAL\" below." \
  --output_dir results/obj_swap_1 \
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
  --video_path assets/old_man.mp4 \
  --prompt "Swap the old man with long white hair and a blue checkered shirt at the left side of the frame with a woman with curly brown hair and a denim shirt." \
  --output_dir results/obj_swap_2 \
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
  --video_path assets/pants.mp4 \
  --prompt "swap the white pants worn by the individual the light blue jeans." \
  --output_dir results/obj_swap_3 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope
