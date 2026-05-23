export CUDA_VISIBLE_DEVICES=0

# sample_id: 001
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/woman_ballon.mp4 \
  --prompt "Add the woman in a floral dress pointing at the balloon on the left." \
  --output_dir results/obj_add_1 \
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
  --video_path assets/greenhouse.mp4 \
  --prompt "A white Samoyed is watching the man, who crouches in a greenhouse. The Samoyed is covered in thick, fluffy white fur, giving it a very soft and plush appearance. Its ears are erect and triangular, making it look alert and intelligent. The Samoyed's face features its signature smile, with bright black eyes that convey friendliness and curiosity." \
  --output_dir results/obj_add_2 \
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
  --video_path assets/gameplay.mp4 \
  --prompt "Add the woman holding the blue game controller to the left of the man, engaged in gameplay." \
  --output_dir results/obj_add_3 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope

# sample_id: 004
torchrun --nproc_per_node=1 fast_infer.py \
  --video_path assets/dog.mp4 \
  --prompt "Add the brown and white beagle interacting with and drinking from the metallic bowl on the wooden floor." \
  --output_dir results/obj_add_4 \
  --model_name models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope
