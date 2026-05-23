export CUDA_VISIBLE_DEVICES=0,1,2,3

## for fast inference
torchrun --nproc_per_node=4 fast_infer.py \
  --test_json assets/teaser_test.json \
  --output_dir results/parallel_infer_fast \
  --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
  --videocof_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope

# ## for normal inference
# torchrun --nproc_per_node=4 inference.py \
#   --test_json assets/teaser_test.json \
#   --output_dir results/parallel_infer \
#   --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --videocof_path videocof_weight/videocof.safetensors