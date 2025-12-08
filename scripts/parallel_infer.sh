export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 inference.py \
  --test_json assets/teaser_test.json \
  --output_dir results \
  --model_name /scratch3/yan204/models/Wan2.1-T2V-14B \
  --seed 0 \
  --num_frames 33 \
  --source_frames 33 \
  --reasoning_frames 4 \
  --repeat_rope \
  --videocof_path videocof_weight/videocof.safetensors