# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/4tasks_rem_add_swap_local-style_test.json  \
#   --output_dir results/video_cot_4tasks_test_in_domain_200_videos_33_frames \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_4tasks_2epochs_1.3b_repeat_rope/checkpoint-5562.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/4tasks_rem_add_swap_local-style_test.json  \
#   --output_dir results/video_cot_4tasks_gradual_ground_test \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_4tasks_2epochs_1.3b_0-t_0_0-t_sample_stride2/checkpoint-1390.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/4tasks_rem_add_swap_local-style_test.json  \
#   --output_dir results/14b_1-t+1_0_1-t+1_gradual_ground_edit_decouple_decode \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/local_style_multi_instance_v1.json \
#   --output_dir results/local_style_multi_instance_new_prompt_test_g050 \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors

# torchrun --nproc_per_node=1 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/long_video_swap.json \
#   --output_dir results/length_exploration_black_jacket_141_frames_short_prompt \
#   --seed 0 \
#   --num_frames 141 \
#   --source_frames 141 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors


# torchrun --nproc_per_node=1 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/long_video_swap.json \
#   --output_dir results/length_exploration_black_jacket_121_frames\
#   --seed 0 \
#   --num_frames 121 \
#   --source_frames 121 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/4tasks_rem_add_swap_local-style_test.json  \
#   --output_dir results/1.3b_red_mask_test_bench \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_1.3b_red_mask_4node/checkpoint-1390.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/4tasks_rem_add_swap_local-style_test.json  \
#   --output_dir results/1.3b_black_background_test_bench \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_1.3b_black_background_4node/checkpoint-1390.safetensors

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/long_video_test.json  \
#   --output_dir results/length_exploration_121_frames_retest_4w5_weight_retest \
#   --seed 0 \
#   --num_frames 121 \
#   --source_frames 121 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors



# python metric/gpt_evaluation.py \
#     --input_json data/test_json/4tasks_rem_add_swap_local-style_test.json \
#     --output_json score/gray_alpha_0.5_gpt_evaluation.json \
#     --video_root results/gray_alpha_0.5_retest \
#     --edited_video_root results/gray_alpha_0.5_retest \
#     --api_key ${OPENAI_API_KEY} \
#     --model gpt-4o-2024-05-13 \
#     --api_base https://xinyun.ai/v1 \
#     --num_frames 1 \
#     --num_workers 4 \
#     --original_from_compare_left_half \
#     --print_stream 


# python metric/gpt_success_rate.py \
#     --input_json data/test_json/4tasks_rem_add_swap_local-style_test.json \
#     --output_json score/gray_alpha_0.5_gpt_success_rate.json \
#     --video_root results/gray_alpha_0.5_retest \
#     --edited_video_root results/gray_alpha_0.5_retest \
#     --api_key ${OPENAI_API_KEY} \
#     --model gpt-4o-2024-05-13 \
#     --api_base https://xinyun.ai/v1 \
#     --num_workers 4 \
#     --original_from_compare_left_half \
    


# python metric/compute_clip_score.py \
#   --input_json data/test_json/4tasks_rem_add_swap_local-style_test.json\
#   --video_root results/gray_alpha_0.5_retest  \
#   --edited_video_root results/gray_alpha_0.5_retest \
#   --num_frames 33 \
#   --dino_model_name dinov2_vits14 \
#   --output_json score/gray_alpha_0.5_perceptial_score.json



# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/retest.json  \
#   --output_dir results/best_retest_gradual_6w \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_6w_data/checkpoint-6800.safetensors

# torchrun --nproc_per_node=1 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/single_test_motivation.json  \
#   --output_dir results/single_test_start0_stride1_motivation \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json  data/test_json/senorita_obj_swap_test_new.json  \
#   --output_dir results/video_cot_swap_source_as_edit_test_swap_replace \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_swap_source_as_edit_4w5/checkpoint-5139.safetensors

# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/multi_instance_obj_swap_test.json \
#   --output_dir results/12w_4tasks_obj_swap_multi_instance_new_prompt_template_test \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_12w_lora_rank256_alpha128_flash3_14b/checkpoint-7516.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/local_style_multi_instance_v1.json  \
#   --output_dir results/12w_4tasks_local_style_multi_instance_test \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_12w_lora_rank256_alpha128_flash3_14b/checkpoint-7516.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/20_test.json \
#   --output_dir results/8_steps_test_20_videos \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors

# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/20_test.json \
#   --output_dir results/4_steps_test_20_videos_dmd_cot_cfg_1.0 \
#   --seed 0 \
#   --num_frames 33 \
#   --source_frames 33 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors

# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_cot_json.py \
#   --test_json data/test_json/480frames_test.json \
#   --output_dir results/385frames_test \
#   --seed 0 \
#   --num_frames 385 \
#   --source_frames 385 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors

##   --lora_path experiments/video_cot_12w_lora_rank256_alpha128_flash3_14b/checkpoint-7516.safetensors


# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/multi-shot_test.json \
#   --output_dir results/481frames_multi-shot_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 481 \
#   --source_frames 481 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors


# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/multi-shot_test.json \
#   --output_dir results/257frames_multi-shot_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 257 \
#   --source_frames 257 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors


# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/480frames_test.json \
#   --output_dir results/449frames_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 449 \
#   --source_frames 449 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors


# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/480frames_test.json \
#   --output_dir results/481frames_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 481 \
#   --source_frames 481 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors

# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/480frames_test.json \
#   --output_dir results/485frames_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 485 \
#   --source_frames 485 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors


# torchrun --nproc_per_node=2 examples/wan2.1/predict_v2v_dmd_cot_json.py \
#   --test_json data/test_json/480frames_test.json \
#   --output_dir results/513frames_test_dmd_4steps \
#   --seed 0 \
#   --num_frames 513 \
#   --source_frames 513 \
#   --reasoning_frames 4 \
#   --repeat_rope \
#   --lora_path experiments/video_cot_gradual_ground_1-t+1_0_1-t+1_14b/checkpoint-5562.safetensors \
#   --enable_acceleration_lora \
#   --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors


torchrun --nproc_per_node=1 examples/wan2.1/predict_v2v_dmd_cot_json.py \
  --model_name models/Wan2.1-T2V-14B \
  --test_json assets/multi-shot_test.json \
  --output_dir results/daiyu_513frames_multi-shot_test_dmd_4steps \
  --seed 0 \
  --num_frames 513 \
  --source_frames 513 \
  --reasoning_frames 4 \
  --repeat_rope \
  --lora_path videocof_weight/videocof.safetensors \
  --enable_acceleration_lora \
  --acceleration_lora_path videocof_weight/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors
