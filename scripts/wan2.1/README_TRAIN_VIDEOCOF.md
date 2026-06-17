# VideoCoF Wan2.1 Training

This directory contains the minimal Wan2.1 training code used for VideoCoF CoT LoRA training.
It intentionally does not include the older test/inference scripts from VideoX-Fun.

## Files

- `train_joint_img_video_lora.py`: common python launcher for 14B/1.3B joint dataset training.
- `validation.py`: optional prompt and final DDP validation helpers used by the training entry.
- `train_joint_img_cot_video_lora.sh`: 14B launch script for VideoCoF-50k style CoT training.
- `train_1.3b.sh`: 1.3B launch script for training with `train_lora.py`.
- `train_lora.py`: 1.3B training implementation.
- `slurm_video_gradual_cot_14b.sh`: Slurm 16-GPU example for 14B gradual CoT LoRA training.
- `test_cot_lora.sh`: `torchrun` parallel test wrapper for CoT LoRA inference.
- `../../examples/wan2.1/predict_v2v_cot_json.py`: 50-step parallel CoT inference from a JSON task list.
- `../../examples/wan2.1/predict_v2v_dmd_cot_json.py`: 4-step DMD/FusionX parallel CoT inference from JSON or a single video.

## Dataset Format

`--use_cot_dataset` expects a JSON list or dict. Video samples should contain:

```json
{
  "type": "video",
  "original_video": "relative/or/absolute/source.mp4",
  "grounded_video": "relative/or/absolute/grounded.mp4",
  "edited_video": "relative/or/absolute/edited.mp4",
  "edit_instruction": "remove the red car"
}
```

`ground_video` is also accepted as an alias for `grounded_video`. Image samples can use `original_image` and `edited_image`.

## Launch

Set paths through environment variables so local machine paths do not need to be edited into the script.

### 14B CoT entry (recommended for VideoCoF-50k)

```bash
export MODEL_NAME=/path/to/Wan2.1-T2V-14B
export DATASET_NAME=/path/to/VideoCoF-50k
export DATASET_META_NAME=/path/to/VideoCoF-50k/train.json
export OUTPUT_DIR=experiments/videocof_wan2.1_14b_lora

bash scripts/wan2.1/train_joint_img_cot_video_lora.sh
```

### 1.3B entry

```bash
export MODEL_NAME=/path/to/Wan2.1-T2V-1.3B
export DATASET_NAME=/path/to/VideoCoF-50k
export DATASET_META_NAME=/path/to/VideoCoF-50k/train.json
export OUTPUT_DIR=experiments/videocof_wan2.1_1.3b_lora

bash scripts/wan2.1/train_1.3b.sh
```

For 14B (`train_joint_img_cot_video_lora.sh`), the default command trains a rank-128 LoRA with DeepSpeed ZeRO-2 bf16 config in `config/14b_lora_zero2_bf16_config.json`.
Install `deepspeed` separately if you use the default launch script. Parquet-based NHR image data additionally requires `pandas` and `pyarrow`.

For Slurm clusters, edit the `#SBATCH --account` line and export the same path variables before submitting:

```bash
sbatch scripts/wan2.1/slurm_video_gradual_cot_14b.sh
```

## Parallel Test

```bash
export MODEL_NAME=/path/to/Wan2.1-T2V-14B
export TEST_JSON=/path/to/test.json
export LORA_PATH=/path/to/videocof.safetensors
export ACCELERATION_LORA_PATH=/path/to/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors
export OUTPUT_DIR=results/videocof_cot_parallel_test
export NPROC_PER_NODE=4

bash scripts/wan2.1/test_cot_lora.sh
```

Set `USE_DMD=0` to run the 50-step `predict_v2v_cot_json.py` path instead of the default 4-step DMD path.
