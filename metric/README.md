# VideoCoF evaluation

The [official VideoCoF-Bench release](https://huggingface.co/datasets/XiangpengYang/VideoCoF-Bench) provides input videos and editing instructions for four tasks. This directory contains the recovered CLIP/DINO evaluation code, the GPT-4o judge prompts, an API connectivity check, and a manifest adapter for inference and evaluation.

## Benchmark and evaluation manifest

Download the videos and the combined `videocof_edit.jsonl`:

```bash
hf download XiangpengYang/VideoCoF-Bench --repo-type dataset --local-dir data/VideoCoF-Bench
```

Each public record has `id`, `video` (relative to the dataset root), and `edit_instruction`. Prepare a manifest with resolved input and expected output paths:

```bash
python metric/prepare_benchmark.py \
  --bench-root data/VideoCoF-Bench \
  --edited-root results/videocof_bench \
  --output-json results/videocof_bench_manifest.json
```

The resulting JSON can be used by both the CoT inference script and the evaluators. It retains every instruction, including multiple edits of the same input video. Expected edited-only outputs are named `gen_{id}.mp4`, for example `gen_obj_swap_instance_023_1.mp4`. Use `--edited-pattern` for another method's filenames and `--check-outputs` to check that all edited outputs exist before scoring. Comparison montages are not edited-only outputs.

The current release contains **240 editing samples**, including recovered instance-editing cases. The paper describes a 200-sample benchmark. An exact mapping to the original Table 1 subset and its complete per-sample score files has not been recovered; the current manifest should not be presented as that exact historical evaluation manifest.

## Inference configuration

Use the existing 50-step CoT path to evaluate the full model:

```bash
MODEL_NAME=models/Wan2.1-T2V-14B \
LORA_PATH=videocof_weight/videocof.safetensors \
TEST_JSON=results/videocof_bench_manifest.json \
OUTPUT_DIR=results/videocof_bench \
USE_DMD=0 NPROC_PER_NODE=1 NUM_FRAMES=33 SOURCE_FRAMES=33 REASONING_FRAMES=4 SEED=0 \
bash scripts/wan2.1/test_cot_lora.sh
```

The [50-step implementation](../examples/wan2.1/predict_v2v_cot_json.py) uses Wan2.1-T2V-14B, Flow-UniPC, guidance scale 5.0, shift 3, bf16, LoRA scale 1.0, repeated RoPE, and 10 fps output. Its negative prompt and TeaCache settings are in that file. The generator uses `seed + rank`; input sampling also uses the PyTorch CPU RNG, so this recovered script is not a guarantee of bitwise reproduction of the historical run.

The repository's default fast demo uses a separate 4-step acceleration LoRA path. The command above explicitly selects the 50-step path. These are available code configurations; an exact checkpoint/run/subset mapping for Table 1 is not included.

## CLIP and DINO settings

Install the evaluation dependencies in the PyTorch environment used by the project:

```bash
pip install -r metric/requirements.txt
```

Run the historical launcher settings:

```bash
INPUT_JSON=results/videocof_bench_manifest.json \
VIDEO_ROOT=data/VideoCoF-Bench \
EDITED_VIDEO_ROOT=results/videocof_bench \
OUTPUT_JSON=results/videocof_bench_clip.json \
bash scripts/evaluation/clip_eval.sh
```

| Setting | Recovered implementation |
| --- | --- |
| CLIP checkpoint | OpenAI CLIP `ViT-B/32`; `clip.load` preprocessing and tokenization |
| Text | The editing instruction, with surrounding whitespace stripped |
| Requested frames | 33 per edited video in the historical shell launcher; the Python CLI's legacy default is 3 |
| CLIP-T | Mean raw CLIP image/text logit over sampled edited frames; includes the model's learned logit scale |
| CLIP-F | Mean cosine similarity of consecutive sampled edited-frame CLIP embeddings |
| DINO | `dinov2_vits14` from `facebookresearch/dinov2`, with consecutive edited-frame cosine similarity |
| Aggregation | Mean of valid per-video scores; coverage and failures are reported |

CLIP-F and DINO measure temporal consistency within the edited video. CLIP-T is not an unscaled cosine or a target-caption score. For known video lengths, the sampler rounds evenly spaced indices including the endpoints. The historical implementation falls back to the first N frames if the reader reports an unknown/infinite length. This behavior is preserved and recorded instead of silently changing the metric.

`CLIP_MODEL`, `NUM_FRAMES`, `DINO_MODEL_NAME`, `DINO_REPO`, and `DEVICE` can be set explicitly. The original CLIP package commit, DINO Hub revision, and full dependency lockfile were not recorded in the recovered run artifacts. Report these choices when comparing new scores. Failed samples must be resolved or explicitly reported as a partial evaluation.

## GPT-4o judge and API check

Set credentials and the model through environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-2024-05-13"
python metric/ping_judge_api.py
```

The ping sends a small chat request and reports the requested and returned model IDs. The evaluators use the OpenAI-compatible `/chat/completions` interface. The dated model is preserved from the historical launchers and recovered ablation results. Availability depends on the provider; if unavailable, explicitly set `OPENAI_MODEL` and report the replacement. No automatic model substitution is performed.

Run the three-axis judge and the instance-editing success judge separately:

```bash
INPUT_JSON=results/videocof_bench_manifest.json \
VIDEO_ROOT=data/VideoCoF-Bench \
EDITED_VIDEO_ROOT=results/videocof_bench \
OUTPUT_JSON=results/videocof_bench_gpt_scores.json \
bash scripts/evaluation/gpt_evaluation.sh

INPUT_JSON=results/videocof_bench_manifest.json \
VIDEO_ROOT=data/VideoCoF-Bench \
EDITED_VIDEO_ROOT=results/videocof_bench \
OUTPUT_JSON=results/videocof_bench_gpt_success.json \
bash scripts/evaluation/gpt_success_rate.sh
```

| Setting | Three-axis judge | Instance success judge |
| --- | --- | --- |
| Historical model | `gpt-4o-2024-05-13` | `gpt-4o-2024-05-13` |
| Video evidence | 3 sampled frames from each input/output | First frame of each input/output |
| Temperature | 0.1 | 0 |
| Rubric | Instruction following, visual quality, preservation; each 1–10 | Strict yes/no, including whether the intended instance alone was edited |
| Exact prompt | [`EVALUATION_PROMPT_TEXT` and `build_evaluation_messages`](gpt_evaluation.py) | [`EVALUATION_PROMPT_TEXT` and `build_evaluation_messages`](gpt_success_rate.py) |

Both exact historical rubrics are retained in the source. For unknown video lengths, the three-axis sampler has the same first-N fallback described above. The success judge is a first-frame assessment, not a full-video temporal success test. Failed or unparseable requests are not converted into zero scores. New results record the requested/returned model, request settings, sampling information, and rubric hash. `NUM_WORKERS` defaults to 4, `REQUEST_TIMEOUT` to 120 seconds, and `MAX_RETRIES` to 0; these are explicit runtime settings, not evidence of the historical provider's behavior.

## Historical configurations and release status

- **Training data:** [VideoCoF-50k](https://huggingface.co/datasets/XiangpengYang/VideoCoF-50k) contains `train.json` and the four task archives/annotation files. At [revision `9b8ada7`](https://huggingface.co/datasets/XiangpengYang/VideoCoF-50k/tree/9b8ada741b54f11876ccdb043e3508f938b0cdf2), `train.json` contains 49,177 entries; 4,796 swap entries lack editing instructions. The standalone `obj_swap.json` also contains 35 absolute edited-video paths. These metadata issues need a corrected release; no separately versioned correction has been identified. The benchmark release does not change the training annotations.
- **Public 14B training launcher:** [`train_joint_img_cot_video_lora.sh`](../scripts/wan2.1/train_joint_img_cot_video_lora.sh) uses LoRA rank 128 / alpha 64, 33 source + 33 edited frames, one reasoning frame, eight processes by default, two epochs, learning rate 1e-4, bf16, and DeepSpeed ZeRO-2.
- **Public Slurm example:** [`slurm_video_gradual_cot_14b.sh`](../scripts/wan2.1/slurm_video_gradual_cot_14b.sh) uses rank 256 / alpha 128 and four reasoning frames on 16 GPUs. This is a different example configuration; see the [training guide](../scripts/wan2.1/README_TRAIN_VIDEOCOF.md).
- **Ablations:** historical no-reasoning, repeated-RoPE, mask, and grounding experiments appear in the retained [`scripts/test/test_cot_lora.sh`](../scripts/test/test_cot_lora.sh) command history. Their original checkpoints and complete training/run manifests have not been recovered. These snippets are references, not a complete executable reproduction package.
- **Table 1:** the metric definitions, exact judge rubrics, and the dated judge name above are recovered evidence. The available per-sample judge results are ablations, not a verified Table 1 main-run result set. Do not use them to claim a rerun of the original table.

For new evaluations, retain the dataset revision, prepared manifest, checkpoint hashes, dependency versions, launch command, and evaluation JSON with coverage. This makes the evaluated samples and settings explicit.
