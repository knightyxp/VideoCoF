import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import imageio

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              WanTransformer3DModel)
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, save_videos_grid)
from videox_fun.data.dataset_image_video import derive_ground_object_from_instruction
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def load_video_frames(
    video_path: str,
    source_frames: int = None,
):
    assert source_frames is not None, "请传入 source_frames"

    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)

    stride = max(1, total_frames // source_frames)
    start_frame = torch.randint(0, max(1, total_frames - stride * source_frames), (1,))[0].item()

    frames = []
    original_height, original_width = None, None

    for i in range(source_frames):
        idx = start_frame + i * stride
        if idx >= total_frames:
            break
        try:
            frame = reader.get_data(idx)
            pil_frame = Image.fromarray(frame)
            if original_height is None:
                original_width, original_height = pil_frame.size
                print(f"Original video dimensions: {original_width}x{original_height}")
            frames.append(pil_frame)
        except IndexError:
            break

    reader.close()

    while len(frames) < source_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            w, h = (original_width, original_height) if original_width else (832, 480)
            frames.append(Image.new('RGB', (w, h), (0, 0, 0)))

    assert len(frames) == source_frames
    print(f"Loaded {source_frames} source frames")

    input_video = torch.from_numpy(np.array(frames))
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    input_video = input_video * (2.0 / 255.0) - 1.0

    return input_video, original_height, original_width


def parse_args():
    parser = argparse.ArgumentParser(description="Video-to-video CoT reasoning generation from JSON task list with parallel inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for editing")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_name", type=str, default="/scratch3/yan204/models/Wan2.1-T2V-14B", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible generation")
    parser.add_argument("--videocof_path", type=str, default=None, help="Path to videocof weight checkpoint")
    parser.add_argument("--num_frames", type=int, default=65, help="Total number of frames (input + generated)")
    parser.add_argument("--source_frames", type=int, default=33, help="Number of source frames; default 33")
    parser.add_argument("--reasoning_frames", type=int, default=4, help="Grounding frames in the middle segment (pixel-space)")
    parser.add_argument("--repeat_rope", action="store_true", help="Enable repeat temporal RoPE for src and tgt segments")
    return parser.parse_args()


# Defaults aligned with predict_v2v_json_new.py
GPU_memory_mode = "sequential_cpu_offload"
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False
cfg_skip_ratio = 0
enable_riflex = False
riflex_k = 6

config_path = "config/wan2.1/wan_civitai.yaml"
model_name = "/scratch3/yan204/models/Wan2.1-T2V-14B"
sampler_name = "Flow_Unipc"
shift = 3
transformer_path = None
vae_path = None

fps = 10
weight_dtype = torch.bfloat16
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
guidance_scale = 5.0
num_inference_steps = 50
lora_weight = 1.0


def save_results(tensor: torch.Tensor, file_path: str, fps_out: int = 16):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    B, C, T, H, W = tensor.shape
    arr = tensor[0].cpu().numpy()
    if T == 1:
        img = arr[:, 0].transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(file_path)
    else:
        save_videos_grid(tensor, file_path, fps=fps_out)
    print(f"Saved video → {file_path}")


def _normalize_to_01(video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vmin = float(video.min())
        vmax = float(video.max())
        if vmin < 0.0 or vmax > 1.0:
            video = (video + 1.0) / 2.0
        return video.clamp(0.0, 1.0)


def save_side_by_side(input_tensor: torch.Tensor, sample_tensor: torch.Tensor, file_path: str, fps_out: int = 16):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    a = _normalize_to_01(input_tensor.detach().cpu())
    b = _normalize_to_01(sample_tensor.detach().cpu())

    # Align dimensions by cropping to the minimum across T/H/W
    T = min(a.shape[2], b.shape[2])
    H = min(a.shape[3], b.shape[3])
    W = min(a.shape[4], b.shape[4])
    a = a[:, :, :T, :H, :W]
    b = b[:, :, :T, :H, :W]

    combined = torch.cat([a, b], dim=4)
    save_videos_grid(combined, file_path, fps=fps_out)
    print(f"Saved side-by-side video → {file_path}")


def derive_ground_instruction(edit_instruction_text: str) -> str:
    # Keep wrapper for backward compatibility; reuse the same rule as training dataset
    return derive_ground_object_from_instruction(edit_instruction_text)


def main():
    args = parse_args()

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Running parallel CoT inference with {world_size} GPUs")
        print(f"Using seed: {args.seed}")

    model_name = args.model_name

    # Load tasks
    fname = os.path.basename(args.video_path)
    item = {
        "source_video_path": args.video_path,
        "edit_instruction": args.prompt
    }
    items = [(fname, item)]

    # Filter done
    pending_items = []
    for fname, item in items:
        base = os.path.splitext(fname)[0]
        output_video_path = os.path.join(args.output_dir, f"gen_{base}.mp4")
        if not os.path.exists(output_video_path):
            pending_items.append((fname, item))

    if rank == 0:
        print(f"Total items: {len(items)}, already generated: {len(items) - len(pending_items)}, pending: {len(pending_items)}")

    # Shard across GPUs
    subset_items = pending_items[rank::world_size] if world_size > 0 else pending_items

    print(f"[GPU {rank} | local {local_rank}] Processing {len(subset_items)} items")

    device = torch.device(f"cuda:{local_rank}")

    # Load config and models
    config = OmegaConf.load(config_path)

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        print(f"[GPU {rank}] Loading transformer from checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"[GPU {rank}] Missing keys: {len(m)}, unexpected keys: {len(u)}")

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"[GPU {rank}] Loading VAE from checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"[GPU {rank}] Missing keys: {len(m)}, unexpected keys: {len(u)}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    Choosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    pipeline = WanPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Memory mode
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    # LoRA
    if args.videocof_path is not None:
        pipeline = merge_lora(pipeline, args.videocof_path, lora_weight, device=device)
        print(f"[GPU {rank}] Loaded LoRA from {args.videocof_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    # Grounding indices are now handled inside the pipeline; no forward override needed.

    for fname, item in subset_items:
        base = os.path.splitext(fname)[0]
        output_video_path = os.path.join(args.output_dir, f"gen_{base}.mp4")
        info_path = os.path.join(args.output_dir, f"gen_{base}_info.txt")

        print(f"[GPU {rank}] Processing {fname}...")

        video_path = item["source_video_path"]

        # Match training dataset (ImageVideoCoTDataset) prompt formatting
        edit_text = item.get('text', item.get('qwen_vl_72b_refined_instruction', item.get('edit_instruction', '')))
        ground_instr = derive_ground_instruction(edit_text)
        prompt = (
            "A video sequence showing three parts: first the original scene, "
            f"then grounded {ground_instr}, and finally the same scene but {edit_text}"
        )


        input_video, video_height, video_width = load_video_frames(
            video_path,
            source_frames=args.source_frames,
        )

        with torch.no_grad():
            sample = pipeline(
                video=input_video,
                prompt=prompt,
                num_frames=args.num_frames,
                source_frames=args.source_frames,
                reasoning_frames=args.reasoning_frames,
                negative_prompt=negative_prompt,
                height=video_height,
                width=video_width,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                shift=shift,
                repeat_rope=args.repeat_rope,
                cot=True,
            ).videos

        reason_edit_path = os.path.join(args.output_dir, f"gen_{base}_reason_edit.mp4")
        save_results(sample, reason_edit_path, fps)
        print(f"[GPU {rank}] Saved reason+edit video shape: {sample.shape}")

        edit_video = sample[:, :, -args.source_frames:, :, :]
        save_results(edit_video, output_video_path, fps)
        print(f"[GPU {rank}] Edit video shape: {edit_video.shape}")

        compare_path = os.path.join(args.output_dir, f"gen_{base}_compare.mp4")
        save_side_by_side(input_video, edit_video, compare_path, fps)

        with open(info_path, "w", encoding="utf-8") as info_f:
            info_f.write(prompt)

        print(f"[GPU {rank}] Completed {fname}")

    if args.videocof_path is not None:
        pipeline = unmerge_lora(pipeline, args.videocof_path, lora_weight, device=device)

    print(f"[GPU {rank}] Finished processing all assigned items")


if __name__ == "__main__":
    main()


