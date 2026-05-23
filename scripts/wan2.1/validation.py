import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.data.dataset_image_video import derive_ground_object_from_instruction
from videox_fun.models import WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
try:
    from videox_fun.pipeline import WanI2VPipeline
except ImportError:
    WanI2VPipeline = None
from videox_fun.utils.lora_utils import merge_lora
from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

logger = get_logger(__name__, log_level="INFO")

DEFAULT_NEG_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def filter_kwargs(cls, kwargs):
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def resolve_model_path(model_root, subpath, default_subpath):
    subpath = str(subpath or default_subpath)
    if os.path.isabs(subpath):
        return subpath

    model_root_candidate = os.path.join(model_root, subpath)
    if os.path.exists(model_root_candidate):
        return model_root_candidate

    sibling_candidate = os.path.join(os.path.dirname(os.path.abspath(model_root)), subpath)
    if os.path.exists(sibling_candidate):
        return sibling_candidate

    return subpath


def load_video_frames(video_path: str, source_frames: int) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    assert source_frames is not None and source_frames > 0, "source_frames must be provided"

    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)

    stride = max(1, total_frames // source_frames)
    start_frame = torch.randint(0, max(1, total_frames - stride * source_frames), (1,))[0].item()

    frames: List[Image.Image] = []
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
                logger.info(f"Validation video dimensions: {original_width}x{original_height}")
            frames.append(pil_frame)
        except IndexError:
            break

    reader.close()

    while len(frames) < source_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            w, h = (original_width, original_height) if original_width else (832, 480)
            frames.append(Image.new("RGB", (w, h), (0, 0, 0)))

    assert len(frames) == source_frames
    logger.info(f"Loaded {source_frames} validation source frames")

    input_video = torch.from_numpy(np.array(frames))
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    input_video = input_video * (2.0 / 255.0) - 1.0

    return input_video, original_height, original_width


def _normalize_to_01(video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vmin = float(video.min())
        vmax = float(video.max())
        if vmin < 0.0 or vmax > 1.0:
            video = (video + 1.0) / 2.0
        return video.clamp(0.0, 1.0)


def save_results(tensor: torch.Tensor, file_path: str, fps_out: int = 16) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    B, C, T, H, W = tensor.shape
    arr = tensor[0].cpu().numpy()
    if T == 1:
        img = arr[:, 0].transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(file_path)
    else:
        save_videos_grid(tensor, file_path, fps=fps_out)
    logger.info(f"Saved validation video → {file_path}")


def save_side_by_side(input_tensor: torch.Tensor, sample_tensor: torch.Tensor, file_path: str, fps_out: int = 16) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    a = _normalize_to_01(input_tensor.detach().cpu())
    b = _normalize_to_01(sample_tensor.detach().cpu())

    T = min(a.shape[2], b.shape[2])
    H = min(a.shape[3], b.shape[3])
    W = min(a.shape[4], b.shape[4])
    a = a[:, :, :T, :H, :W]
    b = b[:, :, :T, :H, :W]

    combined = torch.cat([a, b], dim=4)
    save_videos_grid(combined, file_path, fps=fps_out)
    logger.info(f"Saved validation side-by-side video → {file_path}")


def _build_validation_pipeline(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
):
    transformer3d_val = WanTransformer3DModel.from_pretrained(
        resolve_model_path(
            args.pretrained_model_name_or_path,
            config["transformer_additional_kwargs"].get("transformer_subpath", "transformer"),
            "transformer",
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
    ).to(weight_dtype)
    transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = WanPipeline(
        vae=accelerator.unwrap_model(vae).to(weight_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=transformer3d_val,
        scheduler=scheduler,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline = merge_lora(
        pipeline,
        None,
        1,
        accelerator.device,
        state_dict=accelerator.unwrap_model(network).state_dict(),
        transformer_only=True,
    )

    return pipeline, transformer3d_val


def _prepare_generator(args, accelerator):
    if args.seed is None:
        return None
    return torch.Generator(device=accelerator.device).manual_seed(args.seed)


def _run_prompt_validation(pipeline, args, accelerator, generator, weight_dtype, global_step):
    prompts = args.validation_prompts or []
    if len(prompts) == 0:
        return

    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            if args.train_mode != "normal":
                with torch.autocast("cuda", dtype=weight_dtype):
                    video_length = (
                        int(
                            (args.video_sample_n_frames - 1)
                            // pipeline.vae.config.temporal_compression_ratio
                            * pipeline.vae.config.temporal_compression_ratio
                        )
                        + 1
                        if args.video_sample_n_frames != 1
                        else 1
                    )
                    input_video, input_video_mask, _ = get_image_to_video_latent(
                        None,
                        None,
                        video_length=video_length,
                        sample_size=[args.video_sample_size, args.video_sample_size],
                    )
                    sample = pipeline(
                        prompt,
                        num_frames=video_length,
                        negative_prompt=args.validation_negative_prompt or DEFAULT_NEG_PROMPT,
                        height=args.video_sample_size,
                        width=args.video_sample_size,
                        guidance_scale=args.validation_guidance_scale,
                        generator=generator,
                        video=input_video,
                        mask_video=input_video_mask,
                    ).videos
                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                    video_length = 1
                    input_video, input_video_mask, _ = get_image_to_video_latent(
                        None,
                        None,
                        video_length=video_length,
                        sample_size=[args.video_sample_size, args.video_sample_size],
                    )
                    sample = pipeline(
                        prompt,
                        num_frames=video_length,
                        negative_prompt=args.validation_negative_prompt or DEFAULT_NEG_PROMPT,
                        height=args.video_sample_size,
                        width=args.video_sample_size,
                        guidance_scale=args.validation_guidance_scale,
                        generator=generator,
                        video=input_video,
                        mask_video=input_video_mask,
                    ).videos
                    save_videos_grid(
                        sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif")
                    )
            else:
                with torch.autocast("cuda", dtype=weight_dtype):
                    sample = pipeline(
                        prompt,
                        num_frames=args.video_sample_n_frames,
                        negative_prompt=args.validation_negative_prompt or DEFAULT_NEG_PROMPT,
                        height=args.video_sample_size,
                        width=args.video_sample_size,
                        guidance_scale=args.validation_guidance_scale,
                        generator=generator,
                    ).videos
                    os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)
                    save_videos_grid(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.gif"))

                    sample = pipeline(
                        prompt,
                        num_frames=1,
                        negative_prompt=args.validation_negative_prompt or DEFAULT_NEG_PROMPT,
                        height=args.video_sample_size,
                        width=args.video_sample_size,
                        guidance_scale=args.validation_guidance_scale,
                        generator=generator,
                    ).videos
                    save_videos_grid(
                        sample, os.path.join(args.output_dir, f"sample/sample-{global_step}-image-{i}.gif")
                    )


def _prepare_task_items(task_json: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    with open(task_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    if limit is not None and limit > 0:
        return items[:limit]
    return items


def _derive_prompt_text(item: Dict[str, Any]) -> str:
    edit_text = item.get(
        "text",
        item.get("qwen_vl_72b_refined_instruction", item.get("edit_instruction", "")),
    )
    ground_instr = derive_ground_object_from_instruction(edit_text)
    return (
        "A video sequence showing three parts: first the original scene, "
        f"then grounded {ground_instr}, and finally the same scene but {edit_text}"
    )
    # return edit_text


def _run_task_json_validation(pipeline, args, generator, weight_dtype, global_step):
    task_json = getattr(args, "validation_task_json", None)
    if not task_json:
        return

    if not os.path.exists(task_json):
        logger.warning(f"Validation task json {task_json} not found, skip CoT validation.")
        return

    try:
        items = _prepare_task_items(task_json, getattr(args, "validation_task_limit", None))
    except Exception as exc:
        logger.error(f"Failed to parse validation task json {task_json}: {exc}")
        return

    if len(items) == 0:
        logger.warning("Validation task json is empty, skip.")
        return

    output_dir = args.validation_task_output_dir or os.path.join(args.output_dir, "validation_videos")
    os.makedirs(output_dir, exist_ok=True)

    fps = getattr(args, "validation_fps", 10)
    neg_prompt = args.validation_negative_prompt or DEFAULT_NEG_PROMPT

    for idx, item in enumerate(items):
        base_name = item.get("task_name") or f"{item.get('task_type', 'task')}_{item.get('sample_id', idx)}"
        source_video = item.get("source_video_path")
        if not source_video or not os.path.exists(source_video):
            logger.warning(f"[validation] Missing video for task {base_name}, skip.")
            continue

        prompt = _derive_prompt_text(item)
        try:
            input_video, video_height, video_width = load_video_frames(
                source_video,
                source_frames=args.validation_source_frames,
            )
        except Exception as exc:
            logger.error(f"[validation] Failed to load frames for {source_video}: {exc}")
            continue

        with torch.no_grad():
            with torch.autocast("cuda", dtype=weight_dtype):
                sample = pipeline(
                    video=input_video,
                    prompt=prompt,
                    num_frames=args.validation_num_frames,
                    source_frames=args.validation_source_frames,
                    reasoning_frames=args.validation_reasoning_frames,
                    negative_prompt=neg_prompt,
                    height=video_height,
                    width=video_width,
                    generator=generator,
                    guidance_scale=args.validation_guidance_scale,
                    num_inference_steps=args.validation_num_inference_steps,
                    shift=shift,
                    repeat_rope=repeat_rope,
                    cot=True,
                ).videos

        step_tag = f"step{global_step:06d}"
        reason_edit_path = os.path.join(output_dir, f"{step_tag}_gen_{base_name}_reason_edit.mp4")
        save_results(sample, reason_edit_path, fps_out=fps)
        logger.info(f"[validation] Reason+edit video shape: {tuple(sample.shape)}")

        edit_video = sample[:, :, -args.validation_source_frames :, :, :]
        edit_path = os.path.join(output_dir, f"{step_tag}_gen_{base_name}.mp4")
        save_results(edit_video, edit_path, fps_out=fps)
        logger.info(f"[validation] Edit video shape: {tuple(edit_video.shape)}")

        compare_path = os.path.join(output_dir, f"{step_tag}_gen_{base_name}_compare.mp4")
        save_side_by_side(input_video, edit_video, compare_path, fps_out=fps)

        info_path = os.path.join(output_dir, f"{step_tag}_gen_{base_name}_info.txt")
        with open(info_path, "w", encoding="utf-8") as info_f:
            info_f.write(prompt)


def _run_ddp_task_validation(
    pipeline,
    args,
    accelerator,
    weight_dtype,
    global_step,
):
    """Run distributed validation where each GPU processes a subset of tasks (DDP style)."""
    task_json = getattr(args, "validation_json", None)
    if not task_json or not os.path.exists(task_json):
        if accelerator.is_main_process:
            logger.warning(f"Task json not found: {task_json}")
        return

    # Load all tasks on all processes
    with open(task_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    if len(items) == 0:
        if accelerator.is_main_process:
            logger.warning("Task json is empty")
        return

    # Shard tasks across GPUs: each GPU gets every Nth item (DDP pattern)
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    subset_items = items[rank::world_size]
    
    logger.info(f"[GPU {rank}/{world_size}] Processing {len(subset_items)} out of {len(items)} tasks")

    output_dir = os.path.join(args.output_dir, "validation_videos")
    os.makedirs(output_dir, exist_ok=True)

    # Default parameters aligned with predict_v2v_cot_json.py
    fps = 10
    neg_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    guidance_scale = 5.0
    num_inference_steps = 50
    shift = 3.0
    repeat_rope = True
    
    # Get source_frames and reasoning_frames from training args
    source_frames = getattr(args, "source_frames", 33)
    reasoning_frames = getattr(args, "reasoning_frames", 4)
    edit_frames = getattr(args, "edit_frames", 33)
    # Calculate total frames: source + reasoning + target (where target = source)
    num_frames = source_frames + reasoning_frames + edit_frames
    
    # Set up generator with different seed per GPU
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed + rank)
    else:
        generator = None

    for idx, item in enumerate(subset_items):
        base_name = item.get("task_name") or f"{item.get('task_type', 'task')}_{item.get('sample_id', idx)}"
        source_video = item.get("source_video_path")
        
        if not source_video or not os.path.exists(source_video):
            logger.warning(f"[GPU {rank}] Missing video for task {base_name}, skip")
            continue

        # Derive prompt using CoT format
        prompt = _derive_prompt_text(item)

        # Load video frames
        try:
            input_video, video_height, video_width = load_video_frames(source_video, source_frames)
        except Exception as exc:
            logger.error(f"[GPU {rank}] Failed to load frames for {source_video}: {exc}")
            continue

        # Run inference
        try:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=weight_dtype):
                    sample = pipeline(
                        video=input_video,
                        prompt=prompt,
                        num_frames=num_frames,
                        source_frames=source_frames,
                        reasoning_frames=reasoning_frames,
                        negative_prompt=neg_prompt,
                        height=video_height,
                        width=video_width,
                        generator=generator,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        shift=shift,
                        repeat_rope=repeat_rope,
                        cot=True,
                    ).videos

            # Save results (naming aligned with predict_v2v_cot_json.py)
            reason_edit_path = os.path.join(output_dir, f"gen_{base_name}_reason_edit.mp4")
            save_results(sample, reason_edit_path, fps_out=fps)
            logger.info(f"[GPU {rank}] Saved {base_name}, shape: {tuple(sample.shape)}")

            edit_video = sample[:, :, -source_frames:, :, :]
            edit_path = os.path.join(output_dir, f"gen_{base_name}.mp4")
            save_results(edit_video, edit_path, fps_out=fps)

            # Save side-by-side comparison
            compare_path = os.path.join(output_dir, f"gen_{base_name}_compare.mp4")
            save_side_by_side(input_video, edit_video, compare_path, fps_out=fps)

            info_path = os.path.join(output_dir, f"gen_{base_name}_info.txt")
            with open(info_path, "w", encoding="utf-8") as info_f:
                info_f.write(prompt)

        except Exception as exc:
            logger.error(f"[GPU {rank}] Failed to process {base_name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"[DDP validation] All {world_size} GPUs finished. Results saved to {output_dir}")


def log_validation(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
    global_step,
):
    pipeline = None
    transformer3d_val = None
    try:
        logger.info("Running validation...")
        pipeline, transformer3d_val = _build_validation_pipeline(
            vae,
            text_encoder,
            tokenizer,
            clip_image_encoder,
            transformer3d,
            network,
            config,
            args,
            accelerator,
            weight_dtype,
        )

        generator = _prepare_generator(args, accelerator)
        _run_prompt_validation(pipeline, args, accelerator, generator, weight_dtype, global_step)
        _run_task_json_validation(pipeline, args, generator, weight_dtype, global_step)
    except Exception as exc:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.error(f"Validation failed: {exc}")
    finally:
        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_final_ddp_validation(
    vae,
    text_encoder,
    tokenizer,
    clip_image_encoder,
    transformer3d,
    network,
    config,
    args,
    accelerator,
    weight_dtype,
    global_step,
):
    """
    Run a final distributed validation after training completes, using the same hardware.
    Each GPU processes a different subset of tasks in parallel (DDP style).
    """
    if not getattr(args, "run_final_ddp_validation", False):
        return

    task_json = getattr(args, "validation_json", None)
    if task_json is None:
        if accelerator.is_main_process:
            logger.warning("run_final_ddp_validation is enabled but no validation_json was provided. Skipping.")
        return

    # Get validation parameters from training args
    source_frames = getattr(args, "source_frames", 33)
    reasoning_frames = getattr(args, "reasoning_frames", 4)
    
    if accelerator.is_main_process:
        logger.info(f"[DDP eval] Using in-memory weights (from training) for validation.")
        logger.info(f"[DDP eval] Running distributed validation with {accelerator.num_processes} GPUs...")
        logger.info(f"[DDP eval] Validation task json: {task_json}")
        logger.info(f"[DDP eval] source_frames={source_frames}, reasoning_frames={reasoning_frames}")

    # Set up output directory
    original_output_dir = args.output_dir
    eval_output_dir = getattr(args, "validation_output_dir", None) or os.path.join(original_output_dir, "final_validation")
    
    if accelerator.is_main_process:
        os.makedirs(eval_output_dir, exist_ok=True)
        logger.info(f"[DDP eval] Output directory: {eval_output_dir}")

    # Temporarily update args.output_dir for validation
    args.output_dir = eval_output_dir

    accelerator.wait_for_everyone()

    # Build pipeline on all processes
    pipeline = None
    transformer3d_val = None
    try:
        pipeline, transformer3d_val = _build_validation_pipeline(
            vae,
            text_encoder,
            tokenizer,
            clip_image_encoder,
            transformer3d,
            network,
            config,
            args,
            accelerator,
            weight_dtype,
        )

        # Run DDP validation - each GPU processes different tasks
        _run_ddp_task_validation(pipeline, args, accelerator, weight_dtype, global_step)

    except Exception as exc:
        logger.error(f"[GPU {accelerator.process_index}] DDP validation failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline is not None:
            del pipeline
        if transformer3d_val is not None:
            del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Restore original output_dir
    args.output_dir = original_output_dir
