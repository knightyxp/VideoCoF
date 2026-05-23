import torch
import clip
from PIL import Image
from glob import glob
import numpy as np
from collections import defaultdict
# import openpyxl

from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def crop_read_image_path(image_path):
    origin_image = Image.open(image_path)
    w, h = origin_image.size
    if h > w:
        origin_image = origin_image.crop((0, h-w, w, h))
    return origin_image


def edit_success(image_path, source_prompt,target_prompt):
    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)

    text = clip.tokenize([source_prompt, target_prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #print("CLIP-T-traget:", logits_per_image[:,1])  
    # return probs[0,1] >= probs[0,0], image_features
    return probs[0,1] >= probs[0,0], logits_per_image[:,1], image_features

def edit_success_imagebind(image_path, source_prompt,target_prompt):
    from imagebind import data
    import torch
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    image = preprocess(crop_read_image_path(image_path)).unsqueeze(0).to(device)

    text = [source_prompt, target_prompt].to(device)

    inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image, device),
                }

    with torch.no_grad():
        embeddings = model(inputs)
        probs = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).cpu().numpy()
        image_features = embeddings[ModalityType.VISION]
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
        
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #print("Label probs:", probs)  
    return probs[0,1] >= probs[0,0], image_features


def folder_success(folder, source_prompt, target_prompt):
    # 首先检查jpg文件
    jpg_files = sorted(glob(folder+'/*.jpg'))
    # 如果jpg文件存在，则使用jpg_files，否则检查png文件
    if jpg_files:
        file_list = jpg_files
    else:
        # 只有当没有jpg文件时才检查png文件
        file_list = sorted(glob(folder+'/*.png'))
    normalized_feature_list = []
    CLIP_T_score = 0.0
    #print(file_list)
    count = 0.0
    for f_path in file_list:        
        success, logits_per_image, image_feature = edit_success(f_path, source_prompt,target_prompt)
        if success: count +=1.0
        normalized_feature_list.append(image_feature/torch.sqrt(torch.sum(image_feature**2, axis=1, keepdims=True)))
        CLIP_T_score = CLIP_T_score + logits_per_image
    frame_const_list = []
    frame_const_list_sum = 0.0
    for i in range(len(normalized_feature_list)-1):
        sim_i = torch.sum(normalized_feature_list[i]*normalized_feature_list[i+1], axis=1)
        frame_const_list.append( sim_i )
        frame_const_list_sum += sim_i
    frame_const_list_avg = frame_const_list_sum/(len(normalized_feature_list)-1)
    
    #print(f'average temporal frame consistency: {frame_const_list_avg}')

    return count/len(file_list), frame_const_list_sum/(len(normalized_feature_list)-1), CLIP_T_score/len(file_list), CLIP_T_score/len(file_list) * frame_const_list_sum/(len(normalized_feature_list)-1)

import os
import json
import argparse
import imageio
from typing import List, Dict, Any, Optional, Tuple, Callable

_DINO_COMPONENT_CACHE: Dict[Tuple[str, str], Tuple["torch.nn.Module", Callable[[Image.Image], torch.Tensor]]] = {}

# -----------------------------------------------------------------------------
# Video path resolution and instruction extraction (aligned with gpt_evaluation)
# -----------------------------------------------------------------------------
INSTRUCTION_KEYS: List[str] = [
    "instruction",
    "edit_instruction",
    "edit_prompt",
    "user_instruction",
    "user_prompt",
    "task_instruction",
    "original_instruction",
    "coarse_instruction",
    "instruction_text",
    "qwen_vl_72b_refined_instruction",
    "prompt",
]

CATEGORY_ORDER: List[str] = [
    "obj_removal",
    "obj_addition",
    "obj_swap",
    "local_style_transfer",
]

CATEGORY_LABELS: Dict[str, str] = {
    "obj_removal": "obj_removal (grounding, id-delete)",
    "obj_addition": "obj_addition",
    "obj_swap": "obj_swap",
    "local_style_transfer": "local_style_transfer",
}


def _canonical_task_category(task_type: Optional[str]) -> Optional[str]:
    if not isinstance(task_type, str):
        return None
    t = task_type.strip().lower()
    if not t:
        return None
    if t in {"grounding", "obj_removal", "obj-removal", "id-delete", "id_delete"}:
        return "obj_removal"
    if t in {"obj_addition", "obj-addition"}:
        return "obj_addition"
    if t in {
        "obj_swap",
        "obj-swap",
        "obj_swap_multi_instance",
        "obj-swap-multi-instance",
    }:
        return "obj_swap"
    if t in {
        "local_style_transfer",
        "local-style-transfer",
        "local_style",
        "local-style",
        "local_style-multi-instance",
        "local-style-multi-instance",
    }:
        return "local_style_transfer"
    return None

def _build_pattern_values(sample: Dict[str, Any]) -> Dict[str, str]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or sample.get("video_id") or "").strip()
    sample_id_clean = sample_id.replace("\\", "_").replace("/", "_")
    digits = "".join(ch for ch in sample_id if ch.isdigit())

    task_type_raw = str(sample.get("task_type") or "").strip()
    task_type_clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_type_raw)

    values: Dict[str, str] = {
        "sample_id": sample_id,
        "sample_id_clean": sample_id_clean,
        "sample_id_digits": digits,
        "task_type": task_type_raw,
        "task_type_lower": task_type_raw.lower(),
        "task_type_clean": task_type_clean,
        "id": str(sample.get("id") or ""),
    }
    if digits:
        values.setdefault("sample_id_zfill3", digits.zfill(3))
        values.setdefault("sample_id_zfill4", digits.zfill(4))
        values.setdefault("sample_id_zfill5", digits.zfill(5))
    return values

def resolve_instruction(sample: Dict[str, Any]) -> Optional[str]:
    for key in INSTRUCTION_KEYS:
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            nested = val.get("text")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None

def resolve_video_path(video_root: Optional[str], rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    rp = str(rel_path).strip()
    if not rp:
        return None
    rp = os.path.expanduser(rp)
    if os.path.isabs(rp) and os.path.exists(rp):
        return rp
    if video_root:
        candidate = os.path.join(video_root, rp.lstrip("./"))
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(rp):
        return os.path.abspath(rp)
    return None

def construct_paths_for_sample(sample: Dict[str, Any], original_root: str, edited_root: str, edited_pattern: str, force_compare_left: bool = False) -> Tuple[Optional[str], Optional[str], bool]:
    values = _build_pattern_values(sample)
    task_type = values.get("task_type") or values.get("task_type_lower") or ""
    sample_id = values.get("sample_id") or values.get("id") or ""
    if not task_type or not sample_id:
        return (None, None, False)
    base_name = f"gen_{task_type}_{sample_id}"
    original_input_rel = f"{base_name}_input.mp4"
    compare_rel = f"{base_name}_compare.mp4"
    original_path: Optional[str] = None
    original_crop_left = False
    if force_compare_left:
        compare_abs = resolve_video_path(original_root, compare_rel)
        if compare_abs:
            original_path = compare_abs
            original_crop_left = True
        else:
            original_path = resolve_video_path(original_root, original_input_rel)
    else:
        original_path = resolve_video_path(original_root, original_input_rel)
        if not original_path:
            compare_abs = resolve_video_path(original_root, compare_rel)
            if compare_abs:
                original_path = compare_abs
                original_crop_left = True
    try:
        edited_rel = edited_pattern.format(**values) if edited_pattern else f"{base_name}.mp4"
    except KeyError:
        edited_rel = f"{base_name}.mp4"
    edited_path = resolve_video_path(edited_root, edited_rel)
    return (original_path, edited_path, original_crop_left)

# -----------------------------------------------------------------------------
# Frame extraction (PIL) aligned with gpt_evaluation spacing logic
# -----------------------------------------------------------------------------
def _get_video_length(video_path: str) -> Optional[int]:
    try:
        reader = imageio.get_reader(video_path)
        try:
            length = reader.get_length()
        finally:
            reader.close()
        if isinstance(length, (int, float)):
            if length == float("inf"):
                return None
            if length > 0:
                return int(length)
    except Exception:
        pass
    try:
        import cv2  # optional
        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                length_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if length_cv > 0:
                    return length_cv
        finally:
            cap.release()
    except Exception:
        pass
    return None

def _compute_evenly_spaced_indices(total_frames: Optional[int], num_frames: int) -> List[int]:
    if num_frames <= 0:
        return []
    if total_frames is None or total_frames <= 0:
        return list(range(num_frames))
    if num_frames == 1:
        return [0]
    step = (total_frames - 1) / (num_frames - 1)
    indices = [min(int(round(step * i)), total_frames - 1) for i in range(num_frames)]
    for i in range(1, len(indices)):
        if indices[i] < indices[i - 1]:
            indices[i] = indices[i - 1]
    return indices

def extract_frames_by_indices_pil(video_path: str, indices: List[int], crop_left_half: bool = False) -> List[Image.Image]:
    frames: List[Image.Image] = []
    try:
        reader = imageio.get_reader(video_path)
        try:
            try:
                total = reader.get_length()
            except Exception:
                total = None
            for idx in indices:
                if total is not None and (idx < 0 or idx >= total):
                    continue
                try:
                    frame = reader.get_data(idx)
                except Exception:
                    continue
                pil_image = Image.fromarray(frame)
                if crop_left_half:
                    width, height = pil_image.size
                    pil_image = pil_image.crop((0, 0, max(1, width // 2), height))
                frames.append(pil_image)
        finally:
            reader.close()
        if frames:
            return frames
    except Exception:
        pass
    return frames

def extract_evenly_spaced_frames_pil(video_path: str, num_frames: int, crop_left_half: bool = False) -> List[Image.Image]:
    if num_frames <= 0:
        return []
    total_frames = _get_video_length(video_path)
    primary_indices = _compute_evenly_spaced_indices(total_frames, num_frames)
    frames = extract_frames_by_indices_pil(video_path, primary_indices, crop_left_half=crop_left_half)
    if len(frames) >= num_frames:
        return frames[:num_frames]
    seen = set(primary_indices)
    if total_frames is not None:
        fallback_pool = [idx for idx in range(total_frames) if idx not in seen]
    else:
        fallback_limit = max(num_frames * 4, (len(primary_indices) or 1) * 4)
        fallback_pool = [idx for idx in range(fallback_limit) if idx not in seen]
    if fallback_pool:
        extra_frames = extract_frames_by_indices_pil(video_path, fallback_pool, crop_left_half=crop_left_half)
        for img in extra_frames:
            if len(frames) >= num_frames:
                break
            frames.append(img)
    return frames[:num_frames]

# -----------------------------------------------------------------------------
# Metric computation using CLIP (instruction as text)
# -----------------------------------------------------------------------------
def compute_clip_temporal_q(edited_video_path: str, instruction: str, frames_per_video: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        frames = extract_evenly_spaced_frames_pil(edited_video_path, frames_per_video, crop_left_half=False)
        if not frames:
            return (None, None, None)
        text_tokens = clip.tokenize([instruction]).to(device)
        normalized_feature_list = []
        clip_t_logits_sum = None
        with torch.no_grad():
            for pil_img in frames:
                image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                image_features = model.encode_image(image_tensor)
                logits_per_image, _ = model(image_tensor, text_tokens)
                if clip_t_logits_sum is None:
                    clip_t_logits_sum = logits_per_image[:, 0]
                else:
                    clip_t_logits_sum = clip_t_logits_sum + logits_per_image[:, 0]
                normalized = image_features / torch.sqrt(torch.sum(image_features ** 2, dim=1, keepdim=True))
                normalized_feature_list.append(normalized)
        # Temporal consistency (adjacent cosine similarities)
        if len(normalized_feature_list) >= 2:
            frame_const_sum = torch.zeros(1, device=normalized_feature_list[0].device)
            for i in range(len(normalized_feature_list) - 1):
                sim_i = torch.sum(normalized_feature_list[i] * normalized_feature_list[i + 1], dim=1)
                frame_const_sum += sim_i
            clip_f_temporal_consistency = frame_const_sum / (len(normalized_feature_list) - 1)
        else:
            clip_f_temporal_consistency = torch.tensor([1.0], device=device)
        clip_t_avg = clip_t_logits_sum / max(1, len(frames))
        q_edit = clip_t_avg * clip_f_temporal_consistency
        return (
            clip_t_avg.detach().cpu().numpy().item(),
            clip_f_temporal_consistency.detach().cpu().numpy().item(),
            q_edit.detach().cpu().numpy().item(),
        )
    except Exception:
        return (None, None, None)

# -----------------------------------------------------------------------------
# Frame-wise DINO consistency
# -----------------------------------------------------------------------------
def _resolve_torch_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        try:
            return torch.device(device_str)
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dino_transform(transforms_obj: Any) -> Callable[[Image.Image], torch.Tensor]:
    if isinstance(transforms_obj, dict):
        for key in ("eval", "val", "test"):
            transform = transforms_obj.get(key)
            if transform is not None:
                return transform
        for transform in transforms_obj.values():
            if transform is not None:
                return transform
    return transforms_obj


_DINO_DEFAULT_IMAGE_SIZES: Dict[str, int] = {
    "dinov2_vits14": 518,
    "dinov2_vitb14": 518,
    "dinov2_vitl14": 518,
    "dinov2_vitg14": 518,
    "dinov2_vits14_reg": 224,
    "dinov2_vitb14_reg": 224,
}


def _resolve_default_dino_image_size(model_name: str) -> int:
    for key, size in _DINO_DEFAULT_IMAGE_SIZES.items():
        if model_name.lower().startswith(key):
            return size
    return 518


def _build_default_dino_transform(model_name: str) -> Callable[[Image.Image], torch.Tensor]:
    image_size = _resolve_default_dino_image_size(model_name)
    return tv_transforms.Compose(
        [
            tv_transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            tv_transforms.CenterCrop(image_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _get_dino_components(model_name: str, device_str: str) -> Tuple["torch.nn.Module", Callable[[Image.Image], torch.Tensor]]:
    target_device = _resolve_torch_device(device_str)
    cache_key = (model_name, str(target_device))
    if cache_key in _DINO_COMPONENT_CACHE:
        return _DINO_COMPONENT_CACHE[cache_key]
    try:
        dino_model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as exc:
        raise RuntimeError(f"Unable to load DINOv2 model '{model_name}': {exc}") from exc
    dino_model.eval()
    dino_model.to(target_device)
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None
    try:
        dino_transforms = torch.hub.load("facebookresearch/dinov2", "dinov2_transforms")
    except Exception as exc:
        print(f"Warning: unable to load DINOv2 transforms from torch.hub: {exc}. Falling back to default preprocessing.")
        dino_transforms = None
    if dino_transforms is not None:
        transform = _select_dino_transform(dino_transforms)
    if not callable(transform):
        transform = _build_default_dino_transform(model_name)
    _DINO_COMPONENT_CACHE[cache_key] = (dino_model, transform)
    return _DINO_COMPONENT_CACHE[cache_key]


def _extract_dino_feature_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        priority_keys = (
            "x_norm_clstoken",
            "class_token",
            "pooled_cls_token",
            "global_output",
            "pooler_output",
            "last_hidden_state",
        )
        for key in priority_keys:
            tensor = output.get(key)
            if isinstance(tensor, torch.Tensor):
                if key == "last_hidden_state" and tensor.dim() >= 2:
                    return tensor[:, 0]
                return tensor
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _extract_dino_feature_tensor(item)
            if tensor is not None:
                return tensor
    return None


def compute_dino_temporal_consistency(
    edited_video_path: str,
    num_frames: int,
    model_name: str,
    device_str: Optional[str] = None,
) -> Optional[float]:
    target_num_frames = max(2, int(num_frames)) if num_frames else 2
    frames = extract_evenly_spaced_frames_pil(edited_video_path, target_num_frames, crop_left_half=False)
    if not frames:
        return None
    try:
        dino_model, dino_transform = _get_dino_components(model_name, device_str or "")
    except Exception as exc:
        print(f"Warning: failed to prepare DINO components for '{model_name}': {exc}")
        return None
    target_device = next(dino_model.parameters()).device
    normalized_features: List[torch.Tensor] = []
    try:
        with torch.no_grad():
            for pil_img in frames:
                image_rgb = pil_img.convert("RGB")
                tensor = dino_transform(image_rgb)
                if isinstance(tensor, (list, tuple)):
                    tensor = tensor[0]
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(target_device, non_blocking=True)
                output = dino_model(tensor)
                feat = _extract_dino_feature_tensor(output)
                if feat is None:
                    continue
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                elif feat.dim() > 2:
                    feat = feat.view(feat.shape[0], -1)
                feat = feat.to(target_device)
                feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-6)
                normalized_features.append(feat)
    except Exception as exc:
        print(f"Warning: error while computing DINO features for '{edited_video_path}': {exc}")
        return None
    if len(normalized_features) < 2:
        return None
    cosine_values: List[torch.Tensor] = []
    for first, second in zip(normalized_features[:-1], normalized_features[1:]):
        cosine_values.append(torch.sum(first * second, dim=1))
    if not cosine_values:
        return None
    stacked = torch.cat(cosine_values)
    temporal_consistency = stacked.mean()
    return float(temporal_consistency.detach().cpu().numpy())

# -----------------------------------------------------------------------------
# IO helpers and main
# -----------------------------------------------------------------------------
def load_and_normalize_samples(input_json_path: str) -> List[Dict[str, Any]]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], list):
            return data['results']
        converted: List[Dict[str, Any]] = []
        for sid, item in data.items():
            if isinstance(item, dict):
                converted.append({"id": sid, **item})
        if converted:
            return converted
    raise ValueError("Unsupported JSON structure: expected list or dict with 'results' or id->sample mapping")

def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP-T, temporal consistency, and Q-edit over edited videos (instruction as text). Paths resolved like gpt_evaluation.py")
    parser.add_argument("--input_json", required=True, type=str, help="Path to input JSON")
    parser.add_argument("--video_root", required=True, type=str, help="Root containing original videos (for path construction)")
    parser.add_argument("--edited_video_root", type=str, default=None, help="Root containing edited videos (default: --video_root)")
    parser.add_argument("--edited_video_pattern", type=str, default="gen_{task_type}_{sample_id}.mp4", help="Edited filename pattern, e.g., gen_{task_type}_{sample_id}.mp4")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to sample per video")
    parser.add_argument("--original_from_compare_left_half", action="store_true", help="Force using left half of *_compare.mp4 as original when *_input.mp4 missing")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save per-sample metrics JSON")
    parser.add_argument("--dino_model_name", type=str, default=None, help="Optional DINOv2 model name for frame-wise consistency (e.g., dinov2_vits14)")
    return parser.parse_args()

def main():
    args = parse_args()
    frames_per_video = max(1, int(args.num_frames))
    dino_model_name = args.dino_model_name.strip() if args.dino_model_name and args.dino_model_name.strip() else None
    cfg = {
        "input_json": args.input_json,
        "original_video_root": args.video_root,
        "edited_video_root": args.edited_video_root or args.video_root,
        "edited_video_pattern": args.edited_video_pattern,
        "frames_per_video": frames_per_video,
        "force_compare_left": bool(args.original_from_compare_left_half),
        "output_json": args.output_json,
        "dino_model_name": dino_model_name,
    }
    samples = load_and_normalize_samples(cfg["input_json"])
    clip_t_list: List[float] = []
    clip_f_list: List[float] = []
    q_list: List[float] = []
    dino_temp_list: List[float] = []
    results: List[Dict[str, Any]] = []
    category_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"clip_t": [], "clip_f": [], "q_edit": [], "dino_temporal_consistency": []}
    )
    for i, sample in enumerate(samples):
        instruction = resolve_instruction(sample) or ""
        orig_path, edited_path, _crop_left = construct_paths_for_sample(
            sample=sample,
            original_root=cfg["original_video_root"],
            edited_root=cfg["edited_video_root"],
            edited_pattern=cfg["edited_video_pattern"],
            force_compare_left=cfg["force_compare_left"],
        )
        if not edited_path:
            sid = sample.get("sample_id") or sample.get("id") or f"idx_{i}"
            print(f"Warning: edited video not found for sample {sid}")
            continue
        clip_t, clip_f, q_edit = compute_clip_temporal_q(edited_path, instruction, cfg["frames_per_video"])
        print(f"video id, instruction, {sample.get('sample_id') or sample.get('id')}, {instruction}")
        print(f"clip_t {clip_t}")
        print(f"clip_f (CLIP-F) {clip_f}")
        print(f"q_edit {q_edit}")
        dino_score = None
        if cfg["dino_model_name"]:
            dino_score = compute_dino_temporal_consistency(
                edited_path,
                cfg["frames_per_video"],
                cfg["dino_model_name"],
                device,
            )
            print(f"dino_temporal_consistency {dino_score}")
        print()
        if clip_t is not None and clip_f is not None and q_edit is not None:
            clip_t_list.append(clip_t)
            clip_f_list.append(clip_f)
            q_list.append(q_edit)
            result_entry: Dict[str, Any] = {
                **sample,
                "instruction": instruction,
                "edited_video_path": edited_path,
                "clip_t": clip_t,
                "clip_f": clip_f,
                "q_edit": q_edit,
            }
            if cfg["dino_model_name"] and dino_score is not None:
                result_entry["dino_temporal_consistency"] = dino_score
            results.append(result_entry)
            category = _canonical_task_category(sample.get("task_type"))
            if category:
                metrics_entry = category_metrics[category]
                metrics_entry["clip_t"].append(clip_t)
                metrics_entry["clip_f"].append(clip_f)
                metrics_entry["q_edit"].append(q_edit)
                if cfg["dino_model_name"] and dino_score is not None:
                    metrics_entry["dino_temporal_consistency"].append(dino_score)
        if cfg["dino_model_name"] and dino_score is not None:
            dino_temp_list.append(dino_score)
    dataset_summary: Optional[Dict[str, Any]] = None
    if clip_t_list:
        dataset_average_clip_t = float(np.array(clip_t_list).mean())
        dataset_average_clip_f = float(np.array(clip_f_list).mean())
        dataset_average_q = float(np.array(q_list).mean())
        dataset_summary = {
            "num_samples": len(clip_t_list),
            "clip_t_avg": dataset_average_clip_t,
            "clip_f_avg": dataset_average_clip_f,
            "q_edit_avg": dataset_average_q,
        }
        print("clip_t list :")
        print(clip_t_list)
        print("clip_f list (CLIP-F) :")
        print(clip_f_list)
        print("q_edit list :")
        print(q_list)
        print(f"dataset_average_clip_t {dataset_average_clip_t}")
        print(f"dataset_average_clip_f (CLIP-F) {dataset_average_clip_f}")
        print(f"dataset_average_q_edit {dataset_average_q}")
    else:
        print("No valid CLIP metrics computed.")
    if cfg["dino_model_name"]:
        if dino_temp_list:
            dataset_average_dino_temp = float(np.array(dino_temp_list).mean())
            print("dino_temporal_consistency list :")
            print(dino_temp_list)
            print(f"dataset_average_dino_temporal_consistency {dataset_average_dino_temp}")
            if dataset_summary is None:
                dataset_summary = {}
            dataset_summary["dino_temporal_consistency_avg"] = dataset_average_dino_temp
            dataset_summary["dino_num_samples"] = len(dino_temp_list)
        else:
            print("No valid DINO temporal consistency computed.")

    averages_by_category: Dict[str, Dict[str, Any]] = {}
    for category, metric_lists in category_metrics.items():
        count = len(metric_lists["clip_t"])
        if count <= 0:
            continue
        category_entry = {
            "num_scored": count,
            "clip_t_avg": float(np.array(metric_lists["clip_t"]).mean()),
            "clip_f_avg": float(np.array(metric_lists["clip_f"]).mean()),
            "q_edit_avg": float(np.array(metric_lists["q_edit"]).mean()),
        }
        if metric_lists["dino_temporal_consistency"]:
            category_entry["dino_temporal_consistency_avg"] = float(np.array(metric_lists["dino_temporal_consistency"]).mean())
        averages_by_category[category] = category_entry
    if dataset_summary:
        overall_entry = {
            "num_scored": dataset_summary["num_samples"],
            "clip_t_avg": dataset_summary["clip_t_avg"],
            "clip_f_avg": dataset_summary["clip_f_avg"],
            "q_edit_avg": dataset_summary["q_edit_avg"],
        }
        if "dino_temporal_consistency_avg" in dataset_summary:
            overall_entry["dino_temporal_consistency_avg"] = dataset_summary["dino_temporal_consistency_avg"]
        averages_by_category["overall"] = overall_entry
    if averages_by_category:
        print("\nPer-task-type averages:")
        for cat in CATEGORY_ORDER:
            label = CATEGORY_LABELS.get(cat, cat)
            stats = averages_by_category.get(cat)
            if not stats:
                print(f"  {label}: no scored samples")
                continue
            line = (
                f"  {label}: num_scored={stats['num_scored']}, "
                f"clip_t_avg={stats['clip_t_avg']:.6f}, "
                f"clip_f_avg={stats['clip_f_avg']:.6f} (CLIP-F), "
                f"q_edit_avg={stats['q_edit_avg']:.6f}"
            )
            if "dino_temporal_consistency_avg" in stats:
                line += f", dino_temporal_consistency_avg={stats['dino_temporal_consistency_avg']:.6f}"
            print(line)
        extra_categories = [
            c for c in averages_by_category.keys() if c not in CATEGORY_ORDER and c != "overall"
        ]
        for cat in sorted(extra_categories):
            label = CATEGORY_LABELS.get(cat, cat)
            stats = averages_by_category[cat]
            line = (
                f"  {label}: num_scored={stats['num_scored']}, "
                f"clip_t_avg={stats['clip_t_avg']:.6f}, "
                f"clip_f_avg={stats['clip_f_avg']:.6f} (CLIP-F), "
                f"q_edit_avg={stats['q_edit_avg']:.6f}"
            )
            if "dino_temporal_consistency_avg" in stats:
                line += f", dino_temporal_consistency_avg={stats['dino_temporal_consistency_avg']:.6f}"
            print(line)
        if "overall" in averages_by_category:
            stats = averages_by_category["overall"]
            line = (
                f"  Overall (all cases): num_scored={stats['num_scored']}, "
                f"clip_t_avg={stats['clip_t_avg']:.6f}, "
                f"clip_f_avg={stats['clip_f_avg']:.6f} (CLIP-F), "
                f"q_edit_avg={stats['q_edit_avg']:.6f}"
            )
            if "dino_temporal_consistency_avg" in stats:
                line += f", dino_temporal_consistency_avg={stats['dino_temporal_consistency_avg']:.6f}"
            print(line)

    if cfg["output_json"]:
        os.makedirs(os.path.dirname(cfg["output_json"]) or ".", exist_ok=True)
        payload: Dict[str, Any] = {"results": results}
        if dataset_summary:
            payload["averages"] = dataset_summary
        if averages_by_category:
            payload["averages_by_task_type"] = averages_by_category
        with open(cfg["output_json"], "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()