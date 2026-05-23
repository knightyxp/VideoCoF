import json
import os
import base64
import argparse
from typing import Dict, Any, List, Optional, Union, Tuple
import io
from PIL import Image
import imageio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

try:
    import cv2  # Optional fallback for video decoding
except Exception:
    cv2 = None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Video Edit Evaluation (GPT-based scoring, streaming via relay)')
    parser.add_argument('--input_json', required=True, type=str, help='Path to input JSON (same structure as GPT/Gemini versions)')
    parser.add_argument('--output_json', required=True, type=str, help='Path to write results JSON')
    parser.add_argument('--video_root', required=True, type=str, help='Root directory containing ORIGINAL videos')
    parser.add_argument('--edited_video_root', type=str, default=None, help='Optional root directory containing EDITED videos (defaults to --video_root)')
    parser.add_argument('--api_key', required=True, help='OpenAI-compatible API key')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model name (e.g., gpt-4o)')
    parser.add_argument('--api_base', default='https://api.openai.com/v1', type=str, help='OpenAI-compatible API base URL')
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames to sample from each video (default: 3)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads for processing')
    parser.add_argument('--print_stream', action='store_true', help='Print streaming content while receiving')
    parser.add_argument('--edited_video_pattern', type=str, default='gen_{task_type}_{sample_id}.mp4', help='Format string to derive edited video filename, e.g., "gen_{task_type}_{sample_id}.mp4"')
    parser.add_argument('--original_from_compare_left_half', action='store_true', help='Force using left half of gen_{task_type}_{sample_id}_compare.mp4 as original (fallback to input if compare missing)')
    return parser.parse_args()


def get_config(args):
    return {
        "input_json": args.input_json,
        "output_json": args.output_json,
        "original_video_root": args.video_root,
        "edited_video_root": args.edited_video_root or args.video_root,
        "api_key": args.api_key,
        "model": args.model,
        "api_base": args.api_base,
        "frames_per_video": max(1, int(args.num_frames)),
        "num_workers": args.num_workers,
        "print_stream": bool(args.print_stream),
        "edited_video_pattern": args.edited_video_pattern,
        "original_from_compare_left_half": bool(args.original_from_compare_left_half),
    }


def load_and_normalize_samples(input_json_path: str) -> List[Dict[str, Any]]:
    with open(input_json_path, "r", encoding="utf-8") as f:
        samples: Union[List[Any], Dict[str, Any]] = json.load(f)

    if isinstance(samples, dict):
        if 'results' in samples and isinstance(samples['results'], list):
            return samples['results']
        converted: List[Dict[str, Any]] = []
        for sid, item in samples.items():
            if isinstance(item, dict):
                converted.append({"id": sid, **item})
        if converted:
            return converted
        raise ValueError("Unsupported JSON structure: dict without 'results' or id->sample mapping")
    elif isinstance(samples, list):
        return samples
    else:
        raise ValueError("Unsupported JSON structure: expected list or dict")


ORIGINAL_VIDEO_KEYS: List[str] = [
    "original_video_path",
    "source_video_path",
    "source_video",
    "src_video",
    "input_video",
    "video_path",
    "original_video",
]

EDITED_VIDEO_KEYS: List[str] = [
    "edited_video_path",
    "edited_video",
    "output_video",
    "result_video",
    "generated_video",
    "target_video",
    "target_video_path",
    "edited_path",
    "generated_video_path",
]

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

EVALUATION_PROMPT_TEXT: str = """# **Role**
You are an evaluator for instructional video editing tasks. Your job is to assess how well the edited video fulfills the user's specific instructions.
# **Input**
1. The user's instruction
2. The original video (first video)
3. The edited video (second video)
# **Task**
Please evaluate the instruct editing score:
- Instruct follow: Does the edit precisely follow the given instruction?
- Quality: Is the edit result video visually seamless and natural-looking?
- Preservation: Does the edit maintain coherence with the original video context?
Scoring rules:
Instruct follow score: 1-3: Edit does not follow the instruction. 4-6: Edit follows the instruction partially. 7-10: Edit follows the instruction fully.
Quality score: 1-3: Edit result video is not visually seamless, not natural-looking and not aesthetics. 4-6: Edit result video is visually seamless partially, natural-looking partially, and aesthetics partially. 7-10: Edit result video is visually seamless fully, natural-looking fully, and aesthetics fully.
Preservation score: 1-3: Edit result video does not maintain coherence with the original video context. 4-6: Edit result video maintains coherence with the original video context partially. 7-10: Edit result video maintains coherence with the original video context fully.
Using the following Output format:
# **Output**
Structure the output in JSON format with:
- instruction: Repeat the user's instruction.
- instruct follow score (1-10): Your score number
- quality score (1-10): Your score number
- preservation score (1-10): Your score number
- reason: The reasons for the score you gave
"""


def _build_pattern_values(sample: Dict[str, Any]) -> Dict[str, str]:
    sample_id = str(
        sample.get("sample_id")
        or sample.get("id")
        or sample.get("video_id")
        or ""
    ).strip()
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


def extract_frames_by_indices(video_path: str, indices: List[int], crop_left_half: bool = False) -> List[str]:
    """
    Extract specific frames by absolute indices and return them as base64 JPEGs.
    Attempts imageio first, then falls back to OpenCV if available.
    Skips indices that cannot be read.
    """
    frames_b64: List[str] = []

    # Try imageio first
    try:
        reader = imageio.get_reader(video_path)
        try:
            # Try to get total frames if available (may be -1 or raise in some containers)
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
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                buffer.seek(0)
                frames_b64.append(base64.b64encode(buffer.read()).decode('utf-8'))
        finally:
            reader.close()
        if frames_b64:
            return frames_b64
    except Exception:
        pass
        
    return frames_b64


def _get_video_length(video_path: str) -> Optional[int]:
    """
    Attempt to obtain the total number of frames in a video.
    Returns None if unavailable.
    """
    try:
        reader = imageio.get_reader(video_path)
        try:
            length = reader.get_length()
        finally:
            reader.close()
        if isinstance(length, (int, float)):
            if length == float("inf"):
                length = None
            elif length > 0:
                return int(length)
    except Exception:
        pass

    if cv2 is not None:
        try:
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


def extract_evenly_spaced_frames(video_path: str, num_frames: int, crop_left_half: bool = False) -> List[str]:
    """
    Extract up to num_frames frames evenly spaced throughout the video.
    Attempts to gather additional frames if evenly spaced sampling fails.
    """
    if num_frames <= 0:
        return []

    total_frames = _get_video_length(video_path)
    primary_indices = _compute_evenly_spaced_indices(total_frames, num_frames)
    frames = extract_frames_by_indices(video_path, primary_indices, crop_left_half=crop_left_half)
    if len(frames) >= num_frames:
        return frames[:num_frames]

    seen = set(primary_indices)
    if total_frames is not None:
        fallback_pool = [idx for idx in range(total_frames) if idx not in seen]
    else:
        fallback_limit = max(num_frames * 4, (len(primary_indices) or 1) * 4)
        fallback_pool = [idx for idx in range(fallback_limit) if idx not in seen]

    if fallback_pool:
        extra_frames = extract_frames_by_indices(video_path, fallback_pool, crop_left_half=crop_left_half)
        for b64 in extra_frames:
            if len(frames) >= num_frames:
                break
            frames.append(b64)

    return frames[:num_frames]


def construct_standard_paths(
    sample: Dict[str, Any],
    root_dir: str,
    edited_pattern: Optional[str],
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Construct original/edited video paths based solely on task_type and sample_id.
    Original video: gen_{task_type}_{sample_id}_input.mp4
      - Fallback: gen_{task_type}_{sample_id}_compare.mp4 (use left half)
    Edited video: uses edited_pattern (default: gen_{task_type}_{sample_id}.mp4)
    Returns (original_path, edited_path, original_needs_left_crop)
    """
    values = _build_pattern_values(sample)
    task_type = values.get("task_type") or values.get("task_type_lower") or ""
    sample_id = values.get("sample_id") or values.get("id") or ""
    if not task_type or not sample_id:
        return (None, None, False)

    base_name = f"gen_{task_type}_{sample_id}"

    # Original input candidate
    original_input_rel = f"{base_name}_input.mp4"
    original_input_abs = resolve_video_path(root_dir, original_input_rel)
    if original_input_abs:
        original_path = original_input_abs
        original_crop_left = False
    else:
        # Fallback to compare video (left half)
        compare_rel = f"{base_name}_compare.mp4"
        compare_abs = resolve_video_path(root_dir, compare_rel)
        original_path = compare_abs
        original_crop_left = bool(compare_abs)

    # Edited path from pattern or default
    if edited_pattern:
        try:
            edited_rel = edited_pattern.format(**values)
        except KeyError:
            edited_rel = f"{base_name}.mp4"
    else:
        edited_rel = f"{base_name}.mp4"
    edited_path = resolve_video_path(root_dir, edited_rel)

    return (original_path, edited_path, original_crop_left)


def build_evaluation_messages(instruction: str, original_frames: List[str], edited_frames: List[str]) -> List[Dict[str, Any]]:
    user_instruction = (instruction or "").strip()
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": EVALUATION_PROMPT_TEXT.strip(),
        },
        {
            "type": "text",
            "text": f"User instruction:\n{user_instruction or 'N/A'}",
        },
        {
            "type": "text",
            "text": "Original video frames (chronological order):",
        },
    ]

    if original_frames:
        for b64 in original_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                },
            })
    else:
        content.append({
            "type": "text",
            "text": "[No original frames available]",
        })

    content.append({
        "type": "text",
        "text": "Edited video frames (chronological order):",
    })

    if edited_frames:
        for b64 in edited_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                },
            })
    else:
        content.append({
            "type": "text",
            "text": "[No edited frames available]",
        })

    content.append({
        "type": "text",
        "text": "Return only the JSON object described above. Do not add commentary.",
    })

    system_content = (
        "You are a precise video editing evaluation assistant. "
        "Follow the scoring rubric exactly and respond with a single JSON object."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content},
    ]


def parse_evaluation_response(response_text: str) -> Dict[str, Any]:
    if not response_text:
        return {}

    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    objects: List[str] = []
    depth = 0
    start: Optional[int] = None
    for idx, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start = idx
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(cleaned[start:idx + 1])
                    start = None

    for candidate in objects:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def stream_chat_completion(url: str, api_key: str, payload: Dict[str, Any], print_stream: bool, timeout: Optional[int] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Call relay chat/completions with stream=True, accumulate content, optionally print stream.
    Returns (full_text, usage_dict_or_none).
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    # Ensure streaming
    payload = dict(payload)
    payload["stream"] = True

    full_text_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            if line.startswith('data: '):
                line = line[6:]
            if line == '[DONE]':
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Standard OpenAI-like stream chunk
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    full_text_parts.append(content)
                    if print_stream:
                        print(content, end='', flush=True)
            # Some relays may append usage in the final event
            if "usage" in chunk and isinstance(chunk["usage"], dict):
                usage = chunk["usage"]
    if print_stream:
        print()
    return ("".join(full_text_parts), usage)


def evaluate_sample(
    original_video_path: str,
    edited_video_path: str,
    instruction: str,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    original_crop_left_half: bool = False,
) -> Optional[Dict[str, Any]]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or sample.get("video_id") or "unknown")
    original_name = os.path.basename(original_video_path) if original_video_path else ""
    edited_name = os.path.basename(edited_video_path) if edited_video_path else ""

    try:
        frames_per_video = max(1, int(cfg.get("frames_per_video", 3)))
        original_frames = [
            f for f in extract_evenly_spaced_frames(original_video_path, frames_per_video, crop_left_half=original_crop_left_half) if f
        ]
        edited_frames = [
            f for f in extract_evenly_spaced_frames(edited_video_path, frames_per_video, crop_left_half=False) if f
        ]

        if not original_frames:
            print(f"[ERROR] No frames extracted from original video '{original_name}' (sample_id={sample_id})")
            return None
        if not edited_frames:
            print(f"[ERROR] No frames extracted from edited video '{edited_name}' (sample_id={sample_id})")
            return None

        messages = build_evaluation_messages(instruction, original_frames, edited_frames)

        url = f"{cfg['api_base'].rstrip('/')}/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.1,
        }
        response_text, usage = stream_chat_completion(
            url=url,
            api_key=cfg["api_key"],
            payload=payload,
            print_stream=bool(cfg.get("print_stream", False)),
        )

        if not response_text or not response_text.strip():
            print(f"[ERROR] Empty response for sample_id={sample_id}")
            return None

        evaluation = parse_evaluation_response(response_text)
        if not evaluation:
            print(f"[ERROR] Failed to parse JSON response for sample_id={sample_id}")
            return None

        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "instruction": instruction,
            "original_video_path": original_video_path,
            "edited_video_path": edited_video_path,
            "original_from_compare_left_half": bool(original_crop_left_half),
            "original_video_name": original_name,
            "edited_video_name": edited_name,
            "frames_per_video_requested": frames_per_video,
            "original_frames_sampled": len(original_frames),
            "edited_frames_sampled": len(edited_frames),
            "evaluation": evaluation,
            "raw_response": response_text.strip(),
            "model": cfg["model"],
        }

        if len(original_frames) < frames_per_video or len(edited_frames) < frames_per_video:
            result["sampling_warning"] = (
                f"Requested {frames_per_video} frames; received {len(original_frames)} original and {len(edited_frames)} edited."
            )

        if usage is not None:
            result["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }

        return result
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] HTTP error while evaluating sample_id={sample_id}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to evaluate sample_id={sample_id}: {e}")
        return None


def resolve_video_path_from_keys(
    sample: Dict[str, Any],
    keys: List[str],
    root: Optional[str],
    pattern: Optional[str] = None,
) -> Optional[str]:
    formatted_path = None
    if pattern:
        try:
            formatted_path = pattern.format(**_build_pattern_values(sample))
        except KeyError as exc:
            print(
                f"Warning: Edited video pattern missing key {exc} for sample {sample.get('sample_id') or sample.get('id')}"
            )
            formatted_path = None

        if formatted_path:
            resolved = resolve_video_path(root, formatted_path)
            if resolved:
                return resolved

    for key in keys:
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            resolved = resolve_video_path(root, val)
            if resolved:
                return resolved

    return None


def load_existing_results_list(output_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return []
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get('results'), list):
        return data['results']
    return []


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Find the first number in the string
        match = re.search(r'[-+]?\d+(?:\.\d+)?', value.strip())
        if match:
            try:
                return float(match.group(0))
            except Exception:
                return None
    return None


def _get_by_aliases(d: Dict[str, Any], aliases: List[str]) -> Any:
    if not isinstance(d, dict):
        return None
    lower_map = {str(k).lower(): k for k in d.keys()}
    for alias in aliases:
        k = lower_map.get(alias.lower())
        if k is not None:
            return d[k]
    return None


def compute_average_scores(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    instruct_aliases = [
        "instruct follow score",
        "instruct_follow_score",
        "instruct follow score (1-10)",
    ]
    quality_aliases = [
        "quality score",
        "quality_score",
        "quality score (1-10)",
    ]
    preservation_aliases = [
        "preservation score",
        "preservation_score",
        "preservation score (1-10)",
    ]

    total_instruct = 0.0
    total_quality = 0.0
    total_preservation = 0.0
    count = 0

    for item in results:
        eval_obj = item.get("evaluation")
        if not isinstance(eval_obj, dict):
            continue
        v_instruct = _to_number(_get_by_aliases(eval_obj, instruct_aliases))
        v_quality = _to_number(_get_by_aliases(eval_obj, quality_aliases))
        v_preservation = _to_number(_get_by_aliases(eval_obj, preservation_aliases))
        if v_instruct is None or v_quality is None or v_preservation is None:
            continue
        total_instruct += v_instruct
        total_quality += v_quality
        total_preservation += v_preservation
        count += 1

    averages = {
        "num_results": len(results),
        "num_scored": count,
        "instruct_follow_avg": (total_instruct / count) if count else None,
        "quality_avg": (total_quality / count) if count else None,
        "preservation_avg": (total_preservation / count) if count else None,
    }
    return averages


def _canonical_task_category(task_type: Optional[str]) -> Optional[str]:
    if not isinstance(task_type, str):
        return None
    t = task_type.strip().lower()
    if not t:
        return None
    if t in {"grounding", "obj_removal","ID-Delete","id-delete", "id_delete"}:
        return "obj_removal"
    if t in {"obj_addition"}:
        return "obj_addition"
    if t in {"obj_swap", "obj-swap", "obj_swap_multi_instance", "obj-swap-multi-instance"}:
        return "obj_swap"
    if t in {"local_style_transfer", "local-style-transfer", "local_style", "local-style", "local_style-multi-instance", "local-style-multi-instance"}:
        return "local_style_transfer"
    return None


def compute_average_scores_by_task_type(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    instruct_aliases = [
        "instruct follow score",
        "instruct_follow_score",
        "instruct follow score (1-10)",
    ]
    quality_aliases = [
        "quality score",
        "quality_score",
        "quality score (1-10)",
    ]
    preservation_aliases = [
        "preservation score",
        "preservation_score",
        "preservation score (1-10)",
    ]

    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for item in results:
        task_type = item.get("task_type")
        category = _canonical_task_category(task_type)
        if category is None:
            continue
        eval_obj = item.get("evaluation")
        if not isinstance(eval_obj, dict):
            continue
        v_instruct = _to_number(_get_by_aliases(eval_obj, instruct_aliases))
        v_quality = _to_number(_get_by_aliases(eval_obj, quality_aliases))
        v_preservation = _to_number(_get_by_aliases(eval_obj, preservation_aliases))
        if v_instruct is None or v_quality is None or v_preservation is None:
            continue
        if category not in sums:
            sums[category] = {"instruct": 0.0, "quality": 0.0, "preservation": 0.0}
            counts[category] = 0
        sums[category]["instruct"] += v_instruct
        sums[category]["quality"] += v_quality
        sums[category]["preservation"] += v_preservation
        counts[category] += 1

    summary: Dict[str, Any] = {}
    for category, total_map in sums.items():
        count = counts.get(category, 0)
        if count <= 0:
            continue
        summary[category] = {
            "num_scored": count,
            "instruct_follow_avg": total_map["instruct"] / count,
            "quality_avg": total_map["quality"] / count,
            "preservation_avg": total_map["preservation"] / count,
        }
    return summary


def save_results_with_summary(results: List[Dict[str, Any]], output_path: str, averages: Dict[str, Any], averages_by_task_type: Dict[str, Any]):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    payload = {
        "results": results,
        "averages": averages,
        "averages_by_task_type": averages_by_task_type,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def resolve_first_existing(video_root: Optional[str], rel_paths: List[str]) -> Optional[str]:
    """
    Try multiple relative paths in order and return the first that exists under video_root.
    """
    for rel in rel_paths:
        resolved = resolve_video_path(video_root, rel)
        if resolved:
            return resolved
    return None


def _expand_task_type_variants(task_type_raw: str) -> List[str]:
    """
    Generate a list of candidate task_type strings for filename resolution, including
    multi-instance variants and hyphen/underscore alternatives.
    """
    variants: List[str] = []

    def _add(v: str):
        if v and v not in variants:
            variants.append(v)

    raw = task_type_raw or ""
    lower = raw.lower()
    hyphen = lower.replace("_", "-")
    underscore = lower.replace("-", "_")

    # Always consider raw (preserve case for cases like 'ID-Delete')
    _add(raw)
    # Common lowercase forms
    _add(lower)
    _add(hyphen)
    _add(underscore)

    # obj_swap family (include multi-instance)
    if "obj" in lower and "swap" in lower:
        _add("obj_swap")
        _add("obj-swap")
        _add("obj_swap_multi_instance")
        _add("obj-swap-multi-instance")

    # local style family (include transfer and multi-instance)
    if "local" in lower and "style" in lower:
        _add("local_style_transfer")
        _add("local-style-transfer")
        _add("local_style")
        _add("local-style")
        _add("local_style-multi-instance")
        _add("local-style-multi-instance")

    return variants


def resolve_instruction(sample: Dict[str, Any]) -> Optional[str]:
    """
    Try multiple common keys to find the edit instruction from sample.
    """
    for key in INSTRUCTION_KEYS:
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            nested_text = val.get("text")
            if isinstance(nested_text, str) and nested_text.strip():
                return nested_text.strip()
    return None


def main():
    args = parse_arguments()
    cfg = get_config(args)

    samples = load_and_normalize_samples(cfg["input_json"])
    print(f"Loaded {len(samples)} samples")

    output_path = cfg["output_json"]
    results: List[Dict[str, Any]] = []
    start_idx = 0
    if os.path.exists(output_path):
        try:
            existing = load_existing_results_list(output_path)
            if isinstance(existing, list) and existing:
                results = existing
                start_idx = len(results)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")

    print(f"Using {cfg['num_workers']} worker threads")

    def _first_non_empty(sample_obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            val = sample_obj.get(k)
            if isinstance(val, str) and val.strip():
                return val
        return None

    def _worker(i: int, sample: Dict[str, Any]):
        sample_identifier = sample.get("sample_id") or sample.get("id") or f"idx_{i}"

        instruction = resolve_instruction(sample)
        if not instruction:
            print(f"Warning: Instruction missing for sample index {i} ({sample_identifier})")
            return (i, None, True)

        # Construct paths purely from task_type and sample_id; never read paths from JSON
        values = _build_pattern_values(sample)
        task_type = values.get("task_type") or values.get("task_type_lower") or ""
        sample_id_val = values.get("sample_id") or values.get("id") or ""
        if not (task_type and sample_id_val):
            print(f"Warning: Missing task_type or sample_id for sample index {i} ({sample_identifier})")
            return (i, None, True)
        task_variants = _expand_task_type_variants(task_type)
        base_names: List[str] = [f"gen_{tv}_{sample_id_val}" for tv in task_variants]

        # Original: prefer input, fallback to compare (left half)
        force_compare = bool(cfg.get("original_from_compare_left_half", False))
        original_crop_left = False
        original_path = None
        tried_originals: List[str] = []
        if force_compare:
            # Try compare across all variants, then input across all variants
            for bn in base_names:
                rel = f"{bn}_compare.mp4"
                tried_originals.append(rel)
                candidate = resolve_video_path(cfg["original_video_root"], rel)
                if candidate:
                    original_path = candidate
                    original_crop_left = True
                    break
            if not original_path:
                for bn in base_names:
                    rel = f"{bn}_input.mp4"
                    tried_originals.append(rel)
                    candidate = resolve_video_path(cfg["original_video_root"], rel)
                    if candidate:
                        original_path = candidate
                        original_crop_left = False
                        break
        else:
            # Try input across all variants, then compare across all variants
            for bn in base_names:
                rel = f"{bn}_input.mp4"
                tried_originals.append(rel)
                candidate = resolve_video_path(cfg["original_video_root"], rel)
                if candidate:
                    original_path = candidate
                    original_crop_left = False
                    break
            if not original_path:
                for bn in base_names:
                    rel = f"{bn}_compare.mp4"
                    tried_originals.append(rel)
                    candidate = resolve_video_path(cfg["original_video_root"], rel)
                    if candidate:
                        original_path = candidate
                        original_crop_left = True
                        break
        if not original_path:
            tried_list = ", ".join(tried_originals)
            print(
                f"Warning: Original video not found for sample index {i} ({sample_identifier}) | "
                f"tried [{tried_list}] under root='{cfg['original_video_root']}'"
            )
            return (i, None, True)

        # Edited path strictly from pattern/default under edited root
        edited_pattern = cfg.get("edited_video_pattern")
        edited_candidates: List[str] = []
        # Pattern-based candidates for each task variant
        for tv in task_variants:
            vals = dict(values)
            vals["task_type"] = tv
            vals["task_type_lower"] = tv.lower()
            vals["task_type_clean"] = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tv)
            try:
                rel = edited_pattern.format(**vals) if edited_pattern else f"gen_{tv}_{sample_id_val}.mp4"
            except KeyError:
                rel = f"gen_{tv}_{sample_id_val}.mp4"
            if rel not in edited_candidates:
                edited_candidates.append(rel)
            # Also try _gen variant
            base_rel = f"gen_{tv}_{sample_id_val}.mp4"
            gen_rel = f"gen_{tv}_{sample_id_val}_gen.mp4"
            for alt in (gen_rel, base_rel):
                if alt not in edited_candidates:
                    edited_candidates.append(alt)
        edited_path = resolve_first_existing(cfg["edited_video_root"], edited_candidates)
        if not edited_path:
            tried_list = ", ".join(edited_candidates)
            print(
                f"Warning: Edited video not found for sample index {i} ({sample_identifier}) | "
                f"tried [{tried_list}] under root='{cfg['edited_video_root']}'"
            )
            return (i, None, True)

        result = evaluate_sample(original_path, edited_path, instruction, sample, cfg, original_crop_left_half=original_crop_left)
        if result:
            merged = {**sample, **result}
            return (i, merged, False)
        return (i, None, True)

    next_save_idx = start_idx
    done: Dict[int, Any] = {}

    with ThreadPoolExecutor(max_workers=cfg["num_workers"]) as executor:
        future_to_idx = {
            executor.submit(_worker, i, samples[i]): i
            for i in range(start_idx, len(samples))
        }

        total_to_process = len(samples) - start_idx
        with tqdm(total=total_to_process, desc="Evaluating", unit="sample", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                try:
                    i, merged, skip_flag = future.result()
                except Exception as e:
                    i = future_to_idx[future]
                    print(f"[ERROR] Worker failed for index {i}: {e}")
                    merged, skip_flag = None, True
                done[i] = (merged, skip_flag)

                while next_save_idx in done:
                    merged_res, is_skip = done.pop(next_save_idx)
                    if not is_skip and merged_res is not None:
                        results.append(merged_res)
                        try:
                            save_results(results, output_path)
                        except Exception as e:
                            print(f"[ERROR] Failed to save intermediate results: {e}")
                    next_save_idx += 1

                pbar.update(1)

    save_results(results, output_path)
    # Compute and print averages; then write final JSON with summary
    averages = compute_average_scores(results)
    averages_by_task_type = compute_average_scores_by_task_type(results)
    if averages.get("num_scored", 0):
        print(
            f"\nAverages over {averages['num_scored']} scored samples "
            f"(out of {averages['num_results']} total): "
            f"instruct_follow_avg={averages['instruct_follow_avg']:.3f}, "
            f"quality_avg={averages['quality_avg']:.3f}, "
            f"preservation_avg={averages['preservation_avg']:.3f}"
        )
    else:
        print(f"\nNo valid scored samples found among {averages['num_results']} results.")
    # Print per-task-type breakdown (fixed order)
    ordered_categories = ["obj_removal", "obj_addition", "obj_swap", "local_style_transfer"]
    print("\nPer-task-type averages:")
    for cat in ordered_categories:
        s = averages_by_task_type.get(cat)
        if not s:
            print(f"  {cat}: no scored samples")
            continue
        print(
            f"  {cat}: num_scored={s['num_scored']}, "
            f"instruct_follow_avg={s['instruct_follow_avg']:.3f}, "
            f"quality_avg={s['quality_avg']:.3f}, "
            f"preservation_avg={s['preservation_avg']:.3f}"
        )
    save_results_with_summary(results, output_path, averages, averages_by_task_type)
    print(f"\nProcessing complete! Total samples evaluated: {len(results)}")


if __name__ == "__main__":
    main()
