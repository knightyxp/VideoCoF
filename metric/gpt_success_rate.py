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

# EVALUATION_PROMPT_TEXT: str = """Prompt Template: Success Ratio
# You are an advanced vision–language model tasked with verifying video edits.
# You will be shown one frame from the original video and one frame from the edited video.
# The video-editing prompt for this case is: <instruct_prompt>
# Your task is to:
# 1. understand the content implied by the original frame,
# 2. examine the edited frame and determine whether the changes match the editing prompt,
# 3. ensure that no additional edits are introduced—only the modifications required by the prompt are allowed.
# You must reply with only a single lowercase word: yes or no. No explanations, punctuation, or extra text.
# """

EVALUATION_PROMPT_TEXT: str = """You are an advanced vision–language model tasked with verifying instance-level video edits in multi-instance scenes.
You will be shown one frame from the original video and one frame from the edited video.
The video-editing prompt for this case is: <instruct_prompt>

The edit prompt may target a specific instance using a natural-language descriptor and a position (for example: "the young woman wearing a hijab on the right").
Your job is to verify three things, using the two frames:
1) The targeted instance in the original frame must match the descriptor and position in the edit prompt (confirm the intended target is present and identifiable).
2) In the edited frame, that same targeted instance must have been modified exactly as the edit prompt requires (e.g., replaced, swapped, or restyled). If the described change was applied to a different person/object than the one specified, this is a failure.
3) No other people or objects in the scene may have been changed beyond minor, incidental pixel noise — the only allowed modification is the one specified for the targeted instance. Any additional edits (different person swapped, extra objects altered, or obvious new artifacts affecting other instances) make the case a failure.

Decide: does the edited frame correctly implement the prompt on the specified instance and only that instance?
Reply with exactly one lowercase word and nothing else: yes or no.
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


def build_evaluation_messages(instruction: str, original_frame: Optional[str], edited_frame: Optional[str]) -> List[Dict[str, Any]]:
    user_instruction = (instruction or "").strip()
    eval_prompt = EVALUATION_PROMPT_TEXT.replace("<instruct_prompt>", user_instruction)

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": eval_prompt.strip(),
        }
    ]

    if original_frame:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{original_frame}"
            },
        })
    else:
        content.append({
            "type": "text",
            "text": "[No original frame available]",
        })

    if edited_frame:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{edited_frame}"
            },
        })
    else:
        content.append({
            "type": "text",
            "text": "[No edited frame available]",
        })

    system_content = (
        "You are a precise video editing evaluation assistant. "
        "Follow the scoring rubric exactly and respond with a single lowercase token: yes or no."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content},
    ]


def parse_yes_no_response(response_text: str) -> Optional[bool]:
    """
    Attempt to interpret the model response as a boolean yes/no answer.
    Handles bare strings as well as simple JSON objects that wrap the answer.
    """
    if not response_text:
        return None

    text = response_text.strip()
    if not text:
        return None

    def _normalize(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"yes", "y"}:
                return True
            if v in {"no", "n"}:
                return False
        return None

    # Direct string response
    normalized = _normalize(text.strip("\"' "))
    if normalized is not None:
        return normalized

    # Try parsing as JSON
    parsed: Optional[Any] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        candidate_keys = [
            "response",
            "result",
            "success_ratio",
            "success",
            "answer",
            "value",
        ]
        for key in candidate_keys:
            if key in parsed:
                normalized = _normalize(parsed[key])
                if normalized is not None:
                    return normalized
        # Fall back to scanning all values
        for val in parsed.values():
            normalized = _normalize(val)
            if normalized is not None:
                return normalized

    # Fallback: look for standalone yes/no tokens in the text
    for match in re.finditer(r"\b(yes|no)\b", text.lower()):
        token = match.group(1)
        if token == "yes":
            return True
        if token == "no":
            return False

    return None


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
        original_frames = extract_frames_by_indices(original_video_path, [0], crop_left_half=original_crop_left_half)
        edited_frames = extract_frames_by_indices(edited_video_path, [0], crop_left_half=False)

        original_frame = original_frames[0] if original_frames else None
        edited_frame = edited_frames[0] if edited_frames else None

        if not original_frame:
            print(f"[ERROR] No frames extracted from original video '{original_name}' (sample_id={sample_id})")
            return None
        if not edited_frame:
            print(f"[ERROR] No frames extracted from edited video '{edited_name}' (sample_id={sample_id})")
            return None

        messages = build_evaluation_messages(instruction, original_frame, edited_frame)

        url = f"{cfg['api_base'].rstrip('/')}/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": 0.0,
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

        success = parse_yes_no_response(response_text)
        if success is None:
            print(f"[ERROR] Unrecognized yes/no response for sample_id={sample_id}: {response_text!r}")
            return None

        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "instruction": instruction,
            "task_type": sample.get("task_type"),
            "success": bool(success),
            "original_video_path": original_video_path,
            "edited_video_path": edited_video_path,
            "original_from_compare_left_half": bool(original_crop_left_half),
            "original_video_name": original_name,
            "edited_video_name": edited_name,
            "frame_indices_sampled": [0],
            "raw_response": response_text.strip(),
            "model": cfg["model"],
        }

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


def compute_success_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    evaluated = 0
    success = 0
    for item in results:
        val = item.get("success")
        if isinstance(val, bool):
            evaluated += 1
            if val:
                success += 1
    rate = (success / evaluated) if evaluated else None
    return {
        "num_results": total,
        "num_evaluated": evaluated,
        "num_success": success,
        "success_rate": rate,
    }


def _canonical_task_category(task_type: Optional[str]) -> Optional[str]:
    if not isinstance(task_type, str):
        return None
    t = task_type.strip().lower()
    if not t:
        return None
    if t in {"grounding", "obj_removal", "id-delete", "id_delete"}:
        return "obj_removal"
    if t in {"obj_addition"}:
        return "obj_addition"
    if t in {"obj_swap", "obj-swap", "obj_swap_multi_instance", "obj-swap-multi-instance"}:
        return "obj_swap"
    if t in {"local_style_transfer", "local-style-transfer", "local_style", "local-style", "local_style-multi-instance", "local-style-multi-instance"}:
        return "local_style_transfer"
    return None


def compute_success_summary_by_task_type(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    data: Dict[str, Dict[str, int]] = {}
    for item in results:
        category = _canonical_task_category(item.get("task_type"))
        success = item.get("success")
        if category is None or not isinstance(success, bool):
            continue
        if category not in data:
            data[category] = {"num_evaluated": 0, "num_success": 0}
        data[category]["num_evaluated"] += 1
        if success:
            data[category]["num_success"] += 1

    summary: Dict[str, Any] = {}
    for category, counts in data.items():
        evaluated = counts["num_evaluated"]
        success = counts["num_success"]
        rate = success / evaluated if evaluated else None
        summary[category] = {
            "num_evaluated": evaluated,
            "num_success": success,
            "success_rate": rate,
        }
    return summary


def save_results_with_summary(results: List[Dict[str, Any]], output_path: str, overall: Dict[str, Any], per_task: Dict[str, Any]):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    payload = {
        "results": results,
        "success_summary": overall,
        "success_summary_by_task_type": per_task,
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
        plain_bases: List[str] = []
        gen_bases: List[str] = []
        for tv in task_variants:
            plain = f"{tv}_{sample_id_val}"
            gen_name = f"gen_{tv}_{sample_id_val}"
            if plain not in plain_bases:
                plain_bases.append(plain)
            if gen_name not in gen_bases:
                gen_bases.append(gen_name)

        # Original: prefer input, fallback to compare (left half). Optionally force compare first.
        force_compare = bool(cfg.get("original_from_compare_left_half", False))
        original_candidates: List[Tuple[str, bool]] = []
        seen_original_rels: set = set()

        def _add_original_candidates(bases: List[str], suffix: str, crop_left: bool):
            for base in bases:
                rel = f"{base}{suffix}"
                if rel not in seen_original_rels:
                    original_candidates.append((rel, crop_left))
                    seen_original_rels.add(rel)

        if force_compare:
            _add_original_candidates(plain_bases, "_compare.mp4", True)
            _add_original_candidates(gen_bases, "_compare.mp4", True)
            _add_original_candidates(plain_bases, "_input.mp4", False)
            _add_original_candidates(gen_bases, "_input.mp4", False)
        else:
            _add_original_candidates(plain_bases, "_input.mp4", False)
            _add_original_candidates(gen_bases, "_input.mp4", False)
            _add_original_candidates(plain_bases, "_compare.mp4", True)
            _add_original_candidates(gen_bases, "_compare.mp4", True)

        original_path = None
        original_crop_left = False
        tried_originals: List[str] = []
        for rel, crop_flag in original_candidates:
            tried_originals.append(rel)
            candidate = resolve_video_path(cfg["original_video_root"], rel)
            if candidate:
                original_path = candidate
                original_crop_left = crop_flag
                break
        if not original_path:
            tried_list = ", ".join(tried_originals)
            print(
                f"Warning: Original video not found for sample index {i} ({sample_identifier}) | "
                f"tried [{tried_list}] under root='{cfg['original_video_root']}'"
            )
            return (i, None, True)

        # Edited path strictly from filename conventions under edited root
        edited_pattern = cfg.get("edited_video_pattern")
        edited_candidates: List[str] = []
        seen_edited: set = set()

        def _add_edited_candidate(rel: str):
            if rel and rel not in seen_edited:
                edited_candidates.append(rel)
                seen_edited.add(rel)

        for tv in task_variants:
            vals = dict(values)
            vals["task_type"] = tv
            vals["task_type_lower"] = tv.lower()
            vals["task_type_clean"] = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tv)
            if edited_pattern:
                try:
                    rel = edited_pattern.format(**vals)
                    _add_edited_candidate(rel)
                except KeyError:
                    pass

        for base in plain_bases + gen_bases:
            _add_edited_candidate(f"{base}.mp4")
            _add_edited_candidate(f"{base}_gen.mp4")
            _add_edited_candidate(f"{base}_result.mp4")
            _add_edited_candidate(f"{base}_edit.mp4")

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
    # Compute and print success rates; then write final JSON with summary
    overall = compute_success_summary(results)
    per_task = compute_success_summary_by_task_type(results)

    evaluated = overall.get("num_evaluated", 0)
    success = overall.get("num_success", 0)
    rate = overall.get("success_rate")
    if evaluated:
        rate_pct = rate * 100 if rate is not None else None
        if rate_pct is not None:
            print(
                f"\nSuccess rate: {success}/{evaluated} ({rate_pct:.2f}%) evaluated samples "
                f"out of {overall.get('num_results', len(results))} total"
            )
        else:
            print(
                f"\nSuccess rate: {success}/{evaluated} evaluated samples "
                f"out of {overall.get('num_results', len(results))} total"
            )
    else:
        print(f"\nNo successful evaluations among {overall.get('num_results', len(results))} results.")

    ordered_categories = ["obj_removal", "obj_addition", "obj_swap", "local_style_transfer"]
    print("\nPer-task-type success rates:")
    for cat in ordered_categories:
        stats = per_task.get(cat)
        if not stats:
            print(f"  {cat}: no evaluated samples")
            continue
        rate = stats.get("success_rate")
        rate_pct = rate * 100 if rate is not None else None
        if rate_pct is not None:
            print(
                f"  {cat}: {stats['num_success']}/{stats['num_evaluated']} success ({rate_pct:.2f}%)"
            )
        else:
            print(
                f"  {cat}: {stats['num_success']}/{stats['num_evaluated']} success"
            )

    save_results_with_summary(results, output_path, overall, per_task)
    print(f"\nProcessing complete! Total samples evaluated: {len(results)}")


if __name__ == "__main__":
    main()
