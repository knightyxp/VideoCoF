import json
import os
import base64
import argparse
import hashlib
import time
import sys
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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Video Edit Evaluation (GPT-based scoring, streaming via relay)')
    parser.add_argument('--input_json', required=True, type=str, help='Input JSON or JSONL evaluation manifest')
    parser.add_argument('--output_json', required=True, type=str, help='Path to write results JSON')
    parser.add_argument('--video_root', default='.', type=str, help='Root directory containing ORIGINAL videos')
    parser.add_argument('--edited_video_root', type=str, default=None, help='Optional root directory containing EDITED videos (defaults to --video_root)')
    parser.add_argument('--api_key', default=os.environ.get('OPENAI_API_KEY'), help='Defaults to OPENAI_API_KEY')
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-2024-05-13'), help='Judge model (historical snapshot by default)')
    parser.add_argument('--api_base', default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'), type=str, help='Defaults to OPENAI_BASE_URL')
    parser.add_argument('--request_timeout', type=float, default=120, help='HTTP connect/read timeout in seconds')
    parser.add_argument('--max_retries', type=int, default=0, help='Retries for transient HTTP failures (historical default: 0)')
    parser.add_argument('--num_frames', type=int, default=3, help='Number of frames to sample from each video (default: 3)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel threads for processing')
    parser.add_argument('--print_stream', action='store_true', help='Print each completed judge response')
    parser.add_argument('--edited_video_pattern', type=str, default='gen_{task_type}_{sample_id}.mp4', help='Format string to derive edited video filename, e.g., "gen_{task_type}_{sample_id}.mp4"')
    parser.add_argument('--original_from_compare_left_half', action='store_true', help='Force using left half of gen_{task_type}_{sample_id}_compare.mp4 as original (fallback to input if compare missing)')
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error('Set OPENAI_API_KEY or pass --api_key')
    if args.request_timeout <= 0 or args.max_retries < 0 or args.num_workers < 1 or args.num_frames < 1:
        parser.error('Timeout, workers and frames must be positive; retries cannot be negative')
    return args


def get_config(args):
    return {
        "input_json": args.input_json,
        "output_json": args.output_json,
        "original_video_root": args.video_root,
        "edited_video_root": args.edited_video_root or args.video_root,
        "api_key": args.api_key,
        "model": args.model,
        "api_base": args.api_base,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "frames_per_video": max(1, int(args.num_frames)),
        "num_workers": args.num_workers,
        "print_stream": bool(args.print_stream),
        "edited_video_pattern": args.edited_video_pattern,
        "original_from_compare_left_half": bool(args.original_from_compare_left_half),
    }


def load_and_normalize_samples(input_json_path: str) -> List[Dict[str, Any]]:
    with open(input_json_path, "r", encoding="utf-8-sig") as f:
        if str(input_json_path).lower().endswith('.jsonl'):
            samples = [json.loads(line) for line in f if line.strip()]
        else:
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
        if not all(isinstance(sample, dict) for sample in samples):
            raise ValueError('Every sample must be a JSON object')
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


def stream_chat_completion(url: str, api_key: str, payload: Dict[str, Any], print_stream: bool, timeout: Optional[float] = 120, max_retries: int = 0) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Read one complete SSE response, with bounded retries and redacted errors."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json; charset=utf-8"}
    payload = dict(payload, stream=True)
    for attempt in range(max_retries + 1):
        parts, usage, returned_model = [], None, None
        completed = False
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode('utf-8', errors='replace').strip()
                    if not line.startswith('data:'):
                        continue
                    line = line[5:].strip()
                    if line == '[DONE]':
                        completed = True
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get('error'):
                        raise RuntimeError('Judge API returned a streaming error')
                    if isinstance(chunk.get('model'), str):
                        returned_model = chunk['model']
                    choices = chunk.get('choices')
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        finish_reason = choice.get('finish_reason')
                        if finish_reason not in (None, 'stop'):
                            raise RuntimeError('Judge response did not finish normally')
                        completed = completed or finish_reason == 'stop'
                        content = (choice.get('delta') or {}).get('content')
                        if isinstance(content, str):
                            parts.append(content)
                    if isinstance(chunk.get('usage'), dict):
                        usage = chunk['usage']
            if not completed:
                raise RuntimeError('Judge stream ended before a completion marker')
            text = ''.join(parts)
            if print_stream:
                print(text, flush=True)
            return text, usage, returned_model
        except requests.RequestException as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            transient = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or status == 429 or (isinstance(status, int) and status >= 500)
            if transient and attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError('Judge API request failed' + (f' (HTTP {status})' if status else '')
                               + '; check credentials, model availability and connectivity') from None


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
        response_text, usage, returned_model = stream_chat_completion(
            url=url,
            api_key=cfg["api_key"],
            payload=payload,
            print_stream=bool(cfg.get("print_stream", False)),
            timeout=cfg.get("request_timeout", 120),
            max_retries=cfg.get("max_retries", 0),
        )

        if not response_text or not response_text.strip():
            print(f"[ERROR] Empty response for sample_id={sample_id}")
            return None

        evaluation = parse_evaluation_response(response_text)
        if not evaluation:
            print(f"[ERROR] Failed to parse JSON response for sample_id={sample_id}")
            return None

        if not compute_average_scores([{"evaluation": evaluation}])["num_scored"]:
            print(f"[ERROR] Missing or invalid 1-10 scores for sample_id={sample_id}")
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
            "judge_metadata": {
                "requested_model": cfg["model"], "returned_model": returned_model,
                "temperature": 0.1, "stream": True,
                "request_timeout": cfg.get("request_timeout", 120),
                "max_retries": cfg.get("max_retries", 0),
                "rubric_sha256": hashlib.sha256(EVALUATION_PROMPT_TEXT.encode()).hexdigest(),
            },
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
        print(f"[ERROR] HTTP error while evaluating sample_id={sample_id}: {type(e).__name__}")
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
    raise ValueError('Existing output must be a result list or an object containing results')


def save_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
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
        if any(value is None or not 1 <= value <= 10 for value in (v_instruct, v_quality, v_preservation)):
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
        if any(value is None or not 1 <= value <= 10 for value in (v_instruct, v_quality, v_preservation)):
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


def save_results_with_summary(results: List[Dict[str, Any]], output_path: str, averages: Dict[str, Any], averages_by_task_type: Dict[str, Any], total_samples: Optional[int] = None):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    payload = {
        "results": results,
        "averages": averages,
        "averages_by_task_type": averages_by_task_type,
    }
    payload['coverage'] = {'num_input_samples': total_samples, 'num_evaluated': len(results),
                           'num_missing_or_failed': total_samples - len(results) if total_samples is not None else None}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_video_path(video_root: Optional[str], rel_path: str) -> Optional[str]:
    if not rel_path:
        return None

    rp = str(rel_path).strip()
    if not rp:
        return None

    rp = os.path.expanduser(rp)
    if os.path.isabs(rp):
        return rp if os.path.isfile(rp) else None

    if video_root:
        candidate = os.path.join(video_root, rp)
        if os.path.isfile(candidate):
            return candidate

    if os.path.isfile(rp):
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


def resolve_sample_paths(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], bool]:
    """Explicit manifest paths are authoritative; legacy names remain a fallback."""
    def first(keys):
        return next((sample[key] for key in keys if isinstance(sample.get(key), str) and sample[key].strip()), None)
    original_ref, edited_ref = first(ORIGINAL_VIDEO_KEYS), first(EDITED_VIDEO_KEYS)
    original = resolve_video_path(cfg['original_video_root'], original_ref) if original_ref else None
    edited = resolve_video_path(cfg['edited_video_root'], edited_ref) if edited_ref else None
    crop = bool(sample.get('original_from_compare_left_half', False)) if original_ref else False
    if (original_ref and not original) or (edited_ref and not edited):
        return None, None, False
    if original and edited:
        return original, edited, crop
    values = _build_pattern_values(sample)
    task, sid = values['task_type'], values['sample_id']
    if not task or not sid:
        return None, None, False
    variants = _expand_task_type_variants(task)
    if not original:
        suffixes = [('_compare.mp4', True), ('_input.mp4', False)] if cfg.get('original_from_compare_left_half') else [('_input.mp4', False), ('_compare.mp4', True)]
        for suffix, needs_crop in suffixes:
            original = resolve_first_existing(cfg['original_video_root'], [f'gen_{task_variant}_{sid}{suffix}' for task_variant in variants])
            if original:
                crop = needs_crop
                break
    if not edited:
        candidates = []
        for task_variant in variants:
            pattern_values = dict(values, task_type=task_variant, task_type_lower=task_variant.lower(),
                                  task_type_clean=''.join(c if c.isalnum() or c in '-_' else '_' for c in task_variant))
            pattern = cfg.get('edited_video_pattern') or 'gen_{task_type}_{sample_id}.mp4'
            try:
                candidates.append(pattern.format(**pattern_values))
            except KeyError:
                pass
            candidates.extend([f'gen_{task_variant}_{sid}_gen.mp4', f'gen_{task_variant}_{sid}.mp4'])
        edited = resolve_first_existing(cfg['edited_video_root'], candidates)
    return original, edited, crop


def evaluation_key(sample: Dict[str, Any], cfg: Dict[str, Any], media_hashes: Optional[Dict[str, str]] = None) -> str:
    original, edited, crop = resolve_sample_paths(sample, cfg)
    # The caller may reuse hashes only for the duration of one evaluation run.
    # A fresh call without a cache always inspects the current file contents.
    hashes = media_hashes if media_hashes is not None else {}
    def video_digest(path):
        if path is None:
            return None
        path = os.path.abspath(path)
        if path not in hashes:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[path] = digest.hexdigest()
        return hashes[path]
    identity = {'task_type': sample.get('task_type'),
                'sample_id': str(sample.get('sample_id') or sample.get('id') or sample.get('video_id') or ''),
                'instruction': resolve_instruction(sample),
                'original_video_path': os.path.abspath(original) if original else None,
                'edited_video_path': os.path.abspath(edited) if edited else None,
                'original_video_sha256': video_digest(original),
                'edited_video_sha256': video_digest(edited),
                'api_base_sha256': hashlib.sha256(cfg['api_base'].rstrip('/').encode()).hexdigest(),
                'crop_left_half': crop, 'model': cfg['model'],
                'frames_per_video': cfg.get('frames_per_video', 1),
                'rubric_sha256': hashlib.sha256(EVALUATION_PROMPT_TEXT.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def main():
    args = parse_arguments()
    cfg = get_config(args)
    samples = load_and_normalize_samples(cfg['input_json'])
    media_hashes = {}
    keys = [evaluation_key(sample, cfg, media_hashes) for sample in samples]
    if len(set(keys)) != len(keys):
        raise ValueError('Manifest contains duplicate evaluation cases')
    output_path = cfg['output_json']
    existing = load_existing_results_list(output_path)
    by_key = {}
    for row in existing:
        key = row.get('evaluation_key') if isinstance(row, dict) else None
        if not key:
            raise ValueError('Cannot safely resume legacy output without evaluation_key; select a new --output_json')
        if key not in keys:
            raise ValueError('Existing output does not match this manifest/judge configuration; select a new --output_json')
        by_key[key] = row
    pending = [i for i, key in enumerate(keys) if key not in by_key]
    print(f'Loaded {len(samples)} samples; resuming {len(by_key)} completed cases; evaluating {len(pending)}')

    def _worker(i):
        sample = samples[i]
        instruction = resolve_instruction(sample)
        original, edited, crop = resolve_sample_paths(sample, cfg)
        if not instruction or not original or not edited:
            print(f'[ERROR] Sample index {i}: instruction or video pair is missing')
            return None
        result = evaluate_sample(original, edited, instruction, sample, cfg, original_crop_left_half=crop)
        return dict(sample, **result, evaluation_key=keys[i]) if result else None

    with ThreadPoolExecutor(max_workers=cfg['num_workers']) as executor:
        future_to_idx = {executor.submit(_worker, i): i for i in pending}
        for future in tqdm(as_completed(future_to_idx), total=len(pending), desc='Evaluating', unit='sample'):
            i = future_to_idx[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f'[ERROR] Sample index {i}: {type(exc).__name__}')
                row = None
            if row is not None:
                by_key[keys[i]] = row
                save_results([by_key[key] for key in keys if key in by_key], output_path)
    results = [by_key[key] for key in keys if key in by_key]
    print(f'Coverage: {len(results)}/{len(samples)} evaluated; {len(samples) - len(results)} missing/failed (excluded from scores)')
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
    save_results_with_summary(results, output_path, averages, averages_by_task_type, total_samples=len(samples))
    print(f"\nProcessing complete! Total samples evaluated: {len(results)}")
    return 0 if len(results) == len(samples) else 1


if __name__ == "__main__":
    sys.exit(main())
