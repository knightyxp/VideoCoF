#!/usr/bin/env python3
"""Resolve the public benchmark JSONL for the inference and evaluation scripts."""

import argparse
import json
from pathlib import Path


TASKS = ("obj_removal", "obj_addition", "obj_swap", "local_style_transfer")


def prepare_manifest(bench_root, edited_root, edited_pattern="gen_{id}.mp4", check_outputs=False):
    bench_root = Path(bench_root).expanduser().resolve()
    edited_root = Path(edited_root).expanduser().resolve()
    records = []
    ids, output_paths = set(), set()
    with (bench_root / "videocof_edit.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError(f"Invalid or duplicate id on line {line_number}: {sample_id!r}")
            if "/" in sample_id or "\\" in sample_id or sample_id in (".", ".."):
                raise ValueError(f"Unsafe id on line {line_number}: {sample_id!r}")
            task = next((task for task in TASKS if sample_id.startswith(task + "_")), None)
            if task is None:
                raise ValueError(f"Unknown task in id: {sample_id}")
            suffix = sample_id[len(task) + 1:]
            if not suffix:
                raise ValueError(f"Missing sample identifier: {sample_id}")
            instruction = row["edit_instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing editing instruction: {sample_id}")
            video = (bench_root / row["video"]).resolve()
            if bench_root not in video.parents:
                raise ValueError(f"Video must be inside the benchmark directory: {sample_id}")
            if not video.is_file():
                raise FileNotFoundError(f"Missing input for {sample_id}: {video}")
            edited = (edited_root / edited_pattern.format(
                id=sample_id, task_type=task, sample_id=suffix
            )).resolve()
            if edited in output_paths:
                raise ValueError(f"Edited filename collision for {sample_id}: {edited}")
            if check_outputs and not edited.is_file():
                raise FileNotFoundError(f"Missing edited output for {sample_id}: {edited}")
            ids.add(sample_id)
            output_paths.add(edited)
            records.append({
                "id": sample_id,
                "sample_id": suffix,
                "task_type": task,
                "instruction": instruction,
                "source_video_path": str(video),
                "original_video_path": str(video),
                "edited_video_path": str(edited),
            })
    if not records:
        raise ValueError("The benchmark manifest is empty")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", required=True, help="Downloaded VideoCoF-Bench directory")
    parser.add_argument("--edited-root", required=True, help="Generated, edited-only video directory")
    parser.add_argument("--output-json", required=True, help="Resolved inference/evaluation manifest to write")
    parser.add_argument("--edited-pattern", default="gen_{id}.mp4",
                        help="Output filename template; supports {id}, {task_type}, {sample_id}")
    parser.add_argument("--check-outputs", action="store_true", help="Also require every edited video to exist")
    args = parser.parse_args()
    records = prepare_manifest(args.bench_root, args.edited_root, args.edited_pattern, args.check_outputs)
    output = Path(args.output_json).expanduser().resolve()
    source = Path(args.bench_root).expanduser().resolve() / "videocof_edit.jsonl"
    if output == source:
        parser.error("--output-json must not overwrite videocof_edit.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} samples to {output}")


if __name__ == "__main__":
    main()
