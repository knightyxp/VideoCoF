#!/usr/bin/env python
"""Compatibility entry for older VideoCoF/VideoX-Fun launch scripts.

The CoT training path is implemented in train_joint_img_video_lora.py and
enabled through command-line flags such as --use_cot_dataset.
"""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("train_joint_img_video_lora.py")),
    run_name="__main__",
)
