<div align="center">

  <h1 style="margin: 0; font-size: 2.4em;">
    Unified Video Editing with Temporal Reasoner
  </h1>

  <h4 style="margin: 15px 0; color: #2c3e50;">
    👁️ See &rarr; 🧠 Reason &rarr; ✏️ Edit
  </h4>

  <h4 style="margin: 15px 0; color: #2c3e50;">
    🚀 A Chain of Frames video editing method enbale temporal reasoning and 4x video length extrapolation with just 50k training pairs!
  </h4>

  [![Hugging Face Daily Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Daily%20Paper-yellow)](https://huggingface.co/papers/2512.07469)
  [![arXiv](https://img.shields.io/badge/arXiv-2512.07469-b31b1b.svg)](https://arxiv.org/abs/2512.07469)
  [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://videocof.github.io)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/XiangpengYang/VideoCoF)
  [![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/XiangpengYang/VideoCoF)

</div>

<div align="center">
  <b>
    <a href="https://scholar.google.com/citations?user=reiIeYMAAAAJ">Xiangpeng Yang</a><sup>1</sup>,
    <a href="https://horizonwind2004.github.io/">Ji Xie</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=OvfI_HMAAAAJ">Yiyuan Yang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=zfeWd6gAAAAJ">Yan Huang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=sCuACdkAAAAJ">Min Xu</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=sCuACdkAAAAJ">Qiang Wu</a><sup>1</sup>
  </b>
  <br>
  <span style="font-size: 1em; color: #555;"><sup>1</sup>University of Technology Sydney, <sup>2</sup>Zhejiang University</span>
</div>

<br>

## 💿 Introduction

https://github.com/user-attachments/assets/26f7d347-3d6c-43cf-9645-6eb5906f6ad6

## 🔥 News

- **2025.12.13**: 🚀 We released a **4-step fast inference script** (~20-30s per video) and launched the Hugging Face demo! Please try it at [Hugging Face Spaces](https://huggingface.co/spaces/XiangpengYang/VideoCoF).
- **2025.12.09**: Paper available on arXiv.
- **2025.12.08**: Release the inference code and videocof-50k weight.
- **2025.12.06**: 🔥 Project Page and README updated!


## 📑 Table of Contents

- [🔧 Quick Start](#-quick-start)
- [🏆 Model Zoo](#-model-zoo)
- [🍭 Results](#-results)
- [🎨 Edit Comparison](#-edit-comparison)
- [🚧 TODO](#-todo)
- [🙏 Acknowledgments](#-acknowledgments)
- [📜 License](#-license)
- [📮 Contact](#-contact)
- [📄 Citation](#-citation)

## 🔧 Quick Start

**Online Demo:** Try out our enhanced VideoCoF demo on Hugging Face Spaces [here](https://huggingface.co/spaces/XiangpengYang/VideoCoF)!

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/videocof/VideoCoF.git
    cd VideoCoF
    ```

2.  **Install dependencies:**

    ```bash
    # 1. Create and activate a conda environment
    conda create -n videocof python=3.10
    conda activate videocof

    # 2. Install PyTorch (Choose version compatible with your CUDA)
    # For standard GPUs (CUDA 12.1):
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    
    # For Hopper GPUs (e.g., H100/H800) requiring fast inference:
    # pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    
    # 3. Install other dependencies
    pip install -r requirements.txt
    ```

    **Note on Flash Attention:**
    We recommend using **FlashAttention-3** (currently beta) for optimal performance, especially on NVIDIA H100/H800 GPUs. 
    If you are using these GPUs, please follow the [official FlashAttention-3 installation guide](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release) after installing the compatible PyTorch version (e.g., PyTorch 2.8 + CUDA 12.8).


3.  **Download Models:**

    **Wan-2.1-T2V-14B Pretrained Weights:**
        
        ```bash
        git lfs install
        git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
        
        # Or using huggingface-cli:
        # hf download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B
        ```

    **VideoCoF Checkpoint & Acceleration LoRA:**
        
        ```bash
        git lfs install
        git clone https://huggingface.co/XiangpengYang/VideoCoF videocof_weight

        # Or using huggingface-cli:
        # hf download XiangpengYang/VideoCoF --local-dir videocof_weight
        
        # Download Acceleration LoRA (FusionX)
        wget -P videocof_weight https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors
        ```

4.  **Inference:**

    🚀We provide **Fast 4-step inference** (Default, ~30s/video on H100) using acceleration LoRA.

    For single inference tasks:

    ```bash
    # Object Removal
    sh scripts/obj_rem.sh

    # Object Addition
    sh scripts/obj_add.sh

    # Object Swap
    sh scripts/obj_swap.sh

    # Local Style Transfer
    sh scripts/local_style.sh
    ```

    For parallel inference:

    ```bash
    sh scripts/parallel_infer.sh
    ```

5.  **Gradio Demo:**

    Launch the Gradio interface for interactive testing:

    ```bash
    # Ensure Wan2.1-T2V-14B (model_name), videocof_weight and dmd lora are in the current directory or properly referenced
    python examples/app.py
    ```
    
    The demo supports fast inference (~30s per video) online.

## 🏆 Model Zoo

Our models are available on Hugging Face:

| Model Name | Description | Link |
|------------|-------------|------|
| VideoCoF-Base | Base model trained on 50k video pairs | [Hugging Face](https://huggingface.co/XiangpengYang/VideoCoF) |

## 🍭 Results

### Why We Need Reasoning Before Editing?
![](assets/motivation_v2.gif)

Current video editing methods typically follow two paths:
1.  **Expert models**: Rely on external masks for precision but sacrifice unification.
2.  **Unified in-context learning models**: Mask-free but often struggle with spatial accuracy due to the lack of explicit cues.

**VideoCoF** bridges this gap by predicting reasoning tokens before generating the target video tokens.

### Key Capabilities

1.  **Seeing, Reasoning, Editing**: VideoCoF adopts a "seeing, reasoning, editing" approach, ensuring edits are applied accurately to the intended targets.
2.  **Length Extrapolation**: Trained on only **50k** data (33 frames), VideoCoF demonstrates robust multi-shot editing and length generalization (e.g., 4&times; length extrapolation).
3.  **Diverse Editing Tasks**: Supports fine-grained (instance and part level, spatial aware) Object Removal, Object Addition, Object Swap, and Local Style Transfer.

### Gallery Highlights

> Please refer to our [Project Page](https://videocof.github.io) for the full gallery.

*   **Object Removal**: Remove people or objects based on text prompts.
*   **Object Addition**: Add elements like animals, objects, or people.
*   **Object Swap**: Change specific attributes or objects.
*   **Local Style Transfer**: Modify texture, materials or colors.

## 🚧 TODO

- [x] Release paper.
- [x] Release inference code and weights.
- [x] Release Hugging Face demo (~30s infer a video online), try it at [Hugging Face Spaces](https://huggingface.co/spaces/XiangpengYang/VideoCoF).
- [ ] Release videocof-50k training data.
- [ ] Release training code.


## 🙏 Acknowledgments

We thank the authors of related works and the open-source community [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) and [Wan](https://github.com/Wan-Video/Wan2.1) for their contributions.

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 📮 Contact

For any questions, please feel free to reach out to the author Xiangpeng Yang [@knightyxp](https://github.com/knightyxp), email: knightyxp@gmail.com/Xiangpeng.Yang@student.uts.edu.au

## 📄 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{yang2025videocof,
  title={Unified Video Editing with Temporal Reasoner},
  author={Yang, Xiangpeng and Xie, Ji and Yang, Yiyuan and Huang, Yan and Xu, Min and Wu, Qiang},
  journal={arXiv preprint arXiv:2512.07469},
  year={2025}
}
```

<div align="center">
  ⭐ **If you find this project helpful, please consider giving it a star!** ⭐
</div>

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=knightyxp/VideoCoF&type=Date&legend=top-left)](https://star-history.com/#knightyxp/VideoCoF&Date)