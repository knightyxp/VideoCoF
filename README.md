<div align="center">

  <h1 style="margin: 0; font-size: 1.8em;">
    Unified Video Editing with Temporal Reasoner
  </h1>

  <h4 style="margin: 15px 0; color: #2c3e50;">
    👁️ See &rarr; 🧠 Reason &rarr; ✏️ Edit
  </h4>

  <h4 style="margin: 15px 0; color: #2c3e50;">
    🚀 A Chain of Frames video editing method enbale temporal reasoning and 4x video length extrapolation with just 50k training pairs!
  </h4>

  [![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/2400.00000)
  [![arXiv](https://img.shields.io/badge/arXiv-2400.00000-b31b1b.svg)](https://arxiv.org/abs/2400.00000)
  [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://videocof.github.io)
  [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/XiangpengYang/VideoCoF)
  [![Hugging Face Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow)](https://huggingface.co/datasets)

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

https://github.com/user-attachments/assets/0e3eafae-3a62-4cd8-bf4a-37d9b280d05d

## 🔥 News

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

    *   **Wan-2.1-T2V-14B Pretrained Weights:**
        
        Using `git lfs`:
        ```bash
        git lfs install
        git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
        ```

        Or using `huggingface-cli`:
        ```bash
        hf download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B
        ```

    *   **VideoCoF Checkpoint:**
        
        Using `git lfs`:
        ```bash
        gif lfs install
        git clone https://huggingface.co/XiangpengYang/VideoCoF videocof_weight
        ```

        Or using `huggingface-cli`:
        ```bash
        hf download XiangpengYang/VideoCoF --local-dir videocof_weight
        ```

4.  **Inference:**

    For single inference tasks:

    ```bash
    # Multi-Instance Object Removal
    sh scripts/obj_rem.sh

    # Object Addition
    sh scripts/obj_add.sh

    # Local Style Transfer
    sh scripts/local_style.sh
    ```

    For parallel inference:

    ```bash
    sh scripts/parallel_infer.sh
    ```

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
- [ ] Release training code.
- [ ] Release training data.
- [ ] Add Hugging Face demo.

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
  journal={arXiv preprint arXiv:2400.00000},
  year={2025}
}
```

<div align="center">
  ⭐ **If you find this project helpful, please consider giving it a star!** ⭐
</div>
