import os
import sys
import time
import torch
import gradio as gr
import numpy as np
import imageio
from PIL import Image

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from videox_fun.ui.wan_ui import Wan_Controller, css
from videox_fun.ui.ui import (
    create_model_type, create_model_checkpoints, create_finetune_models_checkpoints,
    create_teacache_params, create_cfg_skip_params, create_cfg_riflex_k,
    create_prompts, create_samplers, create_height_width,
    create_generation_methods_and_video_length, create_generation_method,
    create_cfg_and_seedbox, create_ui_outputs
)
from videox_fun.data.dataset_image_video import derive_ground_object_from_instruction
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import save_videos_grid, timer

def load_video_frames(video_path: str, source_frames: int):
    assert source_frames is not None, "source_frames is required"
    
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)

    stride = max(1, total_frames // source_frames)
    # Using random start frame as in inference.py
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

    input_video = torch.from_numpy(np.array(frames))
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    input_video = input_video * (2.0 / 255.0) - 1.0

    return input_video, original_height, original_width

class VideoCoF_Controller(Wan_Controller):
    @timer
    def generate(
        self,
        diffusion_transformer_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        validation_video_mask,
        control_video,
        denoise_strength,
        seed_textbox,
        ref_image=None,
        enable_teacache=None, 
        teacache_threshold=None, 
        num_skip_start_steps=None, 
        teacache_offload=None, 
        cfg_skip_ratio=None,
        enable_riflex=None, 
        riflex_k=None,
        # Custom args
        source_frames_slider=33,
        reasoning_frames_slider=4,
        repeat_rope_checkbox=True,
        fps=16,
        is_api=False,
    ):
        self.clear_cache()
        print(f"VideoCoF Generation started.")

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            self.update_lora_model(lora_model_dropdown)

        # Scheduler setup
        scheduler_config = self.pipeline.scheduler.config
        if sampler_dropdown in ["Flow_Unipc", "Flow_DPM++"]:
            scheduler_config['shift'] = 1
        self.pipeline.scheduler = self.scheduler_dict[sampler_dropdown].from_config(scheduler_config)

        # LoRA merging
        if self.lora_model_path != "none":
            print(f"Merge Lora.")
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        # Seed
        if int(seed_textbox) != -1 and seed_textbox != "": 
            torch.manual_seed(int(seed_textbox))
        else: 
            seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device=self.device).manual_seed(int(seed_textbox))

        try:
            # VideoCoF logic
            # We expect a video input. Using validation_video or control_video field from UI.
            # In Wan UI, 'Video to Video' usually sets control_video. 
            # If validation_video is used in other modes, we should check which one is active.
            # For simplicity, let's use `validation_video` if available (usually from "Video to Video" or "Image to Video" with video input?)
            # Actually, standard UI "Video to Video" updates `control_video`.
            
            input_video_path = control_video if control_video is not None else validation_video
            
            if input_video_path is None:
                raise ValueError("Please upload a video for VideoCoF generation.")

            # CoT Prompt Construction
            edit_text = prompt_textbox
            ground_instr = derive_ground_object_from_instruction(edit_text)
            prompt = (
                "A video sequence showing three parts: first the original scene, "
                f"then grounded {ground_instr}, and finally the same scene but {edit_text}"
            )
            print(f"Constructed prompt: {prompt}")

            # Load video frames
            input_video_tensor, video_height, video_width = load_video_frames(
                input_video_path,
                source_frames=source_frames_slider
            )

            # Adjust dimensions if needed (standard UI sliders vs loaded video)
            # inference.py uses loaded video dims. We should probably use them too, or resize?
            # inference.py uses video_height/width passed to pipeline.
            # If user set custom width/height, we might want to respect that?
            # inference.py passes video_height, video_width.
            
            # Using loaded video dimensions
            h, w = video_height, video_width

            print(f"Running pipeline with frames={length_slider}, source={source_frames_slider}, reasoning={reasoning_frames_slider}")
            
            sample = self.pipeline(
                video=input_video_tensor,
                prompt=prompt,
                num_frames=length_slider,
                source_frames=source_frames_slider,
                reasoning_frames=reasoning_frames_slider,
                negative_prompt=negative_prompt_textbox,
                height=h,
                width=w,
                generator=generator,
                guidance_scale=cfg_scale_slider,
                num_inference_steps=sample_step_slider,
                # shift is handled in scheduler config or inference.py passed it explicitly?
                # inference.py passed shift=3 but standard UI sets scheduler shift=1 for Flow_Unipc?
                # let's trust the scheduler config we just set.
                repeat_rope=repeat_rope_checkbox,
                cot=True,
            ).videos

            # Process output
            # sample contains combined video.
            # inference.py extracts edit_video: sample[:, :, -source_frames:, :, :]
            # But wait, WanPipelineOutput has .videos, .ground_videos, .edit_videos if cot=True
            # But the pipeline returns WanPipelineOutput object, so sample is that object?
            # In inference.py: sample = pipeline(...).videos
            # In my code above: sample = self.pipeline(...).videos
            
            # If pipeline implementation returns WanPipelineOutput, accessing .videos gives the full tensor.
            # inference.py logic:
            # edit_video = sample[:, :, -args.source_frames:, :, :]
            
            # We want to show the full CoT process or just the result?
            # User probably wants the result, maybe the full video too.
            # Let's return the full video for now.
            
            final_video = sample

        except Exception as e:
            print(f"Error: {e}")
            if self.lora_model_path != "none":
                 self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
            return gr.update(), gr.update(), f"Error: {str(e)}"

        # Unmerge LoRA
        if self.lora_model_path != "none":
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)

        # Save output
        save_sample_path = self.save_outputs(
            False, length_slider, final_video, fps=fps
        )
        
        return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"

def ui(GPU_memory_mode, scheduler_dict, config_path, compile_dit, weight_dtype):
    controller = VideoCoF_Controller(
        GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
        config_path=config_path, compile_dit=compile_dit,
        weight_dtype=weight_dtype
    )

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# VideoCoF Demo")
        
        with gr.Column(variant="panel"):
            diffusion_transformer_dropdown, _ = create_model_checkpoints(controller, visible=True)
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider, _ = create_finetune_models_checkpoints(controller, visible=True)
            
            with gr.Row():
                enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload = create_teacache_params(True, 0.10, 1, False)
                cfg_skip_ratio = create_cfg_skip_params(0)
                enable_riflex, riflex_k = create_cfg_riflex_k(False, 6)

        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts()
            
            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)
                    
                    # Custom VideoCoF Params
                    with gr.Group():
                        gr.Markdown("### VideoCoF Parameters")
                        source_frames_slider = gr.Slider(label="Source Frames", minimum=1, maximum=100, value=33, step=1)
                        reasoning_frames_slider = gr.Slider(label="Reasoning Frames", minimum=1, maximum=20, value=4, step=1)
                        repeat_rope_checkbox = gr.Checkbox(label="Repeat RoPE", value=True)
                        
                    resize_method, width_slider, height_slider, base_resolution = create_height_width(
                        default_height=480, default_width=832
                    )
                    
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation"], 
                            default_video_length=65, 
                            maximum_video_length=161
                        )
                    
                    # Simplified input for VideoCoF - mainly Video to Video
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Video to Video (视频到视频)"], prompt_textbox, support_end_image=False
                    )
                    
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(True)
                    generate_button = gr.Button(value="Generate", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

        # Event handlers
        generate_button.click(
            fn=controller.generate,
            inputs=[
                diffusion_transformer_dropdown,
                base_model_dropdown,
                lora_model_dropdown, 
                lora_alpha_slider,
                prompt_textbox, 
                negative_prompt_textbox, 
                sampler_dropdown, 
                sample_step_slider, 
                resize_method,
                width_slider, 
                height_slider, 
                base_resolution, 
                generation_method, 
                length_slider, 
                overlap_video_length, 
                partial_video_length, 
                cfg_scale_slider, 
                start_image, 
                end_image, 
                validation_video,
                validation_video_mask,
                control_video,
                denoise_strength, 
                seed_textbox,
                ref_image, 
                enable_teacache, 
                teacache_threshold, 
                num_skip_start_steps, 
                teacache_offload, 
                cfg_skip_ratio,
                enable_riflex, 
                riflex_k,
                # New inputs
                source_frames_slider,
                reasoning_frames_slider,
                repeat_rope_checkbox
            ],
            outputs=[result_image, result_video, infer_progress]
        )

    return demo, controller

if __name__ == "__main__":
    from videox_fun.ui.controller import flow_scheduler_dict
    
    GPU_memory_mode = "sequential_cpu_offload"
    compile_dit = False
    weight_dtype = torch.bfloat16
    server_name = "0.0.0.0"
    server_port = 7860
    config_path = "config/wan2.1/wan_civitai.yaml"

    demo, controller = ui(GPU_memory_mode, flow_scheduler_dict, config_path, compile_dit, weight_dtype)
    
    demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True
    )
    
    while True:
        time.sleep(5)

