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

# Try importing spaces, if not available, define a dummy decorator
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(duration=120):
            def decorator(func):
                return func
            return decorator

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

global_controller = None

@spaces.GPU(duration=300)
@timer
def generate_wrapper(*args):
    global global_controller
    return global_controller.generate(*args)

def create_height_width_english(default_height, default_width, maximum_height, maximum_width):
    resize_method = gr.Radio(
        ["Generate by", "Resize according to Reference"],
        value="Generate by",
        show_label=False,
        visible=False # Hide since we force input resolution
    )
    
    width_slider     = gr.Slider(label="Width", value=default_width, minimum=128, maximum=maximum_width, step=16, visible=False)
    height_slider    = gr.Slider(label="Height", value=default_height, minimum=128, maximum=maximum_height, step=16, visible=False)
    base_resolution  = gr.Radio(label="Base Resolution", value=512, choices=[512, 640, 768, 896, 960, 1024], visible=False)

    return resize_method, width_slider, height_slider, base_resolution

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
                print(f"Original video dimensions: {original_width}x{original_height}")
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

    assert len(frames) == source_frames, f"Loaded {len(frames)} frames, expected {source_frames}"
    print(f"Loaded {source_frames} source frames")

    input_video = torch.from_numpy(np.array(frames))
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    input_video = input_video * (2.0 / 255.0) - 1.0

    return input_video, original_height, original_width


def preload_models(controller, default_model_path, default_lora_name, acc_lora_path):
    """
    Preload base model and LoRAs before launching the app to avoid first-run latency.
    """
    controller.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure tracking flags exist
    if not hasattr(controller, "_active_lora_path"):
        controller._active_lora_path = None
    if not hasattr(controller, "_acc_lora_active"):
        controller._acc_lora_active = False

    try:
        print(f"[preload] Loading base model: {default_model_path}")
        controller.update_diffusion_transformer(default_model_path)
        # update_base_model expects files under Personalized_Model; skip if not present
        base_candidate = os.path.join(controller.personalized_model_dir, os.path.basename(default_model_path))
        if os.path.exists(base_candidate):
            controller.update_base_model(os.path.basename(base_candidate))
        else:
            print(f"[preload] Skip update_base_model (not found at {base_candidate})")

        print(f"[preload] Loading VideoCoF LoRA: {default_lora_name}")
        controller.update_lora_model(default_lora_name)
        if controller.lora_model_path and controller.lora_model_path != "none":
            controller.pipeline = merge_lora(
                controller.pipeline,
                controller.lora_model_path,
                multiplier=1.0,
                device=controller.device,
            )
            controller._active_lora_path = controller.lora_model_path

        if acc_lora_path and os.path.exists(acc_lora_path):
            print(f"[preload] Loading Acceleration LoRA: {acc_lora_path}")
            controller.pipeline = merge_lora(
                controller.pipeline, acc_lora_path, multiplier=1.0, device=controller.device
            )
            controller._acc_lora_active = True
        else:
            print(f"[preload] Acceleration LoRA not found at {acc_lora_path}")
    except Exception as e:
        print(f"[preload] Warning: preload failed: {e}")
    finally:
        torch.cuda.empty_cache()

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
        # Custom args
        source_frames_slider=33,
        reasoning_frames_slider=4,
        repeat_rope_checkbox=True,
        # New arg for acceleration
        enable_acceleration=True,
        fps=8,
        is_api=False,
    ):
        self.clear_cache()
        print(f"VideoCoF Generation started.")
        
        # Ensure model is on CUDA inside the zero-gpu decorated function
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Ensure pipeline modules are on the chosen device (avoid CPU ops)
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                self.pipeline.to(self.device)
        except Exception as move_e:
            print(f"Warning: failed to move pipeline to {self.device}: {move_e}")
        
        if self.diffusion_transformer_dropdown != diffusion_transformer_dropdown:
            self.update_diffusion_transformer(diffusion_transformer_dropdown)

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            self.update_lora_model(lora_model_dropdown)

        # Track whether LoRAs are already merged to avoid repeat merges/unmerges.
        if not hasattr(self, "_active_lora_path"):
            self._active_lora_path = None
        if not hasattr(self, "_acc_lora_active"):
            self._acc_lora_active = False

        # Scheduler setup
        scheduler_config = self.pipeline.scheduler.config
        if sampler_dropdown in ["Flow_Unipc", "Flow_DPM++"]:
            scheduler_config['shift'] = 1
        self.pipeline.scheduler = self.scheduler_dict[sampler_dropdown].from_config(scheduler_config)

        # LoRA merging
        # 1. Merge VideoCoF LoRA
        if self.lora_model_path != "none":
            # If a different LoRA was previously merged, unmerge it first.
            if self._active_lora_path and self._active_lora_path != self.lora_model_path:
                print(f"Unmerging previous VideoCoF LoRA: {self._active_lora_path}")
                self.pipeline = unmerge_lora(self.pipeline, self._active_lora_path, multiplier=lora_alpha_slider, device=self.device)
                self._active_lora_path = None

            if self._active_lora_path != self.lora_model_path:
                print(f"Merge VideoCoF LoRA: {self.lora_model_path}")
                self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider, device=self.device)
                self._active_lora_path = self.lora_model_path

        # 2. Merge Acceleration LoRA (FusionX) if enabled
        acc_lora_path = os.path.join(project_root, "videocof_weight", "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors")
        if enable_acceleration:
            if os.path.exists(acc_lora_path):
                if not self._acc_lora_active:
                    print(f"Merge Acceleration LoRA: {acc_lora_path}")
                    # FusionX LoRA generally uses multiplier 1.0
                    self.pipeline = merge_lora(self.pipeline, acc_lora_path, multiplier=1.0, device=self.device)
                    self._acc_lora_active = True
            else:
                print(f"Warning: Acceleration LoRA not found at {acc_lora_path}")
        else:
            # If it was previously merged but now disabled, unmerge once.
            if self._acc_lora_active and os.path.exists(acc_lora_path):
                print("Unmerging Acceleration LoRA (disabled)")
                self.pipeline = unmerge_lora(self.pipeline, acc_lora_path, multiplier=1.0, device=self.device)
                self._acc_lora_active = False

        # Seed
        if int(seed_textbox) != -1 and seed_textbox != "": 
            torch.manual_seed(int(seed_textbox))
        else: 
            seed_textbox = np.random.randint(0, 1e10)
        # Ensure generator is created on the same device as the pipeline's transformer
        gen_device = getattr(getattr(self, "pipeline", None), "transformer", None)
        gen_device = gen_device.device if gen_device is not None else self.device
        if gen_device.type == 'meta':
            gen_device = self.device
        generator = torch.Generator(device=gen_device).manual_seed(int(seed_textbox))

        try:
            # VideoCoF logic
            # Use validation_video as source if provided (UI standard for Video-to-Video)
            input_video_path = validation_video
            
            if input_video_path is None:
                # Fallback to control_video if set, but standard UI uses validation_video
                input_video_path = control_video

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

            # Using loaded video dimensions
            h, w = video_height, video_width
            print(f"Input video dimensions: {w}x{h}")

            print(f"Running pipeline with frames={length_slider}, source={source_frames_slider}, reasoning={reasoning_frames_slider}")
            shift = 3
            
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
                shift=shift,
                repeat_rope=repeat_rope_checkbox,
                cot=True,
            ).videos

            # Keep only the edited segment (drop reasoning/original parts)
            final_video = sample[:, :, -source_frames_slider:, :, :]

        except Exception as e:
            print(f"Error: {e}")
            # Unmerge in case of error (LIFO order)
            if self._acc_lora_active and os.path.exists(acc_lora_path):
                 print("Unmerging Acceleration LoRA (due to error)")
                 self.pipeline = unmerge_lora(self.pipeline, acc_lora_path, multiplier=1.0, device=self.device)
                 self._acc_lora_active = False
            
            if self._active_lora_path:
                 print("Unmerging VideoCoF LoRA (due to error)")
                 self.pipeline = unmerge_lora(self.pipeline, self._active_lora_path, multiplier=lora_alpha_slider, device=self.device)
                 self._active_lora_path = None
            return gr.update(), gr.update(), f"Error: {str(e)}"

        # Save output
        save_sample_path = self.save_outputs(
            False, source_frames_slider, final_video, fps=fps
        )
        
        return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"

def ui(GPU_memory_mode, scheduler_dict, config_path, compile_dit, weight_dtype):
    controller = VideoCoF_Controller(
        GPU_memory_mode, scheduler_dict, model_name=None, model_type="Inpaint", 
        config_path=config_path, compile_dit=compile_dit,
        weight_dtype=weight_dtype
    )
    global global_controller
    global_controller = controller

    with gr.Blocks() as demo:
        gr.Markdown("# VideoCoF Demo")
        
        with gr.Column(variant="panel"):
            # Hide model selection
            # Adapt this to local path
            local_model_dir = os.path.join(project_root, "Wan2.1-T2V-14B")
            diffusion_transformer_dropdown, _ = create_model_checkpoints(controller, visible=False, default_model=local_model_dir)
            
            # Local VideoCoF paths
            videocof_lora_path = os.path.join(project_root, "videocof_weight", "videocof.safetensors")
            
            # FusionX Acceleration LoRA
            acc_lora_path = os.path.join(project_root, "videocof_weight", "Wan2.1_Text_to_Video_14B_FusionX_LoRA.safetensors")

            base_model_dropdown, lora_model_dropdown, lora_alpha_slider, _ = create_finetune_models_checkpoints(
                controller, visible=False, default_lora=videocof_lora_path
            )
            
            # Set default LoRA alpha to 1.0 (matching inference.py)
            lora_alpha_slider.value = 1.0

        # Preload heavy weights and LoRAs before launching the UI to avoid first-run latency.
        preload_models(controller, local_model_dir, videocof_lora_path, acc_lora_path)

        with gr.Column(variant="panel"):
            prompt_textbox, negative_prompt_textbox = create_prompts(prompt="Remove the young man with short black hair wearing black shirt on the left.")
            
            with gr.Row():
                with gr.Column():
                    sampler_dropdown, sample_step_slider = create_samplers(controller)
                    
                    # Default steps lowered to 4 for acceleration
                    sample_step_slider.value = 4
                    
                    # Custom VideoCoF Params
                    with gr.Group():
                        gr.Markdown("### VideoCoF Parameters")
                        source_frames_slider = gr.Slider(label="Source Frames", minimum=1, maximum=100, value=33, step=1)
                        reasoning_frames_slider = gr.Slider(label="Reasoning Frames", minimum=1, maximum=20, value=4, step=1)
                        repeat_rope_checkbox = gr.Checkbox(label="Repeat RoPE", value=True)
                        # Add Acceleration Checkbox
                        enable_acceleration = gr.Checkbox(label="Enable 4-step Acceleration (FusionX LoRA)", value=True)
                        
                    # Use custom height/width creation to hide/customize
                    resize_method, width_slider, height_slider, base_resolution = create_height_width_english(
                        default_height=480, default_width=832, maximum_height=1344, maximum_width=1344
                    )
                    
                    # Default video length 65
                    generation_method, length_slider, overlap_video_length, partial_video_length = \
                        create_generation_methods_and_video_length(
                            ["Video Generation"], 
                            default_video_length=65, 
                            maximum_video_length=161
                        )
                    
                    # Simplified input for VideoCoF - mainly Video to Video.
                    image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image = create_generation_method(
                        ["Video to Video"],
                        prompt_textbox,
                        support_end_image=False,
                        default_video="assets/two_man.mp4",
                        video_examples=[
                            ["assets/two_man.mp4", "Remove the young man with short black hair wearing black shirt on the left."],
                            ["assets/three_people.mp4", "Remove the man with short dark hair wearing a gray suit on the right"],
                            ["assets/office.mp4", "Remove the beige CRT computer setup."],
                            ["assets/woman_ballon.mp4", "Add the woman in a floral dress pointing at the balloon on the left."],
                            ["assets/greenhouse.mp4", "A white Samoyed is watching the man, who crouches in a greenhouse. The Samoyed is covered in thick, fluffy white fur, giving it a very soft and plush appearance. Its ears are erect and triangular, making it look alert and intelligent. The Samoyed's face features its signature smile, with bright black eyes that convey friendliness and curiosity."],
                            ["assets/gameplay.mp4", "Add the woman holding the blue game controller to the left of the man, engaged in gameplay."],
                            ["assets/dog.mp4", "Add the brown and white beagle interacting with and drinking from the metallic bowl on the wooden floor."],
                            ["assets/sign.mp4", "Replace the yellow \"SCHOOL\" sign with a red hospital sign, featuring a white hospital emblem on the top and the word \"HOSPITAL\" below."],
                            ["assets/old_man.mp4", "Swap the old man with long white hair and a blue checkered shirt at the left side of the frame with a woman with curly brown hair and a denim shirt."],
                            ["assets/pants.mp4", "swap the white pants worn by the individual the light blue jeans."],
                            ["assets/bowl.mp4", "Make the largest cup on the right white and smooth."],
                            ["assets/ketchup.mp4", "Make the ketchup bottle to the right of the BBQ sauce bottle violet color."],
                            ["assets/fruit.mp4", "Make the pomegranate at the right side of the basket lavender color."]
                        ],
                    )
                    
                    # Ensure validation_video is visible and interactive
                    validation_video.visible = True
                    validation_video.interactive = True

                    # Set default seed to 0
                    cfg_scale_slider, seed_textbox, seed_button = create_cfg_and_seedbox(True)
                    seed_textbox.value = "0"
                    cfg_scale_slider.value = 1.0
                    
                    generate_button = gr.Button(value="Generate", variant='primary')

                result_image, result_video, infer_progress = create_ui_outputs()

        # Event handlers
        generate_button.click(
            fn=generate_wrapper,
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
                # New inputs
                source_frames_slider,
                reasoning_frames_slider,
                repeat_rope_checkbox,
                enable_acceleration
            ],
            outputs=[result_image, result_video, infer_progress]
        )

    return demo, controller

if __name__ == "__main__":
    from videox_fun.ui.controller import flow_scheduler_dict
    
    # Use CPU offload to reduce GPU memory footprint in Space
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
        prevent_thread_lock=True,
        share=False
    )
    
    while True:
        time.sleep(5)
