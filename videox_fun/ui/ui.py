import random

import gradio as gr


def create_model_type(visible):
    gr.Markdown(
        """
        ### Model Type.
        """,
        visible=visible,
    )
    with gr.Row():
        model_type = gr.Dropdown(
            label="The model type of the model",
            choices=["Inpaint", "Control"],
            value="Inpaint",
            visible=visible,
            interactive=True,
        )
    return model_type

def create_fake_model_type(visible):
    gr.Markdown(
        """
        ### Model Type.
        """,
        visible=visible,
    )
    with gr.Row():
        model_type = gr.Dropdown(
            label="The model type of the model",
            choices=["Inpaint", "Control"],
            value="Inpaint",
            interactive=False,
            visible=visible,
        )
    return model_type

def create_model_checkpoints(controller, visible, default_model="none"):
    gr.Markdown(
        """
        ### Model checkpoints.
        """,
        visible=visible
    )
    with gr.Row(visible=visible):
        diffusion_transformer_dropdown = gr.Dropdown(
            label="Pretrained Model Path",
            choices=list(set(controller.diffusion_transformer_list + [default_model])),
            value=default_model,
            interactive=True,
        )
        diffusion_transformer_dropdown.change(
            fn=controller.update_diffusion_transformer, 
            inputs=[diffusion_transformer_dropdown], 
            outputs=[diffusion_transformer_dropdown]
        )
        
        diffusion_transformer_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def refresh_diffusion_transformer():
            controller.refresh_diffusion_transformer()
            return gr.update(choices=controller.diffusion_transformer_list)
        diffusion_transformer_refresh_button.click(fn=refresh_diffusion_transformer, inputs=[], outputs=[diffusion_transformer_dropdown])
    
    return diffusion_transformer_dropdown, diffusion_transformer_refresh_button

def create_fake_model_checkpoints(model_name, visible):
    gr.Markdown(
        """
        ### Model checkpoints.
        """
    )
    with gr.Row(visible=visible):
        diffusion_transformer_dropdown = gr.Dropdown(
            label="Pretrained Model Path",
            choices=[model_name],
            value=model_name,
            interactive=False,
        )
    return diffusion_transformer_dropdown

def create_finetune_models_checkpoints(controller, visible, add_checkpoint_2=False, default_lora="none"):
    with gr.Row(visible=visible):
        base_model_dropdown = gr.Dropdown(
            label="Select base Dreambooth model",
            choices=["none"] + controller.personalized_model_list,
            value="none",
            interactive=True,
        )
        if add_checkpoint_2:
            base_model_2_dropdown = gr.Dropdown(
                label="Select base Dreambooth model",
                choices=["none"] + controller.personalized_model_list,
                value="none",
                interactive=True,
            )
        
        lora_model_dropdown = gr.Dropdown(
            label="Select LoRA model",
            choices=list(set(["none"] + controller.personalized_model_list + [default_lora])),
            value=default_lora,
            interactive=True,
        )
        if add_checkpoint_2:
            lora_model_2_dropdown = gr.Dropdown(
            label="Select LoRA model",
                choices=["none"] + controller.personalized_model_list,
                value="none",
                interactive=True,
            )

        lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.55, minimum=0, maximum=2, interactive=True)
        
        personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def update_personalized_model():
            controller.refresh_personalized_model()
            return [
                gr.update(choices=controller.personalized_model_list),
                gr.update(choices=["none"] + controller.personalized_model_list)
            ]
        personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])

    if not add_checkpoint_2:
        return base_model_dropdown, lora_model_dropdown, lora_alpha_slider, personalized_refresh_button
    else:
        return [base_model_dropdown, base_model_2_dropdown], [lora_model_dropdown, lora_model_2_dropdown], \
            lora_alpha_slider, personalized_refresh_button

def create_fake_finetune_models_checkpoints(visible):
    with gr.Row():
        base_model_dropdown = gr.Dropdown(
            label="Select base Dreambooth model",
            choices=["none"],
            value="none",
            interactive=False,
            visible=False
        )
        with gr.Column(visible=False):
            gr.Markdown(
                """
                ### Minimalism is an example portrait of Lora, triggered by specific prompt words. More details can be found on [Wiki](https://github.com/aigc-apps/CogVideoX-Fun/wiki/Training-Lora).
                """
            )
            with gr.Row():
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model",
                    choices=["none"],
                    value="none",
                    interactive=True,
                )

                lora_alpha_slider = gr.Slider(label="LoRA alpha", value=0.55, minimum=0, maximum=2, interactive=True)
        
    return base_model_dropdown, lora_model_dropdown, lora_alpha_slider

def create_teacache_params(
    enable_teacache = True,
    teacache_threshold = 0.10,
    num_skip_start_steps = 1,
    teacache_offload = False,
):
    enable_teacache = gr.Checkbox(label="Enable TeaCache", value=enable_teacache)
    teacache_threshold = gr.Slider(0.00, 0.25, value=teacache_threshold, step=0.01, label="TeaCache Threshold")
    num_skip_start_steps = gr.Slider(0, 10, value=num_skip_start_steps, step=5, label="Number of Skip Start Steps")
    teacache_offload = gr.Checkbox(label="Offload TeaCache to CPU", value=teacache_offload)
    return enable_teacache, teacache_threshold, num_skip_start_steps, teacache_offload

def create_cfg_skip_params(
    cfg_skip_ratio = 0
):
    cfg_skip_ratio = gr.Slider(0.00, 0.50, value=cfg_skip_ratio, step=0.01, label="CFG Skip Ratio", visible=False)
    return cfg_skip_ratio

def create_cfg_riflex_k(
    enable_riflex = False,
    riflex_k = 6
):
    enable_riflex = gr.Checkbox(label="Enable Riflex", value=enable_riflex, visible=False)
    riflex_k = gr.Slider(0, 10, value=riflex_k, step=1, label="Riflex Intrinsic Frequency Index", visible=False)
    return enable_riflex, riflex_k

def create_prompts(
    prompt="A young woman with beautiful and clear eyes and blonde hair standing and white dress in a forest wearing a crown. She seems to be lost in thought, and the camera focuses on her face. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
):
    gr.Markdown(
        """
        ### Configs for Generation.
        """
    )
    
    prompt_textbox = gr.Textbox(label="Prompt", lines=2, value=prompt)
    negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value=negative_prompt)
    return prompt_textbox, negative_prompt_textbox

def create_samplers(controller, maximum_step=50):
    with gr.Row():
        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(controller.scheduler_dict.keys()), value=list(controller.scheduler_dict.keys())[0])
        sample_step_slider = gr.Slider(label="Sampling steps", value=4, minimum=1, maximum=maximum_step, step=1)
        
    return sampler_dropdown, sample_step_slider

def create_height_width(default_height, default_width, maximum_height, maximum_width):
    resize_method = gr.Radio(
        ["Generate by", "Resize according to Reference"],
        value="Generate by",
        show_label=False,
    )
    width_slider     = gr.Slider(label="Width", value=default_width, minimum=128, maximum=maximum_width, step=16)
    height_slider    = gr.Slider(label="Height", value=default_height, minimum=128, maximum=maximum_height, step=16)
    base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 640, 768, 896, 960, 1024], visible=False)

    return resize_method, width_slider, height_slider, base_resolution

def create_fake_height_width(default_height, default_width, maximum_height, maximum_width):
    resize_method = gr.Radio(
        ["Generate by", "Resize according to Reference"],
        value="Generate by",
        show_label=False,
    )
    width_slider     = gr.Slider(label="Width", value=default_width, minimum=128, maximum=maximum_width, step=16, interactive=False)
    height_slider    = gr.Slider(label="Height", value=default_height, minimum=128, maximum=maximum_height, step=16, interactive=False)
    base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 640, 768, 896, 960, 1024], interactive=False, visible=False)

    return resize_method, width_slider, height_slider, base_resolution

def create_generation_methods_and_video_length(
    generation_method_options,
    default_video_length,
    maximum_video_length
):
    with gr.Group():
        generation_method = gr.Radio(
            generation_method_options,
            value="Video Generation",
            show_label=False,
            visible=False
        )
        with gr.Row():
            length_slider = gr.Slider(label="Animation length", value=default_video_length, minimum=1,   maximum=maximum_video_length,  step=4, visible=False)
            overlap_video_length = gr.Slider(label="Overlap length", value=4, minimum=1,   maximum=4,  step=1, visible=False)
            partial_video_length = gr.Slider(label="Partial video generation length", value=25, minimum=5,   maximum=maximum_video_length,  step=4, visible=False)
                    
    return generation_method, length_slider, overlap_video_length, partial_video_length

def create_generation_method(source_method_options, prompt_textbox, support_end_image=True, support_ref_image=False, default_video=None, video_examples=None):
    default_method = source_method_options[0] if source_method_options else "Text to Video"
    source_method = gr.Radio(
        source_method_options,
        value=default_method,
        show_label=False,
    )
    with gr.Column(visible = (default_method == "Image to Video")) as image_to_video_col:
        start_image = gr.Image(
            label="The image at the beginning of the video",  show_label=True, 
            elem_id="i2v_start", sources="upload", type="filepath", 
        )
        
        template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
        def select_template(evt: gr.SelectData):
            text = {
                "asset/1.png": "A brown dog is shaking its head and sitting on a light colored sofa in a comfortable room. Behind the dog, there is a framed painting on the shelf surrounded by pink flowers. The soft and warm lighting in the room creates a comfortable atmosphere.", 
                "asset/2.png": "A sailboat navigates through moderately rough seas, with waves and ocean spray visible. The sailboat features a white hull and sails, accompanied by an orange sail catching the wind. The sky above shows dramatic, cloudy formations with a sunset or sunrise backdrop, casting warm colors across the scene. The water reflects the golden light, enhancing the visual contrast between the dark ocean and the bright horizon. The camera captures the scene with a dynamic and immersive angle, showcasing the movement of the boat and the energy of the ocean.", 
                "asset/3.png": "A stunningly beautiful woman with flowing long hair stands gracefully, her elegant dress rippling and billowing in the gentle wind. Petals falling off. Her serene expression and the natural movement of her attire create an enchanting and captivating scene, full of ethereal charm.", 
                "asset/4.png": "An astronaut, clad in a full space suit with a helmet, plays an electric guitar while floating in a cosmic environment filled with glowing particles and rocky textures. The scene is illuminated by a warm light source, creating dramatic shadows and contrasts. The background features a complex geometry, similar to a space station or an alien landscape, indicating a futuristic or otherworldly setting.", 
                "asset/5.png": "Fireworks light up the evening sky over a sprawling cityscape with gothic-style buildings featuring pointed towers and clock faces. The city is lit by both artificial lights from the buildings and the colorful bursts of the fireworks. The scene is viewed from an elevated angle, showcasing a vibrant urban environment set against a backdrop of a dramatic, partially cloudy sky at dusk.", 
            }[template_gallery_path[evt.index]]
            return template_gallery_path[evt.index], text

        template_gallery = gr.Gallery(
            template_gallery_path,
            columns=5, rows=1,
            height=140,
            allow_preview=False,
            container=False,
            label="Template Examples",
        )
        template_gallery.select(select_template, None, [start_image, prompt_textbox])
        
        with gr.Accordion("The image at the ending of the video", open=False, visible=support_end_image):
            end_image   = gr.Image(label="The image at the ending of the video", show_label=False, elem_id="i2v_end", sources="upload", type="filepath")

    with gr.Column(visible = (default_method == "Video to Video")) as video_to_video_col:
        with gr.Row():
            validation_video = gr.Video(
                label="The video to convert",  show_label=True, 
                elem_id="v2v", sources=["upload"], value=default_video,
            )
        if video_examples:
            gr.Examples(
                examples=video_examples,
                inputs=[validation_video, prompt_textbox] if len(video_examples[0]) > 1 else validation_video,
                label="Video Examples"
            )

        # Removed Mask Accordion entirely per request or hidden. User said "mask这个不需要"
        # validation_video_mask = gr.Image(
        #     label="The mask of the video to inpaint",
        #     show_label=False, elem_id="v2v_mask", sources="upload", type="filepath",
        #     visible=False
        # )
        validation_video_mask = gr.Image(visible=False, value=None) 
        
        # Denoise strength default 1.0, hidden
        denoise_strength = gr.Slider(label="Denoise strength", value=1.00, minimum=0.10, maximum=1.00, step=0.01, visible=False)

    with gr.Column(visible = False) as control_video_col:
        gr.Markdown(
            """
            Demo pose control video can be downloaded here [URL](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4).
            """
        )
        control_video = gr.Video(
            label="The control video",  show_label=True, 
            elem_id="v2v_control", sources="upload", 
        )
        ref_image = gr.Image(
            label="The reference image for control video",  show_label=True, 
            elem_id="ref_image", sources="upload", type="filepath", visible=support_ref_image
        )
    return image_to_video_col, video_to_video_col, control_video_col, source_method, start_image, template_gallery, end_image, validation_video, validation_video_mask, denoise_strength, control_video, ref_image

def create_cfg_and_seedbox(gradio_version_is_above_4):
    # cfg default 6, hidden
    cfg_scale_slider  = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20, visible=False)
    
    with gr.Row():
        seed_textbox = gr.Textbox(label="Seed", value=43)
        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
        seed_button.click(
            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
            inputs=[], 
            outputs=[seed_textbox]
        )
    return cfg_scale_slider, seed_textbox, seed_button

def create_ui_outputs():
    with gr.Column():
        result_image = gr.Image(label="Generated Image", interactive=False, visible=False)
        result_video = gr.Video(label="Generated Animation", interactive=False)
        infer_progress = gr.Textbox(
            label="Generation Info",
            value="No task currently",
            interactive=False
    )
    return result_image, result_video, infer_progress

def create_config(controller):
    gr.Markdown(
        """
        ### Config Path (配置文件路径)
        """
    )
    with gr.Row():
        config_dropdown = gr.Dropdown(
            label="Config Path",
            choices=controller.config_list,
            value=controller.config_path,
            interactive=True,
        )
        config_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
        def refresh_config():
            controller.refresh_config()
            return gr.update(choices=controller.config_list)
        config_refresh_button.click(fn=refresh_config, inputs=[], outputs=[config_dropdown])
    return config_dropdown, config_refresh_button
