import gc
from typing import *

import gradio as gr
import numpy as np
import PIL.Image
import torch

from api.events import event_handler
from api.events.common import PreUICreateEvent
from api.events.generation import LoadResourceEvent, UNetDenoisingEvent
from api.models.diffusion import ImageGenerationOptions
from api.plugin import get_plugin_id
from modules import config
from modules.logger import logger
from modules.plugin_loader import register_plugin_ui
from modules.shared import hf_diffusers_cache_dir

plugin_id = get_plugin_id()

import onnxruntime
tagger = onnxruntime.InferenceSession(os.path.join(CURRENT_DIRECTORY, ONNX_FILE),providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def ui():
    with gr.Group():
        with gr.Accordion("PFG", open=False):
            enabled = gr.Checkbox(value=False, label="Enable")
            with gr.Row():
                image = gr.Image(type="pil", label="guide image")
            with gr.Row():
                pfg_scale = gr.Slider(minimum=0, maximum=3, step=0.05, label="pfg scale", value=1.0)
            with gr.Row():
                pfg_path = gr.Dropdown(self.model_list, label="pfg model", value = None)
            with gr.Row():
                pfg_num_tokens = gr.Slider(minimum=0, maximum=20, step=1.0, value=10.0, label="pfg num tokens")
    return [
        enabled,
        image,
        pfg_scale,
        pfg_path,
        pdf_num_tokens
    ]

#wd-14-taggerの推論関数
def infer(img:Image):
    img = smart_imread_pil(img)
    img = smart_24bit(img)
    img = make_square(img, 448)
    img = smart_resize(img, 448)
    img = img.astype(np.float32)
    print("inferencing by tensorflow model.")
    probs = self.tagger.run([self.tagger.get_outputs()[0].name],{self.tagger.get_inputs()[0].name: np.array([img])})[0]
    return torch.tensor(probs).squeeze(0).cpu()


@event_handler()
def pre_ui_create(e: PreUICreateEvent):
    register_plugin_ui(ui)

@event_handler()
def load_resource(e: LoadResourceEvent):
    (
        enabled,
        image,
        pfg_scale,
        pfg_path,
        pfg_num_tokens
    ) = e.pipe.plugin_data[plugin_id]
    
    if not enabled:
        return

    if type(e.pipe).__mode__ != "diffusers":
        logger.warning("PFG plugin is only available in diffusers mode")
        e.pipe.plugin_data[plugin_id][0] = False
        return

    hidden_state = infer(image)

    pfg_weight = torch.load(os.path.join(pfg_path))
    weight = pfg_weight["pfg_linear.weight"].cpu() #大した計算じゃないのでcpuでいいでしょう
    bias = pfg_weight["pfg_linear.bias"].cpu()
    pfg_feature = (weight @ pfg_hidden_state + self.bias) * pfg_scale

@event_handler()
def pre_unet_predict(e: UNetDenoisingEvent):
    (
        image,
        enabled,
        _,
        _,
        control_weight,
        start_control_step,
        end_control_step,
        guess_mode,
    ) = e.pipe.plugin_data[plugin_id]
    opts: ImageGenerationOptions = e.pipe.opts
    if (
        not enabled
        or e.step / opts.num_inference_steps < start_control_step
        or e.step / opts.num_inference_steps > end_control_step
    ):
        return

    if guess_mode and e.do_classifier_free_guidance:
        # Infer ControlNet only for the conditional batch.
        controlnet_latent_model_input = e.latents
        controlnet_prompt_embeds = e.prompt_embeds.chunk(2)[1]
    else:
        controlnet_latent_model_input = e.latent_model_input
        controlnet_prompt_embeds = e.prompt_embeds

    down_block_res_samples, mid_block_res_sample = controlnet_model(
        controlnet_latent_model_input,
        e.timestep,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=image,
        conditioning_scale=control_weight,
        guess_mode=guess_mode,
        return_dict=False,
    )

    if guess_mode and e.do_classifier_free_guidance:
        # Infered ControlNet only for the conditional batch.
        # To apply the output of ControlNet to both the unconditional and conditional batches,
        # add 0 to the unconditional batch to keep it unchanged.
        down_block_res_samples = [
            torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
        ]
        mid_block_res_sample = torch.cat(
            [torch.zeros_like(mid_block_res_sample), mid_block_res_sample]
        )

    e.unet_additional_kwargs = {
        "down_block_additional_residuals": down_block_res_samples,
        "mid_block_additional_residual": mid_block_res_sample,
    }
