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

from dbimutils import smart_imread_pil, smart_24bit, make_square, smart_resize

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
    pfg_feature = (weight @ pfg_hidden_state + bias) * pfg_scale
    pfg_feature = pfg_feature.reshape(1, pfg_num_tokens, -1)
    e.pipe.plugin_data[plugin_id][1] = pfg_feature

@event_handler()
def pre_unet_predict(e: UNetDenoisingEvent):
    (
        enabled,
        pfg_feature,
        _,
        _,
        pfg_num_tokens,
    ) = e.pipe.plugin_data[plugin_id]
    if enabled or e.step > 0:
        return
    
    
    print(f"Apply pfg")
    #(batch_size*num_prompts, cond_tokens, dim)
    uncond, cond = e.prompt_embeds.chunk(2)
    #(1, num_tokens, dim)
    pfg_cond = pfg_feature.to(cond.device, dtype = cond.dtype)
    pfg_cond = pfg_cond.repeat(cond.shape[0],1,1)
    #concatenate
    cond = torch.cat([cond,pfg_cond],dim=1)

    #copy zero
    pfg_uncond_zero = torch.zeros(uncond.shape[0],pfg_num_tokens,uncond.shape[2]).to(uncond.device, dtype = uncond.dtype)
    uncond = torch.cat([uncond,pfg_uncond_zero],dim=1)
    e.prompt_embeds = torch.cat([uncond, cond])