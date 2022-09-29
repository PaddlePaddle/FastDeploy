# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import torch
from typing import Optional, Tuple, Union
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel


class VAEDecoder(AutoencoderKL):
    def forward(self, z):
        return self.decode(z, True).sample


# 1. Load VAE model
vae_decoder = VAEDecoder.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True)

# 2. Load UNet model
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
    subfolder="unet",
    use_auth_token=True)

# 3. Load CLIP model
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

vae_decoder.cuda()
unet.cuda()
text_encoder.cuda()

with torch.inference_mode(), torch.autocast("cuda"):
    # Export vae decoder model
    vae_inputs = (torch.randn(1, 4, 64, 64, device='cuda'), )
    torch.onnx.export(
        vae_decoder,  # model being run
        vae_inputs,  # model input (or a tuple for multiple inputs)
        "vae_decoder_v1_4.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input_0'],
        output_names=['output_0'])

    # Export the unet model
    unet_inputs = (torch.randn(
        2, 4, 64, 64, dtype=torch.half, device='cuda'), torch.randn(
            1, dtype=torch.half, device='cuda'), torch.randn(
                2, 77, 768, dtype=torch.half, device='cuda'))
    torch.onnx.export(
        unet,  # model being run
        unet_inputs,  # model input (or a tuple for multiple inputs)
        "unet_v1_4.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input_0', 'input_1', 'input_2'],
        output_names=['output_0'])

    # Export the text_encoder
    text_encoder_inputs = (torch.randint(0, 1, (2, 77), device='cuda'), )
    torch.onnx.export(
        text_encoder,  # model being run
        text_encoder_inputs,  # model input (or a tuple for multiple inputs)
        "text_encoder_v1_4.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input_ids'],
        dynamic_axes={
            'input_ids': {
                0: 'batch_size',
                1: 'sequence'
            },
            'logits': {
                0: 'batch_size',
                1: 'sequence'
            }
        },
        output_names=['logits'])
