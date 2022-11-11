# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import onnxsim
from typing import Optional, Tuple, Union
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default='CompVis/stable-diffusion-v1-4',
        help="The pretrained diffusion model.")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The pretrained diffusion model.")
    return parser.parse_args()


class VAEDecoder(AutoencoderKL):
    def forward(self, z):
        return self.decode(z, True).sample


if __name__ == "__main__":
    args = parse_arguments()

    # 1. Load VAE model
    vae_decoder = VAEDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        revision="fp16",
        subfolder="vae",
        use_auth_token=True)

    # 2. Load UNet model
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        revision="fp16",
        subfolder="unet",
        use_auth_token=True)

    # 3. Load CLIP model
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14")

    vae_decoder.cuda()
    unet.cuda()
    text_encoder.cuda()

    os.makedirs(args.output_path, exist_ok=True)
    vae_decoder_path = os.path.join(args.output_path, "vae_decoder")
    text_encoder_path = os.path.join(args.output_path, "text_encoder")
    unet_path = os.path.join(args.output_path, "unet")
    for p in [vae_decoder_path, text_encoder_path, unet_path]:
        os.makedirs(p, exist_ok=True)

    with torch.inference_mode():
        # Export vae decoder model
        vae_inputs = (torch.randn(
            1, 4, 64, 64, dtype=torch.half, device='cuda'), )
        torch.onnx.export(
            vae_decoder,  # model being run
            vae_inputs,  # model input (or a tuple for multiple inputs)
            os.path.join(
                vae_decoder_path, "inference.onnx"
            ),  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['latent'],
            dynamic_axes={
                'latent': {
                    0: 'batch_size',
                },
                'image': {
                    0: 'batch_size',
                },
            },
            output_names=['image'])
        print("Finish exporting vae decoder.")

        # Export the unet model
        unet_inputs = (torch.randn(
            2, 4, 64, 64, dtype=torch.half, device='cuda'), torch.randn(
                1, dtype=torch.half, device='cuda'), torch.randn(
                    2, 77, 768, dtype=torch.half, device='cuda'))
        torch.onnx.export(
            unet,  # model being run
            unet_inputs,  # model input (or a tuple for multiple inputs)
            os.path.join(
                unet_path, "inference.onnx"
            ),  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=['latent_input', 'timestep', 'encoder_embedding'],
            dynamic_axes={
                'latent_input': {
                    0: 'batch_size',
                },
                'encoder_embedding': {
                    0: 'batch_size',
                    1: 'sequence'
                },
                'latent_output': {
                    0: 'batch_size',
                },
            },
            output_names=['latent_output'])
        print("Finish exporting unet.")

        # Export the text_encoder
        text_encoder_inputs = (torch.randint(0, 1, (2, 77), device='cuda'), )
        torch.onnx.export(
            text_encoder,  # model being run
            text_encoder_inputs,  # model input (or a tuple for multiple inputs)
            os.path.join(
                text_encoder_path, "inference.onnx"
            ),  # where to save the model (can be a file or file-like object)
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
        print("Finish exporting text encoder.")
