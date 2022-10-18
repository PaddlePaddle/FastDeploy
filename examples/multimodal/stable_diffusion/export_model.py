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

import os
import paddle
import paddlenlp

from diffusers_paddle import UNet2DConditionModel, AutoencoderKL
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer


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
    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "text_encoder"))
    vae_decoder = VAEDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")

    # Convert to static graph with specific input description
    text_encoder = paddle.jit.to_static(
        text_encoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64")  # segment_ids
        ])

    # Save text_encoder in static graph model.
    save_path = os.path.join(args.output_path, "text_encoder")
    paddle.jit.save(text_encoder, save_path)
    print(f"Save text_encoder model in {save_path} successfully.")

    # Convert to static graph with specific input description
    vae_decoder = paddle.jit.to_static(
        vae_decoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 4, 64, 64], dtype="float32"),  # latent
        ])
    # Save vae_decoder in static graph model.
    save_path = os.path.join(args.output_path, "vae_decoder")
    paddle.jit.save(vae_decoder, save_path)
    print(f"Save vae_decoder model in {save_path} successfully.")

    # Convert to static graph with specific input description
    unet = paddle.jit.to_static(
        unet,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 4, 64, 64], dtype="float32"),  # latent
            paddle.static.InputSpec(
                shape=[1], dtype="float32"),  # timesteps
            paddle.static.InputSpec(
                shape=[None, None], dtype="float32")  # encoder_embedding
        ])
    save_path = os.path.join(args.output_path, "unet")
    paddle.jit.save(unet, save_path)
    print(f"Save unet model in {save_path} successfully.")
