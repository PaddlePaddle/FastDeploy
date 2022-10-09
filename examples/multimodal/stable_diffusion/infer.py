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

import time

import fastdeploy as fd
from fastdeploy import ModelFormat
import numpy as np
from pipeline_stable_diffusion import StableDiffusionFastDeployPipeline
from transformers import CLIPTokenizer
from scheduling_utils import PNDMScheduler


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unet_onnx_file",
        default='unet_v1_4_sim.onnx',
        help="The onnx file path of unet model.")
    parser.add_argument(
        "--vae_onnx_file",
        default='vae_decoder_v1_4.onnx',
        help="The onnx file path of vae model.")
    parser.add_argument(
        "--text_encoder_onnx_file",
        default='text_encoder_v1_4.onnx',
        help="The onnx file path of text_encoder model.")
    parser.add_argument(
        "--backend",
        type=str,
        default='onnx_runtime',
        choices=['onnx_runtime', 'tensorrt'],
        help="The inference runtime backend of unet model and text encoder model."
    )
    return parser.parse_args()


def create_ort_runtime(onnx_file):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    option.use_gpu()
    option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    return fd.Runtime(option)


def create_trt_runtime(onnx_file, workspace=(1 << 31), dynamic_shape=None):
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu()
    option.enable_trt_fp16()
    option.set_trt_max_workspace_size(workspace)
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.set_trt_input_shape(
                key,
                min_shape=shape_dict["min_shape"],
                opt_shape=shape_dict.get("opt_shape", None),
                max_shape=shape_dict.get("opt_shape", None))
    option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    option.set_trt_cache_file(f"{onnx_file}.trt.cache")
    return fd.Runtime(option)


if __name__ == "__main__":
    args = parse_arguments()
    # 1. Init scheduler
    scheduler = PNDMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        skip_prk_steps=True)

    # 2. Init tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 3. Init runtime
    text_encoder_runtime = create_ort_runtime(args.text_encoder_onnx_file)

    if args.backend == "onnx_runtime":
        vae_decoder_runtime = create_ort_runtime(args.vae_onnx_file)
        start = time.time()
        unet_runtime = create_ort_runtime(args.unet_onnx_file)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    else:
        vae_dynamic_shape = {
            "latent": {
                "min_shape": [1, 4, 64, 64],
                "max_shape": [8, 4, 64, 64],
                "opt_shape": [1, 4, 64, 64],
            }
        }
        vae_decoder_runtime = create_trt_runtime(
            args.vae_onnx_file,
            workspace=(1 << 30),
            dynamic_shape=vae_dynamic_shape)
        unet_dynamic_shape = {
            "latent_input": {
                "min_shape": [1, 4, 64, 64],
                "max_shape": [8, 4, 64, 64],
                "opt_shape": [1, 4, 64, 64],
            },
            "encoder_embedding": {
                "min_shape": [1, 77, 768],
                "max_shape": [8, 77, 768],
                "opt_shape": [1, 77, 768],
            },
        }
        start = time.time()
        unet_runtime = create_trt_runtime(
            args.unet_onnx_file, dynamic_shape=unet_dynamic_shape)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    pipe = StableDiffusionFastDeployPipeline(
        vae_decoder_runtime=vae_decoder_runtime,
        text_encoder_runtime=text_encoder_runtime,
        tokenizer=tokenizer,
        unet_runtime=unet_runtime,
        scheduler=scheduler)

    prompt = "a photo of an astronaut riding a horse on mars"

    start = time.time()
    image = pipe(prompt, num_inference_steps=100)[0]
    time_cost = time.time() - start
    image.save("fd_astronaut_rides_horse.png")
    print(f"Image saved! Total time cost: {time_cost:2f} s")
