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
import os

from pipeline_stable_diffusion import StableDiffusionFastDeployPipeline
from scheduling_utils import PNDMScheduler
from paddlenlp.transformers import CLIPTokenizer

import fastdeploy as fd
from fastdeploy import ModelFormat
import numpy as np


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="paddle_diffusion_model",
        help="The model directory of diffusion_model.")
    parser.add_argument(
        "--model_format",
        default="paddle",
        choices=['paddle', 'onnx'],
        help="The model format.")
    parser.add_argument(
        "--unet_model_prefix",
        default='unet',
        help="The file prefix of unet model.")
    parser.add_argument(
        "--vae_model_prefix",
        default='vae_decoder',
        help="The file prefix of vae model.")
    parser.add_argument(
        "--text_encoder_model_prefix",
        default='text_encoder',
        help="The file prefix of text_encoder model.")
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=100,
        help="The number of unet inference steps.")
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=1,
        help="The number of performance benchmark steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default='ort',
        choices=['ort', 'trt', 'pp', 'pp-trt'],
        help="The inference runtime backend of unet model and text encoder model."
    )
    parser.add_argument(
        "--image_path",
        default="fd_astronaut_rides_horse.png",
        help="The model directory of diffusion_model.")
    return parser.parse_args()


def create_ort_runtime(model_dir, model_prefix, model_format):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    option.use_gpu()
    if model_format == "paddle":
        model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
        params_file = os.path.join(model_dir, model_prefix,
                                   "inference.pdiparams")
        option.set_model_path(model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, model_prefix, "inference.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    return fd.Runtime(option)


def create_paddle_inference_runtime(model_dir,
                                    model_prefix,
                                    use_trt=False,
                                    dynamic_shape=None):
    option = fd.RuntimeOption()
    option.use_paddle_backend()
    if use_trt:
        option.use_trt_backend()
        option.enable_paddle_to_trt()
        # Need to enable collect shape for ernie
        option.enable_paddle_trt_collect_shape()
        if dynamic_shape is not None:
            for key, shape_dict in dynamic_shape.items():
                option.set_trt_input_shape(
                    key,
                    min_shape=shape_dict["min_shape"],
                    opt_shape=shape_dict.get("opt_shape", None),
                    max_shape=shape_dict.get("max_shape", None))
    option.use_gpu()
    model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
    params_file = os.path.join(model_dir, model_prefix, "inference.pdiparams")
    option.set_model_path(model_file, params_file)
    return fd.Runtime(option)


def create_trt_runtime(model_dir,
                       model_prefix,
                       model_format,
                       workspace=(1 << 31),
                       dynamic_shape=None):
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
                max_shape=shape_dict.get("max_shape", None))
    if model_format == "paddle":
        model_file = os.path.join(model_dir, model_prefix, "inference.pdmodel")
        params_file = os.path.join(model_dir, model_prefix,
                                   "inference.pdiparams")
        option.set_model_path(model_file, params_file)
    else:
        onnx_file = os.path.join(model_dir, model_prefix, "inference.onnx")
        option.set_model_path(onnx_file, model_format=ModelFormat.ONNX)
    cache_file = os.path.join(model_dir, model_prefix, "inference.trt")
    option.set_trt_cache_file(cache_file)
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

    # 3. Set dynamic shape for trt backend
    vae_dynamic_shape = {
        "latent": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        }
    }

    unet_dynamic_shape = {
        "latent_input": {
            "min_shape": [1, 4, 64, 64],
            "max_shape": [2, 4, 64, 64],
            "opt_shape": [2, 4, 64, 64],
        },
        "encoder_embedding": {
            "min_shape": [1, 77, 768],
            "max_shape": [2, 77, 768],
            "opt_shape": [2, 77, 768],
        },
    }

    # 4. Init runtime
    if args.backend == "ort":
        text_encoder_runtime = create_ort_runtime(
            args.model_dir, args.text_encoder_model_prefix, args.model_format)
        vae_decoder_runtime = create_ort_runtime(
            args.model_dir, args.vae_model_prefix, args.model_format)
        start = time.time()
        unet_runtime = create_ort_runtime(
            args.model_dir, args.unet_model_prefix, args.model_format)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    elif args.backend == "pp" or args.backend == "pp-trt":
        use_trt = True if args.backend == "pp-trt" else False
        text_encoder_runtime = create_paddle_inference_runtime(
            args.model_dir, args.text_encoder_model_prefix)
        vae_decoder_runtime = create_paddle_inference_runtime(
            args.model_dir, args.vae_model_prefix, use_trt, vae_dynamic_shape)
        start = time.time()
        unet_runtime = create_paddle_inference_runtime(
            args.model_dir, args.unet_model_prefix, use_trt,
            unet_dynamic_shape)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    elif args.backend == "trt":
        text_encoder_runtime = create_ort_runtime(
            args.model_dir, args.text_encoder_model_prefix, args.model_format)
        vae_decoder_runtime = create_trt_runtime(
            args.model_dir,
            args.vae_model_prefix,
            args.model_format,
            workspace=(1 << 30),
            dynamic_shape=vae_dynamic_shape)
        start = time.time()
        unet_runtime = create_trt_runtime(
            args.model_dir,
            args.unet_model_prefix,
            args.model_format,
            dynamic_shape=unet_dynamic_shape)
        print(f"Spend {time.time() - start : .2f} s to load unet model.")
    pipe = StableDiffusionFastDeployPipeline(
        vae_decoder_runtime=vae_decoder_runtime,
        text_encoder_runtime=text_encoder_runtime,
        tokenizer=tokenizer,
        unet_runtime=unet_runtime,
        scheduler=scheduler)

    prompt = "a photo of an astronaut riding a horse on mars"
    # Warm up
    pipe(prompt, num_inference_steps=10)

    time_costs = []
    print(
        f"Run the stable diffusion pipeline {args.benchmark_steps} times to test the performance."
    )
    for step in range(args.benchmark_steps):
        start = time.time()
        image = pipe(prompt, num_inference_steps=args.inference_steps)[0]
        latency = time.time() - start
        time_costs += [latency]
        print(f"No {step:3d} time cost: {latency:2f} s")
    print(
        f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
        f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
    )
    image.save(args.image_path)
    print(f"Image saved in {args.image_path}!")
