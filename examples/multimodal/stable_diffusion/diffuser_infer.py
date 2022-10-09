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
# limitations under the License.from torch import autocast
from diffusers import StableDiffusionPipeline
import time
import torch
import numpy as np


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
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
        "--use_fp16",
        type=ast.literal_eval,
        default=True,
        help="Wether to use tensorrt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.use_fp16:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", use_auth_token=True)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    # Warm up
    pipe(prompt, num_inference_steps=10)

    time_costs = []
    print(
        f"Run the stable diffusion pipeline {args.benchmark_steps} times to test the performance."
    )
    for step in range(args.benchmark_steps):
        start = time.time()
        image = pipe(
            prompt, num_inference_steps=args.inference_steps).images[0]
        latency = time.time() - start
        time_costs += [latency]
        print(f"No {step:3d} time cost: {latency:2f} s")

    print(
        f"Mean latency: {np.mean(time_costs):2f}, p50 latency: {np.percentile(time_costs, 50):2f}, "
        f"p90 latency: {np.percentile(time_costs, 90):2f}, p95 latency: {np.percentile(time_costs, 95):2f}."
    )
    image.save("astronaut_rides_horse.png")
    print(f"Image saved!")
