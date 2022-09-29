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
from diffusers.schedulers import PNDMScheduler
from transformers import CLIPTokenizer

if __name__ == "__main__":
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
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu()
    option.set_model_path(
        "text_encoder_v1_4.onnx", model_format=ModelFormat.ONNX)
    text_encoder_runtime = fd.Runtime(option)

    option.set_model_path("unet_v1_4.onnx", model_format=ModelFormat.ONNX)
    unet_runtime = fd.Runtime(option)

    option.set_model_path(
        "vae_decoder_v1_4.onnx", model_format=ModelFormat.ONNX)
    vae_decoder_runtime = fd.Runtime(option)

    pipe = StableDiffusionFastDeployPipeline(
        vae_decoder_runtime=vae_decoder_runtime,
        text_encoder_runtime=text_encoder_runtime,
        tokenizer=tokenizer,
        unet_runtime=unet_runtime,
        scheduler=scheduler)

    prompt = "a photo of an astronaut riding a horse on mars"

    start = time.time()
    image = pipe(prompt)
    time_cost = time.time() - start
    image.save("fd_astronaut_rides_horse.png")
    print(f"Image saved! Total time cost: {time_cost:2f} s")
