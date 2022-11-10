# Copyright 2022 The HuggingFace Inc. team.
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

import inspect
from typing import Callable, List, Optional, Union
import numpy as np

from paddlenlp.transformers import CLIPTokenizer
import fastdeploy as fd
from scheduling_utils import PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
import PIL
from PIL import Image
import logging


class StableDiffusionFastDeployPipeline(object):
    vae_decoder_runtime: fd.Runtime
    text_encoder_runtime: fd.Runtime
    tokenizer: CLIPTokenizer
    unet_runtime: fd.Runtime
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler,
                     EulerAncestralDiscreteScheduler]

    def __init__(self,
                 vae_decoder_runtime: fd.Runtime,
                 text_encoder_runtime: fd.Runtime,
                 tokenizer: CLIPTokenizer,
                 unet_runtime: fd.Runtime,
                 scheduler: Union[DDIMScheduler, PNDMScheduler,
                                  LMSDiscreteScheduler]):
        self.vae_decoder_runtime = vae_decoder_runtime
        self.text_encoder_runtime = text_encoder_runtime
        self.unet_runtime = unet_runtime
        self.scheduler = scheduler
        self.tokenizer = tokenizer

    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int]=512,
            width: Optional[int]=512,
            num_inference_steps: Optional[int]=50,
            guidance_scale: Optional[float]=7.5,
            negative_prompt: Optional[Union[str, List[str]]]=None,
            num_images_per_prompt: Optional[int]=1,
            eta: Optional[float]=0.0,
            generator: Optional[np.random.RandomState]=None,
            latents: Optional[np.ndarray]=None,
            output_type: Optional[str]="pil",
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, np.ndarray], None]]=None,
            callback_steps: Optional[int]=1,
            **kwargs, ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}.")

        if generator is None:
            generator = np.random

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="np", )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}")
            text_input_ids = text_input_ids[:, :
                                            self.tokenizer.model_max_length]

        input_name = self.text_encoder_runtime.get_input_info(0).name
        text_embeddings = self.text_encoder_runtime.infer({
            input_name: text_input_ids.astype(np.int64)
        })[0]
        text_embeddings = np.repeat(
            text_embeddings, num_images_per_prompt, axis=0)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np")
            uncond_embeddings = self.text_encoder_runtime.infer({
                input_name: uncond_input.input_ids.astype(np.int64)
            })[0]
            uncond_embeddings = np.repeat(
                uncond_embeddings, num_images_per_prompt, axis=0)
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate(
                [uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_dtype = text_embeddings.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8,
                         width // 8)
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)

            # predict the noise residual
            sample_name = self.unet_runtime.get_input_info(0).name
            timestep_name = self.unet_runtime.get_input_info(1).name
            encoder_hidden_states_name = self.unet_runtime.get_input_info(
                2).name
            # Required fp16 input.
            input_type = [np.float16, np.float16, np.float16]
            if self.unet_runtime.get_input_info(0).dtype == fd.FDDataType.FP32:
                input_type = [np.float32, np.int64, np.float32]
            noise_pred = self.unet_runtime.infer({
                sample_name: latent_model_input.astype(input_type[0]),
                timestep_name: np.array(
                    [t], dtype=input_type[1]),
                encoder_hidden_states_name:
                text_embeddings.astype(input_type[2]),
            })[0]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample
            latents = np.array(latents)
            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        sample_name = self.vae_decoder_runtime.get_input_info(0).name
        input_dtype = np.float16
        if self.vae_decoder_runtime.get_input_info(
                0).dtype == fd.FDDataType.FP32:
            input_dtype = np.float32
        image = self.vae_decoder_runtime.infer({
            sample_name: latents.astype(input_dtype)
        })[0]

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images
