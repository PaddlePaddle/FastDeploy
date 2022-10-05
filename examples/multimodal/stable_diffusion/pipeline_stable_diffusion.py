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
from typing import List, Optional, Union

import numpy as np
import fastdeploy as fd
from transformers import CLIPTokenizer
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
import PIL
from PIL import Image


class StableDiffusionFastDeployPipeline(object):
    vae_decoder_runtime: fd.Runtime
    text_encoder_runtime: fd.Runtime
    tokenizer: CLIPTokenizer
    unet_runtime: fd.Runtime
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]

    def __init__(
            self,
            vae_decoder_runtime: fd.Runtime,
            text_encoder_runtime: fd.Runtime,
            tokenizer: CLIPTokenizer,
            unet_runtime: fd.Runtime,
            scheduler: Union[DDIMScheduler, PNDMScheduler,
                             LMSDiscreteScheduler], ):
        self.vae_decoder_runtime = vae_decoder_runtime
        self.text_encoder_runtime = text_encoder_runtime
        self.tokenizer = tokenizer
        self.unet_runtime = unet_runtime
        self.scheduler = scheduler

    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: Optional[int]=512,
            width: Optional[int]=512,
            num_inference_steps: Optional[int]=50,
            guidance_scale: Optional[float]=7.5,
            eta: Optional[float]=0.0,
            latents: Optional[np.ndarray]=None,
            output_type: Optional[str]="pil",
            return_dict: bool=True,
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

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="np")
            uncond_embeddings = self.text_encoder_runtime.infer({
                input_name: uncond_input.input_ids.astype(np.int64)
            })[0]
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate(
                [uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = np.random.randn(*latents_shape).astype(np.float32)
        elif latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

            # predict the noise residual
            sample_name = self.unet_runtime.get_input_info(0).name
            timestep_name = self.unet_runtime.get_input_info(1).name
            encoder_hidden_states_name = self.unet_runtime.get_input_info(
                2).name
            # Required fp16 input.
            noise_pred = self.unet_runtime.infer({
                sample_name: latent_model_input.astype(np.float16),
                timestep_name: np.array(
                    [t], dtype=np.float16),
                encoder_hidden_states_name: text_embeddings.astype(np.float16),
            })[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents,
                                              **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            latents = np.array(latents)
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        sample_name = self.vae_decoder_runtime.get_input_info(0).name
        image = self.vae_decoder_runtime.infer({
            sample_name: latents.astype(np.float32)
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
