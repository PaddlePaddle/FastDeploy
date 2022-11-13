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

import math
from typing import Optional, Tuple, Union, Any
from scipy import integrate

import numpy as np
from config_utils import register_to_config, ConfigMixin
from dataclasses import dataclass
from collections import OrderedDict

SCHEDULER_CONFIG_NAME = "scheduler_config.json"


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            if self.__class__.__name__ in [
                    "StableDiffusionPipelineOutput", "ImagePipelineOutput"
            ] and k == "sample":
                deprecate("samples", "0.6.0",
                          "Please use `.images` or `'images'` instead.")
                return inner_dict["images"]
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class SchedulerMixin:
    """
    Mixin containing common functions for the schedulers.
    """

    config_name = SCHEDULER_CONFIG_NAME

    def set_format(self, tensor_format="pt"):
        return self


class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.
    Args:
        prev_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: np.ndarray


class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (` np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (` np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: np.ndarray
    pred_original_sample: Optional[np.ndarray] = None


class LMSDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: np.ndarray
    pred_original_sample: Optional[np.ndarray] = None


class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`np.ndarray` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: np.ndarray
    pred_original_sample: Optional[np.ndarray] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2)**2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.concatenate(betas).astype(np.float32)


class PNDMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int=1000,
            beta_start: float=0.0001,
            beta_end: float=0.02,
            beta_schedule: str="linear",
            trained_betas: Optional[np.ndarray]=None,
            skip_prk_steps: bool=False,
            set_alpha_to_one: bool=False,
            steps_offset: int=0,
            **kwargs, ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=np.float32)**2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.final_alpha_cumprod = 1.0 if set_alpha_to_one else self.alphas_cumprod[
            0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(
            0, num_train_timesteps)[::-1].copy().astype("int64")
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, **kwargs) -> np.ndarray:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        offset = self.config.steps_offset

        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        self._timesteps = (np.arange(0, num_inference_steps) *
                           step_ratio).round()
        self._timesteps += offset

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([
                self._timesteps[:-1], self._timesteps[-2:-1],
                self._timesteps[-1:]
            ])[::-1].copy()
        else:
            prk_timesteps = np.array(self._timesteps[
                -self.pndm_order:]).repeat(2) + np.tile(
                    np.array([
                        0, self.config.num_train_timesteps //
                        num_inference_steps // 2
                    ]), self.pndm_order)
            self.prk_timesteps = (
                prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][::-1].copy()

        self.timesteps = np.concatenate(
            [self.prk_timesteps, self.plms_timesteps]).astype(np.int64)

        self.ets = []
        self.counter = 0

    def step(
            self,
            model_output: np.ndarray,
            timestep: int,
            sample: np.ndarray,
            return_dict: bool=True, ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.counter < len(
                self.prk_timesteps) and not self.config.skip_prk_steps:
            return self.step_prk(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict)
        else:
            return self.step_plms(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                return_dict=return_dict)

    def step_prk(self,
                 model_output: np.ndarray,
                 timestep: int,
                 sample: np.ndarray,
                 return_dict: bool=True):
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = 0 if self.counter % 2 else self.config.num_train_timesteps // self.num_inference_steps // 2
        prev_timestep = timestep - diff_to_prev
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += 1 / 6 * model_output
            self.ets.append(model_output)
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 3) % 4 == 0:
            model_output = self.cur_model_output + 1 / 6 * model_output
            self.cur_model_output = 0

        # cur_sample should not be `None`
        cur_sample = self.cur_sample if self.cur_sample is not None else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep,
                                            prev_timestep, model_output)
        self.counter += 1
        if not return_dict:
            return (prev_sample, )

        return SchedulerOutput(prev_sample=prev_sample)

    def step_plms(self,
                  model_output: np.ndarray,
                  timestep: int,
                  sample: np.ndarray,
                  return_dict: bool=True):
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.config.skip_prk_steps and len(self.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information.")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        if self.counter != 1:
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (
                23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] +
                                       37 * self.ets[-3] - 9 * self.ets[-4])

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep,
                                            model_output)
        self.counter += 1
        if not return_dict:
            return (prev_sample, )
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: np.ndarray, *args,
                          **kwargs) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`np.ndarray`): input sample
        Returns:
            `np.ndarray`: scaled input sample
        """
        return sample

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        sample_coeff = (alpha_prod_t_prev / alpha_prod_t)**(0.5)

        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev**(0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev)**(0.5)

        prev_sample = (sample_coeff * sample -
                       (alpha_prod_t_prev - alpha_prod_t
                        ) * model_output / model_output_denom_coeff)

        return prev_sample

    def add_noise(
            self,
            original_samples: np.ndarray,
            noise: np.ndarray,
            timesteps: np.ndarray, ) -> np.ndarray:

        sqrt_alpha_prod = self.alphas_cumprod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps])**0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(
                original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.

    """

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int=1000,
            beta_start: float=0.0001,
            beta_end: float=0.02,
            beta_schedule: str="linear",
            trained_betas: Optional[np.ndarray]=None,
            clip_sample: bool=True,
            set_alpha_to_one: bool=True,
            steps_offset: int=0,
            **kwargs, ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=np.float32)**2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = 1.0 if set_alpha_to_one else self.alphas_cumprod[
            0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1]

    def scale_model_input(self,
                          sample: np.ndarray,
                          timestep: Optional[int]=None) -> np.ndarray:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`np.ndarray`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `np.ndarray`: scaled input sample
        """
        return sample

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int, **kwargs):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        offset = self.config.steps_offset

        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        self.timesteps = (np.arange(0, num_inference_steps) *
                          step_ratio).round()[::-1]
        self.timesteps += offset

    def step(
            self,
            model_output: np.ndarray,
            timestep: int,
            sample: np.ndarray,
            eta: float=0.0,
            use_clipped_model_output: bool=False,
            generator=None,
            return_dict: bool=True, ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): TODO
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t**
                                (0.5) * model_output) / alpha_prod_t**(0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = np.clip(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**(0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t**
                            (0.5) * pred_original_sample) / beta_prod_t**(0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**(
            0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev**(
            0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = np.random.randn(*model_output.shape)
            variance = self._get_variance(timestep, prev_timestep)**(
                0.5) * eta * noise

            prev_sample = prev_sample + variance
        if not return_dict:
            return (prev_sample, )
        return DDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def __len__(self):
        return self.config.num_train_timesteps


class LMSDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.

    """

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int=1000,
            beta_start: float=0.0001,
            beta_end: float=0.02,
            beta_schedule: str="linear",
            trained_betas: Optional[np.ndarray]=None,
            **kwargs, ):
        if trained_betas is not None:
            self.betas = trained_betas
        elif beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype=np.float32)**2)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod)**
                          0.5)
        self.sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps,
            dtype=float)[::-1].copy()
        self.derivatives = []

    def scale_model_input(self,
                          sample: np.ndarray,
                          timestep: Union[float, np.ndarray]) -> np.ndarray:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.
        Args:
            sample (`np.ndarray`): input sample
            timestep (`float` or `np.ndarray`): the current timestep in the diffusion chain
        Returns:
            `np.ndarray`: scaled input sample
        """
        step_index = (self.timesteps == timestep).nonzero()[0]
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1)**0.5)
        self.is_scale_input_called = True
        return sample

    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute a linear multistep coefficient.

        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (
                    self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(
            lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(
            0,
            self.config.num_train_timesteps - 1,
            num_inference_steps,
            dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod)**
                          0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = sigmas
        self.timesteps = timesteps

        self.derivatives = []

    def step(
            self,
            model_output: np.ndarray,
            timestep: int,
            sample: np.ndarray,
            order: int=4,
            return_dict: bool=True, ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than LMSDiscreteSchedulerOutput class

        Returns:
            [`~scheduling_utils.LMSDiscreteSchedulerOutput`] or `tuple`:
            [`~scheduling_utils.LMSDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
        sigma = self.sigmas[int(timestep)]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        # 3. Compute linear multistep coefficients
        order = min(timestep + 1, order)
        lms_coeffs = [
            self.get_lms_coefficient(order, timestep, curr_order)
            for curr_order in range(order)
        ]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = sample + sum(coeff * derivative
                                   for coeff, derivative in zip(
                                       lms_coeffs, reversed(self.derivatives)))

        if not return_dict:
            return (prev_sample, )

        return LMSDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
            self,
            original_samples: np.ndarray,
            noise: np.ndarray,
            timesteps: np.ndarray, ) -> np.ndarray:
        sigmas = self.sigmas

        sigma = sigmas[timesteps].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps


class EulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Ancestral sampling with Euler method steps. Based on the original k-diffusion implementation by Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L72
    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.
    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
    """

    _compatible_classes = [
        "DDIMScheduler",
        "DDPMScheduler",
        "LMSDiscreteScheduler",
        "PNDMScheduler",
        "EulerDiscreteScheduler",
        "DPMSolverMultistepScheduler",
    ]

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int=1000,
            beta_start: float=0.0001,
            beta_end: float=0.02,
            beta_schedule: str="linear",
            trained_betas: Optional[np.ndarray]=None, ):
        if trained_betas is not None:
            self.betas = np.array(trained_betas)
        elif beta_schedule == "linear":
            self.betas = np.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_train_timesteps,
                dtype="float32")**2)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod)**
                          0.5)
        self.sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(
            0, num_train_timesteps - 1, num_train_timesteps,
            dtype=float)[::-1].copy()
        self.timesteps = timesteps
        self.is_scale_input_called = False

    def scale_model_input(self,
                          sample: np.ndarray,
                          timestep: Union[float, np.ndarray]) -> np.ndarray:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        Args:
            sample (`np.ndarray`): input sample
            timestep (`float` or `np.ndarray`): the current timestep in the diffusion chain
        Returns:
            `np.ndarray`: scaled input sample
        """
        step_index = (self.timesteps == timestep).nonzero()[0]
        sigma = self.sigmas[step_index]
        sample = sample / ((sigma**2 + 1)**0.5)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(
            0,
            self.config.num_train_timesteps - 1,
            num_inference_steps,
            dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod)**
                          0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = sigmas
        self.timesteps = timesteps

    def step(
            self,
            model_output: np.ndarray,
            timestep: Union[float, np.ndarray],
            sample: np.ndarray,
            return_dict: bool=True, ) -> Union[
                EulerAncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`np.ndarray`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than EulerAncestralDiscreteSchedulerOutput class
        Returns:
            [`~scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            [`~scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] if `return_dict` is True, otherwise
            a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        if not self.is_scale_input_called:
            logger.warn(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example.")
        step_index = (self.timesteps == timestep).nonzero()[0]
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma * model_output
        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from
                    **2)**0.5
        sigma_down = (sigma_to**2 - sigma_up**2)**0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt
        noise = np.random.randn(*model_output.shape).astype(model_output.dtype)

        prev_sample = prev_sample + noise * sigma_up

        if not return_dict:
            return (prev_sample, )

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
            self,
            original_samples: np.ndarray,
            noise: np.ndarray,
            timesteps: np.ndarray, ) -> np.ndarray:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        self.sigmas = self.sigmas.astype(original_samples.dtype)

        schedule_timesteps = self.timesteps
        step_indices = [(schedule_timesteps == t).nonzero() for t in timesteps]

        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
