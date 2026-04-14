# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
Flow Matching Euler Discrete Scheduler for Flux/SD3.

Implements the flow matching ODE solver from:
    Lipman et al., "Flow Matching for Generative Modeling" (2022)

The probability path is a linear interpolation:
    x_t = (1 - t) * x_0 + t * noise
with velocity field v_t(x) predicted by the transformer.
Euler method steps: x_{t-dt} = x_t - dt * v_t(x_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle


@dataclass
class FlowMatchEulerDiscreteScheduler:
    """Euler ODE solver for flow-matching diffusion models.

    Attributes:
        num_train_timesteps: Total training timesteps (defines sigma range).
        shift: Time shift factor — controls noise schedule curvature.
            Flux-dev uses 1.0, Flux-schnell uses 1.0, SD3 uses 3.0.
    """

    num_train_timesteps: int = 1000
    shift: float = 1.0

    # 运行时状态 (Runtime state — set by set_timesteps)
    timesteps: Optional[paddle.Tensor] = field(default=None, init=False, repr=False)
    sigmas: Optional[paddle.Tensor] = field(default=None, init=False, repr=False)
    _step_index: int = field(default=0, init=False, repr=False)
    _num_inference_steps: int = field(default=0, init=False, repr=False)

    def set_timesteps(
        self,
        num_inference_steps: int,
        dtype: paddle.dtype = paddle.float32,
    ) -> None:
        """Compute the sigma schedule for the given number of inference steps.

        For flow matching, sigmas go from 1.0 (pure noise) to 0.0 (clean data).
        Time-shifting is applied: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)

        Args:
            num_inference_steps: Number of denoising steps.
            dtype: Tensor dtype for the schedule.
        """
        self._num_inference_steps = num_inference_steps
        self._step_index = 0

        # 均匀间隔的 sigma 值 (Linearly spaced sigmas from 1→0)
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)

        # 时间偏移 (Time shift — see Flux paper)
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        # 从 sigma 推导 timestep (Derive timesteps: t = sigma * num_train_timesteps)
        timesteps = sigmas[:-1] * self.num_train_timesteps

        self.sigmas = paddle.to_tensor(sigmas, dtype=dtype)
        self.timesteps = paddle.to_tensor(timesteps, dtype=dtype)

    def step(
        self,
        model_output: paddle.Tensor,
        timestep_index: int,
        sample: paddle.Tensor,
    ) -> paddle.Tensor:
        """Perform one Euler step of the flow-matching ODE.

        Euler update: x_{t-dt} = x_t - dt * v_t

        Args:
            model_output: Predicted velocity v_t from the transformer.
            timestep_index: Current step index (0-based).
            sample: Current noisy sample x_t.

        Returns:
            Denoised sample x_{t-dt} after one step.
        """
        sigma = self.sigmas[timestep_index]
        sigma_next = self.sigmas[timestep_index + 1]
        dt = sigma_next - sigma  # dt is negative (moving from noise → data)

        # Euler step: x_{t+dt} = x_t + dt * v_t
        prev_sample = sample + dt * model_output

        self._step_index = timestep_index + 1
        return prev_sample

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timestep_index: int,
    ) -> paddle.Tensor:
        """Add noise to samples at a given sigma level.

        Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise

        Args:
            original_samples: Clean data x_0.
            noise: Gaussian noise.
            timestep_index: Step index to get sigma from.

        Returns:
            Noisy sample x_t.
        """
        sigma = self.sigmas[timestep_index]
        noisy = (1.0 - sigma) * original_samples + sigma * noise
        return noisy

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise level (always 1.0 for flow matching)."""
        return 1.0
