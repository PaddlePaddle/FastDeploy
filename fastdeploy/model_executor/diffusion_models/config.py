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

"""Module for Hackathon 10th Spring No.48."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import paddle


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model inference pipelines.

    Attributes:
        model_name_or_path: HuggingFace model ID or local path to the model.
        model_type: Architecture type — "flux" or "sd3".
        num_inference_steps: Number of denoising steps. Flux default is 28.
        guidance_scale: Classifier-free guidance scale. 0.0 for Flux-schnell,
            3.5 for Flux-dev. SD3 typically uses 7.0.
        image_height: Output image height in pixels (must be divisible by 16).
        image_width: Output image width in pixels (must be divisible by 16).
        scheduler_type: Scheduler to use. "flow_match_euler" is default for Flux.
        dtype: Compute dtype string — "float16", "bfloat16", or "float32".
        vae_path: Optional override path for VAE weights.
        max_sequence_length: Maximum token length for T5 encoder (default 512).
        seed: Random seed for reproducibility. None for random.
    """

    model_name_or_path: str = ""
    model_type: Literal["flux", "sd3"] = "flux"

    # 推理参数 (Inference parameters)
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    image_height: int = 1024
    image_width: int = 1024
    max_sequence_length: int = 512
    seed: Optional[int] = None

    # 调度器 (Scheduler)
    scheduler_type: str = "flow_match_euler"

    # 精度 (Precision)
    dtype: str = "bfloat16"

    # 可选路径覆盖 (Optional path overrides)
    vae_path: Optional[str] = None

    def get_paddle_dtype(self) -> paddle.dtype:
        """Convert string dtype to paddle.dtype."""
        dtype_map = {
            "float16": paddle.float16,
            "bfloat16": paddle.bfloat16,
            "float32": paddle.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Choose from: {list(dtype_map.keys())}")
        return dtype_map[self.dtype]

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.model_name_or_path:
            raise ValueError(
                "model_name_or_path must be specified. "
                "Example: DiffusionConfig(model_name_or_path='black-forest-labs/FLUX.1-dev')"
            )
        if self.image_height % 16 != 0 or self.image_width % 16 != 0:
            raise ValueError(
                f"image_height ({self.image_height}) and image_width ({self.image_width}) " "must be divisible by 16."
            )
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.guidance_scale < 0.0:
            raise ValueError(f"guidance_scale must be >= 0.0, got {self.guidance_scale}")
        if self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1, got {self.max_sequence_length}")
        if self.model_type not in ("flux", "sd3"):
            raise ValueError(f"model_type must be 'flux' or 'sd3', got '{self.model_type}'")
        # Validate dtype is supported
        self.get_paddle_dtype()
