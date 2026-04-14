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
Weight loading utilities for diffusion models.

Supports loading from:
  - PaddlePaddle state dicts (.pdparams)
  - SafeTensors format (.safetensors)

Handles PyTorch → Paddle key mapping for common diffusion model weights.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import paddle

logger = logging.getLogger(__name__)


def _torch_key_to_paddle(key: str) -> str:
    """Convert a PyTorch state dict key to Paddle convention.

    For Flux/SD3 diffusion models, PyTorch and Paddle use identical
    key names (both use [out, in, kH, kW] for Conv2D).  This function
    exists as an extension point for future architectures that may
    require key remapping.
    """
    return key


def load_safetensors_to_paddle(
    filepath: str,
    dtype: Optional[paddle.dtype] = None,
) -> Dict[str, paddle.Tensor]:
    """Load a safetensors file and return a Paddle state dict.

    Args:
        filepath: Path to the .safetensors file.
        dtype: Optional dtype to cast all tensors to.

    Returns:
        Dictionary mapping parameter names to Paddle tensors.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "safetensors package is required for loading .safetensors files. " "Install with: pip install safetensors"
        )

    state_dict = {}
    with safe_open(filepath, framework="numpy") as f:
        for key in f.keys():
            np_tensor = f.get_tensor(key)
            paddle_key = _torch_key_to_paddle(key)
            tensor = paddle.to_tensor(np_tensor)
            if dtype is not None:
                tensor = tensor.cast(dtype)
            state_dict[paddle_key] = tensor

    logger.info("Loaded %d tensors from %s", len(state_dict), filepath)
    return state_dict


def load_paddle_state_dict(filepath: str) -> Dict[str, paddle.Tensor]:
    """Load a Paddle .pdparams state dict.

    Args:
        filepath: Path to the .pdparams file.

    Returns:
        Dictionary mapping parameter names to Paddle tensors.
    """
    state_dict = paddle.load(filepath)
    logger.info("Loaded %d tensors from %s", len(state_dict), filepath)
    return state_dict


def load_model_weights(
    model: paddle.nn.Layer,
    model_path: str,
    subfolder: str = "",
    dtype: Optional[paddle.dtype] = None,
) -> None:
    """Load weights into a model from either safetensors or pdparams.

    Tries in order:
      1. model_state.pdparams (Paddle native)
      2. diffusion_pytorch_model.safetensors (HuggingFace)

    Args:
        model: The paddle.nn.Layer to load weights into.
        model_path: Root model directory.
        subfolder: Optional subfolder within model_path.
        dtype: Optional dtype to cast weights to before loading.
    """
    weight_dir = os.path.join(model_path, subfolder) if subfolder else model_path

    pdparams_path = os.path.join(weight_dir, "model_state.pdparams")
    safetensors_path = os.path.join(weight_dir, "diffusion_pytorch_model.safetensors")

    if os.path.isfile(pdparams_path):
        state_dict = load_paddle_state_dict(pdparams_path)
    elif os.path.isfile(safetensors_path):
        state_dict = load_safetensors_to_paddle(safetensors_path, dtype=dtype)
    else:
        logger.warning(
            "No weight file found in %s (tried model_state.pdparams, " "diffusion_pytorch_model.safetensors)",
            weight_dir,
        )
        return

    missing, unexpected = model.set_state_dict(state_dict)
    if missing:
        logger.warning("Missing keys when loading %s: %s", model.__class__.__name__, missing)
    if unexpected:
        logger.warning("Unexpected keys when loading %s: %s", model.__class__.__name__, unexpected)
    logger.info("Loaded weights into %s from %s", model.__class__.__name__, weight_dir)
