"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn

from fastdeploy.model_executor.utils import h2d_copy


class GELUActivation(nn.Layer):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + paddle.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class Projector(nn.Layer):

    def __init__(self, text_config, vision_config, prefix=""):
        super().__init__()
        self.prefix_name = prefix
        self.text_config = text_config
        self.vision_config = vision_config
        self.merge_kernel_size = (2, 2)

        self.hidden_size = self.vision_config.hidden_size * self.merge_kernel_size[0] * self.merge_kernel_size[1]

        self.pre_norm = nn.LayerNorm(self.vision_config.hidden_size, epsilon=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_1.weight.weight_loader = self.weight_loader
        self.act = GELUActivation()
        self.linear_2 = nn.Linear(self.hidden_size, self.text_config.hidden_size)
        self.linear_2.weight.weight_loader = self.weight_loader

    def _build_merge_permutation(self, image_grid_thw):
        m1, m2 = self.merge_kernel_size
        if isinstance(image_grid_thw, paddle.Tensor):
            image_grid_thw = image_grid_thw.cpu().numpy()

        merge_indices = []
        merge_lengths = []
        start = 0
        for image_grid in image_grid_thw:
            t, h, w = map(int, image_grid)
            if h % m1 != 0 or w % m2 != 0:
                raise ValueError(
                    f"grid {image_grid} is not divisible by merge_kernel_size {self.merge_kernel_size}"
                )
            local = np.arange(t * h * w, dtype=np.int64).reshape((t, h // m1, m1, w // m2, m2))
            local = local.transpose((0, 1, 3, 2, 4)).reshape(-1)
            merge_indices.append(local + start)
            merge_lengths.append(t * (h // m1) * (w // m2))
            start += t * h * w

        if len(merge_indices) == 0:
            return np.empty((0,), dtype=np.int64), merge_lengths
        return np.concatenate(merge_indices, axis=0), merge_lengths

    def forward(self, image_features, image_grid_thw, return_packed: bool = False):
        if isinstance(image_features, (list, tuple)):
            packed_image_features = (
                image_features[0] if len(image_features) == 1 else paddle.concat(image_features, axis=0)
            )
            packed_image_features = self.pre_norm(packed_image_features)
            merge_indices, merge_lengths = self._build_merge_permutation(image_grid_thw)
            merge_indices = paddle.to_tensor(merge_indices, dtype="int64", place=packed_image_features.place)
            packed_image_features = paddle.index_select(packed_image_features, merge_indices, axis=0)
            hidden_states = paddle.reshape(packed_image_features, [-1, self.hidden_size])
            hidden_states = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            if return_packed:
                return hidden_states
            return list(paddle.split(hidden_states, merge_lengths, axis=0))

        dim = image_features.shape[-1]
        image_features = paddle.reshape(image_features, [-1, dim])
        hidden_states = self.pre_norm(image_features)
        hidden_states = paddle.reshape(hidden_states, [-1, self.hidden_size])
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        loaded_weight = loaded_weight.transpose([1, 0])
        if not param._is_initialized():
            param.initialize()
        assert param.shape == loaded_weight.shape, (
            f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
        )
        # Ensure loaded weight dtype matches model param dtype
        if loaded_weight.dtype != param.dtype:
            if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.view(param.dtype)
            else:
                loaded_weight = loaded_weight.cast(param.dtype)
        h2d_copy(param, loaded_weight)
