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

import paddle


def native_top_p_sampling(probs: paddle.Tensor, top_p: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    sorted_indices = paddle.argsort(probs, descending=True)
    sorted_probs = paddle.sort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
    if probs.shape[0] != top_p.shape[0]:
        top_p = paddle.slice(top_p, [0], [0], [probs.shape[0]])
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1]

    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(),
        sorted_indices.flatten(),
        sorted_indices_to_remove.flatten(),
    )

    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)

    return None, next_tokens
