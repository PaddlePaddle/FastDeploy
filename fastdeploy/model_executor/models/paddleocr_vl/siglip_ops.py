"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import List

import paddle
from paddleformers.transformers.activations import ACT2FN

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import fused_neox_rope_embedding, gelu_tanh


def rotate_half(x):
    Dh = x.shape[-1]
    x1 = x[..., : Dh // 2]
    x2 = x[..., Dh // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(x, cos, sin):
    orig_dtype = x.dtype
    x = x.astype("float32")
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.astype(orig_dtype)


def native_neox_rope_embedding(qkv, cos, sin, num_heads):
    B, seq_length, D = qkv.shape
    qkv = qkv.reshape(
        [
            seq_length,
            3,
            num_heads,
            -1,
        ]
    ).transpose(perm=[1, 0, 2, 3])
    q, k, v = qkv.unbind(axis=0)
    q = apply_rotary_pos_emb_vision(q, cos, sin)
    k = apply_rotary_pos_emb_vision(k, cos, sin)
    return q, k, v


def neox_rope_embedding(
    qkv: paddle.Tensor, cos_emb: paddle.Tensor, sin_emb: paddle.Tensor, num_heads: int, head_dim: int
) -> List[paddle.Tensor]:
    if current_platform.is_cuda():
        return fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    else:
        return native_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads)


def get_activation_fn(hidden_act: str):
    if hidden_act == "gelu_pytorch_tanh":
        if current_platform.is_cuda():
            return gelu_tanh
        else:
            return ACT2FN["gelu_new"]
    else:
        return ACT2FN[hidden_act]
