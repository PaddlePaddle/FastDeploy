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

import os
from typing import Optional, Tuple, Union

import paddle
from paddle import Tensor

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)


def flash_attn_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    fixed_seed_offset: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    return_softmax: bool = False,
    is_test: bool = True,
    rng_name: str = "",
) -> Union[Tensor, Tuple[Tensor, ...]]:
    return paddle._C_ops.flash_attn(
        q, k, v, fixed_seed_offset, attn_mask, dropout_prob, causal, return_softmax, is_test, rng_name
    )


def flash_attn_unpadded_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    fixed_seed_offset: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    softmax_scale: float = 1.0,
    dropout: float = 0.0,
    causal: bool = False,
    return_softmax: bool = False,
    is_test: bool = True,
    rng_name: str = "",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    outputs = paddle._C_ops.flash_attn_unpadded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        fixed_seed_offset,
        attn_mask,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        dropout,
        causal,
        return_softmax,
        is_test,
        rng_name,
    )
    return outputs


def flash_attn_kvcache_func(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    seqlens_k: Tensor,
    block_table: Tensor,
    k: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    rotary_cos: Optional[Tensor] = None,
    rotary_sin: Optional[Tensor] = None,
    cache_batch_idx: Optional[Tensor] = None,
    causal: bool = True,
    is_rotary_interleaved: bool = False,
    num_splits: int = 1,
    dropout: float = 0.0,
    return_softmax: bool = False,
) -> Tuple[Tensor, Tensor]:
    out, softmax_lse = paddle._C_ops._run_custom_op(
        "flash_attn_kvcache",
        q,
        k_cache,
        v_cache,
        k,
        v,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        causal,
        is_rotary_interleaved,
        num_splits,
        dropout,
        return_softmax,
    )
    return out, softmax_lse
