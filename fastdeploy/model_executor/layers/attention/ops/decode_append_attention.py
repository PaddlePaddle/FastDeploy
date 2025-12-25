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

from typing import Optional

import paddle

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        decode_append_attention as decode_append_attention_cuda,
    )


def decode_append_attention(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    tmp_workspace: paddle.Tensor,
    tmp_m: paddle.Tensor,
    tmp_d: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    block_tables: paddle.Tensor,
    block_indices: paddle.Tensor,
    num_blocks: paddle.Tensor,
    chunk_size: paddle.Tensor,
    set_max_lengths: paddle.Tensor,
    attn_mask: Optional[paddle.Tensor] = None,
    k_quant_scale: Optional[paddle.Tensor] = None,
    v_quant_scale: Optional[paddle.Tensor] = None,
    k_dequant_scale: Optional[paddle.Tensor] = None,
    v_dequant_scale: Optional[paddle.Tensor] = None,
    cache_k_zp: Optional[paddle.Tensor] = None,
    cache_v_zp: Optional[paddle.Tensor] = None,
    mask_offset: Optional[paddle.Tensor] = None,
    sinks: Optional[paddle.Tensor] = None,
    cache_quant_type: str = "none",
    max_input_length: int = 0,
    quant_max_bound: float = 0.0,
    quant_min_bound: float = 0.0,
    max_tokens_per_batch: int = 1,
    causal: bool = True,
    sliding_window: int = 0,
) -> paddle.Tensor:
    """
    append_attention
    """
    if current_platform.is_cuda():
        out = decode_append_attention_cuda(
            qkv,
            key_cache,
            value_cache,
            tmp_workspace,
            tmp_m,
            tmp_d,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            block_indices,
            num_blocks,
            chunk_size,
            set_max_lengths,
            attn_mask,
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            cache_k_zp,
            cache_v_zp,
            mask_offset,
            sinks,
            cache_quant_type,
            max_input_length,
            quant_max_bound,
            quant_min_bound,
            max_tokens_per_batch,
            causal,
            sliding_window,
        )
        return out
    else:
        raise NotImplementedError
