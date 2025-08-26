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


def gqa_rope_write_cache(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    rotary_embs: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    block_tables: paddle.Tensor,
    kv_batch_ids: paddle.Tensor,
    kv_tile_ids_per_batch: paddle.Tensor,
    kv_num_blocks: paddle.Tensor,
    cache_batch_ids: paddle.Tensor,
    cache_tile_ids_per_batch: paddle.Tensor,
    cache_num_blocks: paddle.Tensor,
    cache_k_quant_scales: Optional[paddle.Tensor] = None,
    cache_v_quant_scales: Optional[paddle.Tensor] = None,
    cache_k_dequant_scales: Optional[paddle.Tensor] = None,
    cache_v_dequant_scales: Optional[paddle.Tensor] = None,
    cache_k_zp: Optional[paddle.Tensor] = None,
    cache_v_zp: Optional[paddle.Tensor] = None,
    kv_signal_data: Optional[paddle.Tensor] = None,
    kv_token_num: int = 1,
    max_seq_len: int = 0,
    cache_quant_type: str = "none",
    rope_3d: bool = False,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import gqa_rope_write_cache

        q, k, v, qkv_ = gqa_rope_write_cache(
            qkv,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            rotary_embs,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            cache_batch_ids,
            cache_tile_ids_per_batch,
            cache_num_blocks,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_dequant_scales,
            cache_v_dequant_scales,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            kv_token_num,
            max_seq_len,
            cache_quant_type,
            rope_3d,
        )
        return q, k, v, qkv_
    else:
        raise NotImplementedError
