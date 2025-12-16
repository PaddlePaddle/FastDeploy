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


def gqa_decoder_rope_norm_with_write_cache(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    block_tables: paddle.Tensor,
    rotary_embs: Optional[paddle.Tensor] = None,
    qkv_bias: Optional[paddle.Tensor] = None,
    qkv_scale: Optional[paddle.Tensor] = None,
    k_quant_scale: Optional[paddle.Tensor] = None,
    v_quant_scale: Optional[paddle.Tensor] = None,
    k_dequant_scale: Optional[paddle.Tensor] = None,
    v_dequant_scale: Optional[paddle.Tensor] = None,
    cache_k_zp: Optional[paddle.Tensor] = None,
    cache_v_zp: Optional[paddle.Tensor] = None,
    kv_signal_data: Optional[paddle.Tensor] = None,
    q_norm_weight: Optional[paddle.Tensor] = None,
    k_norm_weight: Optional[paddle.Tensor] = None,
    rms_norm_eps: float = 1e-6,
    compute_type: str = "bf16",
    cache_quant_type: str = "none",
    use_neox_rotary_style: bool = False,
    rope_3d: bool = False,
    max_input_length: int = 0,
    speculate_decoder: bool = False,
):
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import gqa_decoder_rope_norm_with_write_cache

        gqa_decoder_rope_norm_with_write_cache(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            rotary_embs,
            qkv_bias,
            qkv_scale,
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            cache_k_zp,
            cache_v_zp,
            kv_signal_data,
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps,
            compute_type,
            cache_quant_type,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            speculate_decoder,
        )
    else:
        raise NotImplementedError
