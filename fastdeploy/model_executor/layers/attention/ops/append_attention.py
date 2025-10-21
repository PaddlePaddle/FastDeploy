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
        append_attention as append_attention_gpu,
    )
    from fastdeploy.model_executor.ops.gpu import (
        append_attention_with_output as append_attention_with_output_gpu,
    )


def append_attention(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    block_tables: paddle.Tensor,
    encoder_batch_ids: paddle.Tensor,
    encoder_tile_ids_per_batch: paddle.Tensor,
    encoder_num_blocks: paddle.Tensor,
    kv_batch_ids: paddle.Tensor,
    kv_tile_ids_per_batch: paddle.Tensor,
    kv_num_blocks: paddle.Tensor,
    decoder_batch_ids: paddle.Tensor,
    decoder_tile_ids_per_batch: paddle.Tensor,
    decoder_num_blocks: paddle.Tensor,
    set_max_lengths: paddle.Tensor,
    rotary_embs: Optional[paddle.Tensor] = None,
    attn_mask: Optional[paddle.Tensor] = None,
    qkv_bias: Optional[paddle.Tensor] = None,
    qkv_scale: Optional[paddle.Tensor] = None,
    k_quant_scale: Optional[paddle.Tensor] = None,
    v_quant_scale: Optional[paddle.Tensor] = None,
    k_dequant_scale: Optional[paddle.Tensor] = None,
    v_dequant_scale: Optional[paddle.Tensor] = None,
    cache_k_zp: Optional[paddle.Tensor] = None,
    cache_v_zp: Optional[paddle.Tensor] = None,
    linear_shift: Optional[paddle.Tensor] = None,
    linear_smooth: Optional[paddle.Tensor] = None,
    mask_offset: Optional[paddle.Tensor] = None,
    kv_signal_data: Optional[paddle.Tensor] = None,
    q_norm_weight: Optional[paddle.Tensor] = None,
    k_norm_weight: Optional[paddle.Tensor] = None,
    sinks: Optional[paddle.Tensor] = None,
    rms_norm_eps: float = 1e-6,
    compute_type: str = "bf16",
    cache_quant_type: str = "none",
    use_neox_rotary_style: bool = False,
    rope_3d: bool = False,
    max_input_length: int = 0,
    quant_max_bound: float = 0.0,
    quant_min_bound: float = 0.0,
    out_linear_in_scale: float = -1.0,
    encoder_block_shape_q: int = 64,
    decoder_block_shape_q: int = 16,
    max_partition_size: int = 32768,
    encoder_max_partition_size: int = 32768,
    speculate_max_draft_token_num: int = 1,
    causal: bool = True,
    speculate_decoder: bool = False,
    sliding_window: int = 0,
) -> paddle.Tensor:
    """
    append_attention
    """
    if current_platform.is_cuda():
        out = append_attention_gpu(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            set_max_lengths,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_scale,
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            cache_k_zp,
            cache_v_zp,
            linear_shift,
            linear_smooth,
            mask_offset,
            kv_signal_data,
            q_norm_weight,
            k_norm_weight,
            sinks,
            rms_norm_eps,
            compute_type,
            cache_quant_type,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            quant_max_bound,
            quant_min_bound,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            speculate_decoder,
            sliding_window,
        )
        return out
    else:
        raise NotImplementedError


# TODO: (mengyuan) merge w/o output version append attention after
#       finishing developing sub-graph cudagraph capture to reduce
#       compilation volume
def append_attention_with_output(
    qkv: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    block_tables: paddle.Tensor,
    encoder_batch_ids: paddle.Tensor,
    encoder_tile_ids_per_batch: paddle.Tensor,
    encoder_num_blocks: paddle.Tensor,
    kv_batch_ids: paddle.Tensor,
    kv_tile_ids_per_batch: paddle.Tensor,
    kv_num_blocks: paddle.Tensor,
    decoder_batch_ids: paddle.Tensor,
    decoder_tile_ids_per_batch: paddle.Tensor,
    decoder_num_blocks: paddle.Tensor,
    set_max_lengths: paddle.Tensor,
    out: paddle.tensor,  # attention output
    rotary_embs: Optional[paddle.Tensor] = None,
    attn_mask: Optional[paddle.Tensor] = None,
    qkv_bias: Optional[paddle.Tensor] = None,
    qkv_scale: Optional[paddle.Tensor] = None,
    k_quant_scale: Optional[paddle.Tensor] = None,
    v_quant_scale: Optional[paddle.Tensor] = None,
    k_dequant_scale: Optional[paddle.Tensor] = None,
    v_dequant_scale: Optional[paddle.Tensor] = None,
    cache_k_zp: Optional[paddle.Tensor] = None,
    cache_v_zp: Optional[paddle.Tensor] = None,
    linear_shift: Optional[paddle.Tensor] = None,
    linear_smooth: Optional[paddle.Tensor] = None,
    mask_offset: Optional[paddle.Tensor] = None,
    kv_signal_data: Optional[paddle.Tensor] = None,
    q_norm_weight: Optional[paddle.Tensor] = None,
    k_norm_weight: Optional[paddle.Tensor] = None,
    sinks: Optional[paddle.Tensor] = None,
    rms_norm_eps: float = 1e-6,
    compute_type: str = "bf16",
    cache_quant_type: str = "none",
    use_neox_rotary_style: bool = False,
    rope_3d: bool = False,
    max_input_length: int = 0,
    quant_max_bound: float = 0.0,
    quant_min_bound: float = 0.0,
    out_linear_in_scale: float = -1.0,
    encoder_block_shape_q: int = 64,
    decoder_block_shape_q: int = 16,
    max_partition_size: int = 32768,
    encoder_max_partition_size: int = 32768,
    speculate_max_draft_token_num: int = 1,
    causal: bool = True,
    speculate_decoder: bool = False,
    sliding_window: int = 0,
) -> None:
    """
    append_attention
    """
    if current_platform.is_cuda():
        return append_attention_with_output_gpu(
            qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            set_max_lengths,
            out,
            rotary_embs,
            attn_mask,
            qkv_bias,
            qkv_scale,
            k_quant_scale,
            v_quant_scale,
            k_dequant_scale,
            v_dequant_scale,
            cache_k_zp,
            cache_v_zp,
            linear_shift,
            linear_smooth,
            mask_offset,
            kv_signal_data,
            q_norm_weight,
            k_norm_weight,
            sinks,
            rms_norm_eps,
            compute_type,
            cache_quant_type,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            quant_max_bound,
            quant_min_bound,
            out_linear_in_scale,
            encoder_block_shape_q,
            decoder_block_shape_q,
            max_partition_size,
            encoder_max_partition_size,
            speculate_max_draft_token_num,
            causal,
            speculate_decoder,
            sliding_window,
        )
    else:
        raise NotImplementedError
