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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle

try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except:
    flash_attention_v3_varlen = None

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.gpu import (
    dynamic_quant_cache_decoder_attention,
    dynamic_quant_cache_write_decoder,
    dynamic_quant_cache_write_encoder,
    dynamic_quant_get_kv_from_cache,
    flash_attention_mask,
    get_qk_tokens_num,
    split_qkv_and_rope,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class DynamciQuantCacheAttentionMetadata(AttentionMetadata):
    """
    DynamciQuantCacheAttentionMetadata
    """

    q_input: paddle.Tensor = None
    k_input: paddle.Tensor = None
    v_input: paddle.Tensor = None
    cu_seqlens_k: paddle.Tensor = None
    q_tokens_num: int = 0
    k_tokens_num: int = 0
    max_enc_len_this_time: int = 0
    max_dec_len_this_time: int = 0


class DynamciQuantCacheAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: DynamciQuantCacheAttentionMetadata
    flash_attn_func: callable = None

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        """
        FlashAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: DynamciQuantCacheAttentionMetadata = None
        self.max_seq_len = fd_config.model_config.max_model_len
        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.attn_block_m = 128
        assert self.block_size == 64

    def get_attntion_meta(self):
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        assert kv_cache_quant_type == "dynamic_int2_zp"
        return (
            max_num_blocks,
            self.kv_num_heads,
            self.block_size // 4 + self.block_size // 32 * 4,
            self.head_dim,
        )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = DynamciQuantCacheAttentionMetadata()

        metadata.cu_seqlens_k, qk_token_num = get_qk_tokens_num(
            forward_meta.seq_lens_encoder, forward_meta.seq_lens_this_time, forward_meta.seq_lens_decoder
        )
        metadata.max_enc_len_this_time = qk_token_num[0]
        metadata.max_dec_len_this_time = qk_token_num[1]
        q_token_num = qk_token_num[2]
        k_token_num = qk_token_num[3]
        metadata.q_tokens_num = q_token_num
        metadata.k_tokens_num = k_token_num

        metadata.q_input = paddle.zeros(
            [q_token_num + self.attn_block_m, self.num_heads * self.head_dim], dtype="float16"
        )
        metadata.k_input = paddle.zeros(
            [k_token_num + self.attn_block_m, self.kv_num_heads * self.head_dim], dtype="float16"
        )
        metadata.v_input = paddle.zeros(
            [k_token_num + self.attn_block_m, self.kv_num_heads * self.head_dim], dtype="float16"
        )
        self.attention_metadata = metadata

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ):
        out = paddle.zeros([qkv.shape[0], self.num_heads * self.head_dim], dtype=qkv.dtype)

        metadata = self.attention_metadata
        if metadata.max_enc_len_this_time > 0:
            split_qkv_and_rope(
                qkv,
                metadata.q_input,
                metadata.k_input,
                metadata.v_input,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.cu_seqlens_q,
                metadata.cu_seqlens_k,
                layer.qkv_bias,
                self.num_heads,
                self.kv_num_heads,
                layer.head_dim,
                metadata.max_enc_len_this_time,
                self.max_seq_len,
                getattr(layer, "cache_quant_type_str", "none"),
            )

            dynamic_quant_cache_write_encoder(
                metadata.k_input,
                metadata.v_input,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                layer.cache_k_c16,
                layer.cache_v_c16,
                metadata.cu_seqlens_k,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.block_tables,
                forward_meta.prompt_lens,
                layer.c16_remain_seq_len,
                self.num_heads,
                self.kv_num_heads,
                layer.head_dim,
                metadata.max_enc_len_this_time,
                getattr(layer, "cache_quant_type_str", "none"),
            )

            if metadata.q_tokens_num < metadata.k_tokens_num:
                dynamic_quant_get_kv_from_cache(
                    metadata.k_input,
                    metadata.v_input,
                    forward_meta.caches[2 * layer.layer_id],
                    forward_meta.caches[2 * layer.layer_id + 1],
                    layer.cache_k_c16,
                    layer.cache_v_c16,
                    metadata.cu_seqlens_k,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.block_tables,
                    forward_meta.prompt_lens,
                    layer.c16_remain_seq_len,
                    self.num_heads,
                    self.kv_num_heads,
                    layer.head_dim,
                    metadata.max_enc_len_this_time + metadata.max_dec_len_this_time,
                    getattr(layer, "cache_quant_type_str", "none"),
                )

            flash_attention_mask(
                metadata.q_input,
                metadata.k_input,
                metadata.v_input,
                forward_meta.cu_seqlens_q,
                metadata.cu_seqlens_k,
                forward_meta.seq_lens_encoder,
                out,
                None,
                self.num_heads,
                self.kv_num_heads,
                self.head_dim,
                self.max_seq_len,
                metadata.max_enc_len_this_time,
                metadata.max_dec_len_this_time,
            )

        if metadata.max_dec_len_this_time > 0:
            q_input = dynamic_quant_cache_write_decoder(
                qkv,
                forward_meta.rotary_embs,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                layer.cache_k_c16,
                layer.cache_v_c16,
                forward_meta.cu_seqlens_q,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.block_tables,
                forward_meta.step_idx,
                layer.qkv_bias,
                layer.c16_remain_seq_len,
                self.num_heads,
                self.kv_num_heads,
                layer.head_dim,
                self.max_seq_len,
                getattr(layer, "cache_quant_type_str", "none"),
            )[0]

            dynamic_quant_cache_decoder_attention(
                q_input,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                layer.cache_k_c16,
                layer.cache_v_c16,
                out,
                forward_meta.cu_seqlens_q,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.block_tables,
                layer.c16_remain_seq_len,
                self.num_heads,
                self.kv_num_heads,
                layer.head_dim,
                metadata.max_dec_len_this_time,
                self.max_seq_len,
                getattr(layer, "cache_quant_type_str", "none"),
            )
        return out
