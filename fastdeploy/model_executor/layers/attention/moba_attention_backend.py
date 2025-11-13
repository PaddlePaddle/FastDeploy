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
    from fastdeploy.model_executor.ops.gpu import get_cur_cu_seq_len_k, moba_attention
except:
    moba_attention = None
    get_cur_cu_seq_len_k = None

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)


@dataclass
class PlasAttentionMetadata(AttentionMetadata):
    """
    AppendAttentionMetadata
    """

    q_input: paddle.Tensor = None
    k_input: paddle.Tensor = None
    v_input: paddle.Tensor = None
    cu_seq_q_pack: paddle.Tensor = None
    cu_seqlens_k: paddle.Tensor = None
    q_pack_tokens: paddle.Tensor = None
    max_enc_len_this_time: int = 0
    max_dec_len_this_time: int = 0


class PlasAttentionBackend(AttentionBackend):
    """
    The backend class that uses paddle native attention implementation.
    Which is used only for testing purpose.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ) -> None:
        """
        PlasAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: PlasAttentionMetadata = None
        assert fd_config.plas_attention_config is not None, "plas_attention_config is None"
        self.block_size = fd_config.cache_config.block_size
        self.max_seq_len = fd_config.model_config.max_model_len
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.attn_block_m = 128
        self.plas_block_size = fd_config.plas_attention_config.plas_block_size
        self.plas_encoder_top_k_left = int(fd_config.plas_attention_config.plas_encoder_top_k_left)
        self.plas_encoder_top_k_right = int(fd_config.plas_attention_config.plas_encoder_top_k_right)
        self.plas_use_encoder_seq_limit = int(fd_config.plas_attention_config.plas_use_encoder_seq_limit)
        self.plas_decoder_top_k_left = int(fd_config.plas_attention_config.plas_decoder_top_k_left)
        self.plas_decoder_top_k_right = int(fd_config.plas_attention_config.plas_decoder_top_k_right)
        self.plas_use_decoder_seq_limit = int(fd_config.plas_attention_config.plas_use_decoder_seq_limit)
        self.plas_max_seq_length = fd_config.plas_attention_config.plas_max_seq_length

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Init the metadata for a forward pass."""
        metadata = PlasAttentionMetadata()
        metadata._dtype = paddle.get_default_dtype()
        metadata.cu_seq_q_pack, metadata.cu_seqlens_k, metadata.q_pack_tokens = get_cur_cu_seq_len_k(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            int(self.attn_block_m),
        )
        metadata.max_enc_len_this_time = forward_meta.seq_lens_encoder.max().cpu()
        metadata.max_dec_len_this_time = forward_meta.seq_lens_decoder.max().cpu()
        q_token_num = int(forward_meta.cu_seqlens_q[-1])
        k_token_num = int(metadata.cu_seqlens_k[-1])
        metadata.q_input = paddle.zeros(
            [q_token_num + self.attn_block_m, self.num_heads * self.head_dim], dtype=metadata._dtype
        )
        metadata.k_input = paddle.zeros(
            [k_token_num + self.attn_block_m, self.kv_num_heads * self.head_dim], dtype=metadata._dtype
        )
        metadata.v_input = paddle.zeros(
            [k_token_num + self.attn_block_m, self.kv_num_heads * self.head_dim], dtype=metadata._dtype
        )
        self.attention_metadata = metadata
        assert self.max_seq_len <= self.plas_max_seq_length

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
            value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
        return key_cache_shape, value_cache_shape

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
    ) -> paddle.Tensor:
        """
        Mixed模式的前向传播
        """
        attention_metadata = self.attention_metadata
        out = moba_attention(
            qkv,
            attention_metadata.q_input,
            attention_metadata.k_input,
            attention_metadata.v_input,
            forward_meta.cu_seqlens_q,
            attention_metadata.cu_seqlens_k,
            attention_metadata.cu_seq_q_pack,
            attention_metadata.q_pack_tokens,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.block_tables,
            forward_meta.rotary_embs,
            layer.cache_k_block_means,
            getattr(layer, "attn_gate_weight", None),
            layer.qkv_bias,
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.max_seq_len,
            attention_metadata.max_enc_len_this_time,
            attention_metadata.max_dec_len_this_time,
            self.plas_encoder_top_k_left,
            self.plas_encoder_top_k_right,
            self.plas_use_encoder_seq_limit,
            self.plas_decoder_top_k_left,
            self.plas_decoder_top_k_right,
            self.plas_use_decoder_seq_limit,
            layer.plas_use_mlp,
            getattr(layer, "cache_quant_type_str", "none"),
        )[0]
        return out
