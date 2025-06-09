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
from typing import TYPE_CHECKING, Optional

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention, get_block_shape_and_split_kv_block)

if TYPE_CHECKING:
    from paddle._typing.dtype_like import _DTypeLiteral

from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import \
    AttentionBackend
from fastdeploy.worker.model_runner import ForwardMeta


@dataclass
class AppendAttentionMetadata:
    """
    AppendAttentionMetadata
    """
    max_len_kv: paddle.Tensor = None
    set_max_lengths: int = -1
    encoder_batch_ids: paddle.Tensor = None
    encoder_tile_ids_per_batch: paddle.Tensor = None
    encoder_num_blocks: paddle.Tensor = None
    kv_batch_ids: paddle.Tensor = None
    kv_tile_ids_per_batch: paddle.Tensor = None
    kv_num_blocks: paddle.Tensor = None
    decoder_batch_ids: paddle.Tensor = None
    decoder_tile_ids_per_batch: paddle.Tensor = None
    decoder_num_blocks: paddle.Tensor = None

    _dtype: _DTypeLiteral = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    encoder_block_shape_q: Optional[paddle.Tensor] = None
    decoder_block_shape_q: Optional[paddle.Tensor] = None
    _fuse_kernel_compute_dtype: str = "bf16"


class AppendAttentionBackend(AttentionBackend):
    """
    AppendAttentionBackend backend implementation.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
    ):
        """
        AppendAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: AppendAttentionMetadata = None
        self.block_size = model_runner.args.block_size
        self.max_seq_len = model_runner.args.max_model_len
        self.rope_theta = (10000.0 if model_runner.model_cfg.rope_theta is None
                           else model_runner.model_cfg.rope_theta)
        self.rope_3d = getattr(model_runner.model_cfg, "rope_3d", False)
        self.causal = getattr(model_runner.model_cfg, "causal", True)
        self.speculate_method = model_runner.args.speculate_method
        self.speculate_max_draft_token_num = model_runner.args.speculate_max_draft_tokens
        self.num_heads = model_runner.model_cfg.num_attention_heads // model_runner.nranks
        self.kv_num_heads = int(
            model_runner.model_cfg.num_key_value_heads) // model_runner.nranks

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = AppendAttentionMetadata()
        metadata.encoder_block_shape_q = 64
        metadata.decoder_block_shape_q = 16
        metadata.max_partition_size = 32768
        metadata.encoder_max_partition_size = 32768
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"
        metadata.block_tables = forward_meta.block_tables
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.attn_mask = forward_meta.attn_mask
        metadata.pre_caches_length = forward_meta.pre_caches_length
        (
            metadata.encoder_batch_ids,
            metadata.encoder_tile_ids_per_batch,
            metadata.encoder_num_blocks,
            metadata.kv_batch_ids,
            metadata.kv_tile_ids_per_batch,
            metadata.kv_num_blocks,
            metadata.decoder_batch_ids,
            metadata.decoder_tile_ids_per_batch,
            metadata.decoder_num_blocks,
            metadata.max_len_kv,
            metadata.set_max_lengths,
        ) = get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.cum_offsets,
            metadata.encoder_block_shape_q,
            metadata.decoder_block_shape_q,
            self.num_heads // self.kv_num_heads,
            self.block_size,
            self.speculate_max_draft_token_num + 1,
        )
        self.attention_metadata = metadata

    def get_attntion_meta(self):
        """get_attntion_meta"""
        return self.attention_metadata

    @staticmethod
    def get_kv_cache_shape(
        max_num_blocks: int,
        block_size: int,
        kv_num_head: int,
        head_dim: int,
    ):
        """
        get_kv_cache_shape
        """
        return (max_num_blocks, kv_num_head, block_size, head_dim)

    def forward_mixed(
        self,
        q,
        k,
        v,
        qkv,
        layer: Attention,
        forward_meta: ForwardMeta,
    ):
        """
        forward_mixed
        """
        metadata = self.attention_metadata

        res = append_attention(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.padding_offset,
            forward_meta.cum_offsets,
            metadata.block_tables,
            metadata.encoder_batch_ids,
            metadata.encoder_tile_ids_per_batch,
            metadata.encoder_num_blocks,
            metadata.kv_batch_ids,
            metadata.kv_tile_ids_per_batch,
            metadata.kv_num_blocks,
            metadata.decoder_batch_ids,
            metadata.decoder_tile_ids_per_batch,
            metadata.decoder_num_blocks,
            metadata.set_max_lengths,
            metadata.max_len_kv,
            metadata.rotary_embs,
            metadata.attn_mask,
            layer.qkv_bias,
            layer.qkv_scale,
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            layer.linear_shift,
            layer.linear_smooth,
            None,  # kv_signal_data,
            metadata._fuse_kernel_compute_dtype,
            getattr(layer, "cache_quant_type_str", "none"),
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            getattr(layer, "out_scale", -1.0),
            metadata.encoder_block_shape_q,
            metadata.decoder_block_shape_q,
            metadata.max_partition_size,
            metadata.encoder_max_partition_size,
            self.speculate_max_draft_token_num + 1,
            self.causal,
            self.speculate_method is not None,
        )[0]

        return res
