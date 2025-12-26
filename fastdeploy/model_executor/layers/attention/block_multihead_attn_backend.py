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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import paddle

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)


@dataclass
class BlockAttentionMetadata(AttentionMetadata):
    """
    BlockAttentionMetadata
    """

    encoder_batch_ids: paddle.Tensor = None
    encoder_tile_ids_per_batch: paddle.Tensor = None
    encoder_num_blocks: paddle.Tensor = None
    kv_batch_ids: paddle.Tensor = None
    kv_tile_ids_per_batch: paddle.Tensor = None
    kv_num_blocks: paddle.Tensor = None

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)


class BlockAttentionBackend(AttentionBackend):
    """
    BlockAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: BlockAttentionBackend

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
        BlockAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: BlockAttentionMetadata = None
        self.block_size = fd_config.cache_config.block_size
        self.max_seq_len = fd_config.model_config.max_model_len
        self.rope_theta = 10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        self.rank = fd_config.parallel_config.tensor_parallel_rank

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = fd_config.model_config.head_dim

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = BlockAttentionMetadata()
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
        self.attention_metadata = metadata

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
        q,
        k,
        v,
        qkv,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ):
        """
        forward_mixed
        """
        metadata = self.attention_metadata

        res = paddle.incubate.nn.functional.block_multihead_attention(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cum_offsets,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
            metadata.block_tables,
            getattr(layer, "pre_key_cache", None),
            getattr(layer, "pre_value_cache", None),
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            layer.qkv_scale,
            layer.qkv_bias,
            layer.linear_shift,
            layer.linear_smooth,
            getattr(layer, "max_enc_len_this_time", None),
            getattr(layer, "max_dec_len_this_time", None),
            metadata.rotary_embs,
            metadata.attn_mask,
            None,  # tgt_mask
            self.max_seq_len,
            self.block_size,
            layer.use_neox_rotary_style,
            getattr(layer, "use_dynamic_cachekv_quant", False),
            quant_round_type=getattr(layer, "quant_round_type", 0),
            quant_max_bound=getattr(layer, "quant_max_bound", 0.0),
            quant_min_bound=getattr(layer, "quant_min_bound", 0.0),
            out_scale=getattr(layer, "out_scale", -1.0),
            compute_dtype=metadata._fuse_kernel_compute_dtype,
            rope_theta=self.rope_theta,
        )[0]

        return res
