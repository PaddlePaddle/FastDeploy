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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    init_signal_layerwise, open_shm_and_get_meta_signal)

if TYPE_CHECKING:
    from paddle._typing.dtype_like import _DTypeLiteral

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend, AttentionMetadata)
from fastdeploy.worker.forward_meta import ForwardMeta


@dataclass
class XPUAttentionMetadata(AttentionMetadata):
    """
    XPUAttentionMetadata
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

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[paddle.Tensor] = field(default_factory=list)


class XPUAttentionBackend(AttentionBackend):
    """
    XPUAttentionBackend backend implementation.
    """

    def __init__(self, fd_config: FDConfig, kv_num_heads: int, num_heads: int,
                 head_dim: int):
        """
        XPUAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: XPUAttentionMetadata = None
        # TODO(gongshaotian): Use fd_config parameters in the correct location
        self.block_size: int = fd_config.parallel_config.block_size
        self.max_seq_len: int = fd_config.parallel_config.max_model_len
        self.rope_theta: float = (10000.0
                                  if fd_config.model_config.rope_theta is None
                                  else fd_config.model_config.rope_theta)
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        # self.speculate_method = fd_config.parallel_config.speculate_method
        # self.use_speculate = self.speculate_method is not None
        # self.speculate_max_draft_token_num = fd_config.parallel_config.speculate_max_draft_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.rank: int = fd_config.parallel_config.tensor_parallel_rank

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.num_layers: int = fd_config.model_config.num_layers

        # pd_disaggregation
        self.use_pd_disaggregation: int = int(
            os.getenv("FLAGS_use_pd_disaggregation", 0))
        self.start_layer_index: int = fd_config.model_config.start_layer_index

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = XPUAttentionMetadata()
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

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.use_pd_disaggregation:
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, self.keep_pd_step_flag)
        self.attention_metadata: AttentionMetadata = metadata

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
    ) -> Tuple[int, int, int, int]:
        """
        Caculate kv cache shape
        """
        return (max_num_blocks, self.kv_num_heads, self.block_size,
                self.head_dim)

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        forward_mixed
        """
        metadata = self.attention_metadata

        if self.use_pd_disaggregation:
            metadata.kv_signal_data_list[
                layer.layer_id] = init_signal_layerwise(
                    metadata.kv_signal_metadata,
                    layer.layer_id + self.start_layer_index)

        k_quant_scale = getattr(layer, "cache_k_scale", None)
        v_quant_scale = getattr(layer, "cache_v_scale", None)

        from fastdeploy.model_executor.ops.xpu import block_attn
        res = block_attn(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.cum_offsets,
            metadata.rotary_embs,
            metadata.block_tables,
            None,
            k_quant_scale,
            v_quant_scale,
            forward_meta.enc_batch,
            forward_meta.dec_batch,
            forward_meta.total_enc_len,
            forward_meta.encoder_seq_lod_cpu,
            forward_meta.encoder_batch_map_cpu,
            forward_meta.decoder_context_len_cpu,
            forward_meta.decoder_batch_map_cpu,
        )
        return res
