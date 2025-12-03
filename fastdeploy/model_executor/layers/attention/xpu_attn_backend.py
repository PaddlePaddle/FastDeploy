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
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id


@dataclass
class XPUAttentionMetadata(AttentionMetadata):
    """
    XPUAttentionMetadata
    """

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


class XPUAttentionBackend(AttentionBackend):
    """
    XPUAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: XPUAttentionMetadata

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
        XPUAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: XPUAttentionMetadata = None
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
            fd_config.model_config, "use_3d_rope", False
        )
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # pd_disaggregation
        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.rank, self.device_id = init_rank_and_device_id(fd_config)

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = XPUAttentionMetadata()
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
        if self.pd_disaggregation_mode == "per_chunk" and not forward_meta.is_profiling:
            if not self.keep_pd_step_flag:
                init_kv_signal_per_query(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_decoder,
                    self.rank,
                    self.num_layers + self.num_layers_draft_model,
                )
        elif self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, int(self.device_id), self.keep_pd_step_flag
            )

        self.attention_metadata: AttentionMetadata = metadata

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[list, list]:
        """
        Calculate kv cache shape
        """
        key_cache_shape = value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
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
        forward_mixed
        """
        metadata = self.attention_metadata
        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        k_quant_scale = getattr(layer, "cache_k_scale", None)
        v_quant_scale = getattr(layer, "cache_v_scale", None)

        cache_k_out_scale = getattr(layer, "cache_k_out_scale", None)
        cache_v_out_scale = getattr(layer, "cache_v_out_scale", None)

        k_zp = getattr(self, "cache_k_zp", None)
        v_zp = getattr(self, "cache_v_zp", None)

        from fastdeploy.model_executor.ops.xpu import block_attn

        res = block_attn(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.cum_offsets,
            metadata.rotary_embs,
            metadata.block_tables,
            forward_meta.prefix_block_tables,
            forward_meta.len_info_cpu,
            forward_meta.encoder_seq_lod_cpu,
            forward_meta.decoder_seq_lod_cpu,
            forward_meta.encoder_kv_lod_cpu,
            forward_meta.encoder_batch_map_cpu,
            forward_meta.decoder_context_len_cpu,
            forward_meta.decoder_context_len_cache_cpu,
            forward_meta.decoder_batch_map_cpu,
            forward_meta.prefix_len_cpu,
            k_quant_scale,
            v_quant_scale,
            cache_k_out_scale,
            cache_v_out_scale,
            k_zp,  # zero_point_quant_scale
            v_zp,  # zero_point_quant_scale
            None,  # shift
            None,  # smooth
            metadata.kv_signal_data_list[layer.layer_id],  # kv_signal_data
            forward_meta.kv_signal_sender,  # kv_signal_sender
            forward_meta.pos_emb_type,
            self.rope_3d,
        )

        return res
