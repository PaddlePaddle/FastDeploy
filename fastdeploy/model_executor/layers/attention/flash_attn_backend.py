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
from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """
    FlashAttentionMetadata
    """

    max_len_kv: paddle.Tensor = None
    set_max_lengths: int = -1
    rotary_embs: Optional[paddle.Tensor] = None
    block_tables: Optional[paddle.Tensor] = None
    encoder_batch_ids: paddle.Tensor = None
    encoder_tile_ids_per_batch: paddle.Tensor = None
    encoder_num_blocks: paddle.Tensor = None
    kv_batch_ids: paddle.Tensor = None
    kv_tile_ids_per_batch: paddle.Tensor = None
    kv_num_blocks: paddle.Tensor = None
    decoder_batch_ids: paddle.Tensor = None
    decoder_tile_ids_per_batch: paddle.Tensor = None
    decoder_num_blocks: paddle.Tensor = None

    encoder_block_shape_q: Optional[paddle.Tensor] = None
    decoder_block_shape_q: Optional[paddle.Tensor] = None

    cu_seqlens_q: paddle.Tensor = None
    cu_seqlens_k: paddle.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[paddle.Tensor] = field(default_factory=list)


class FlashAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation
    """

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
    ):
        """
        FlashAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: FlashAttentionMetadata = None
        self.max_seq_len = fd_config.parallel_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.hidden_size = fd_config.model_config.hidden_size
        self.block_size = fd_config.parallel_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

    def get_attntion_meta(self):
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
    ):
        """
        Caculate kv cache shape
        """
        return (
            max_num_blocks,
            self.kv_num_heads,
            self.block_size,
            self.head_dim,
        )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = FlashAttentionMetadata()
        metadata.encoder_block_shape_q = 64
        metadata.decoder_block_shape_q = 16
        metadata.cu_seqlens_q = forward_meta.cu_seqlens_q
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.block_tables = forward_meta.block_tables
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
            metadata.encoder_block_shape_q,
            metadata.decoder_block_shape_q,
            self.num_heads // self.kv_num_heads,
            self.block_size,
            self.speculate_max_draft_token_num + 1,
        )

        (
            metadata.cu_seqlens_k,
            metadata.pre_cache_batch_ids,
            metadata.pre_cache_tile_ids_per_batch,
            metadata.pre_cache_num_blocks_cpu,
            metadata.kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            metadata.set_max_lengths[2],
            self.block_size,
        )

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
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
        self.attention_metadata = metadata
        forward_meta.decoder_batch_ids.copy_(metadata.decoder_batch_ids, False)
        forward_meta.decoder_tile_ids_per_batch.copy_(metadata.decoder_tile_ids_per_batch, False)

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
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        q, k, v, _ = gqa_rope_write_cache(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            metadata.rotary_embs,
            forward_meta.seq_lens_this_time,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.padding_offset,
            forward_meta.cum_offsets,
            metadata.block_tables,
            metadata.kv_batch_ids,
            metadata.kv_tile_ids_per_batch,
            metadata.kv_num_blocks,
            metadata.pre_cache_batch_ids,
            metadata.pre_cache_tile_ids_per_batch,
            metadata.pre_cache_num_blocks_cpu,
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            metadata.kv_signal_data_list[layer.layer_id],
            metadata.kv_token_num_cpu[0],
            self.max_seq_len,
            getattr(layer, "cache_quant_type_str", "none"),
        )
        res = flash_attention_v3_varlen(
            q,
            k,
            v,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            max_seqlen_q=metadata.set_max_lengths[0],
            max_seqlen_k=metadata.set_max_lengths[3],
            causal=self.causal,
        )[0].reshape([-1, self.hidden_size])
        return res
