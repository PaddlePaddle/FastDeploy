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
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import paddle

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from paddleformers.utils.log import logger

from fastdeploy.model_executor.ops.gcu import flash_attn_var_len, fused_rotary_embedding


@dataclass
class GCUFlashAttnMetadata(AttentionMetadata):
    """
    GCUFlashAttnMetadata
    """

    _dtype: paddle.dtype = paddle.bfloat16

    seq_lens_encoder: Optional[paddle.Tensor] = None
    seq_lens_decoder: Optional[paddle.Tensor] = None
    seq_lens_this_time: Optional[paddle.Tensor] = None
    batch_id_per_token: Optional[paddle.Tensor] = None

    cu_seqlens_q: Optional[paddle.Tensor] = None
    cu_seqlens_k: Optional[paddle.Tensor] = None
    caches: Optional[paddle.Tensor] = None

    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None

    pre_caches_length: int = 0


class GCUFlashAttnBackend(AttentionBackend):
    """
    GCUFlashAttnBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: GCUFlashAttnBackend

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
        GCUFlashAttnBackend __init__
        """
        super().__init__()
        self.attention_metadata: GCUFlashAttnMetadata = None
        self.block_size = fd_config.cache_config.block_size
        self.max_seq_len = fd_config.model_config.max_model_len
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs

        self.causal = getattr(fd_config.model_config, "causal", True)

        self.rank = fd_config.parallel_config.tensor_parallel_rank
        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.position_ids_base = np.arange(self.max_seq_len)

        # TODO(zhengjun): Need to adapt the allocation logic and
        # temporarily allocate according to fixed size
        self.all_block_tables: List[List[int]] = None
        self.all_slot_mapping: List[List[int]] = None

        self.rotary_embs = None
        self.enable_monitor: bool = bool(os.getenv("FD_GCU_ATTN_MONITOR", False))

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = GCUFlashAttnMetadata()

        metadata.forward_mode = forward_meta.forward_mode

        metadata._dtype = paddle.get_default_dtype()

        metadata.seq_lens_encoder = forward_meta.seq_lens_encoder
        metadata.seq_lens_decoder = forward_meta.seq_lens_decoder
        metadata.seq_lens_this_time = forward_meta.seq_lens_this_time
        metadata.batch_id_per_token = forward_meta.batch_id_per_token

        metadata.cu_seqlens_q = forward_meta.cu_seqlens_q
        metadata.cu_seqlens_k = forward_meta.cu_seqlens_k
        metadata.caches = forward_meta.caches

        # metadata.block_tables = forward_meta.block_tables
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.attn_mask = forward_meta.attn_mask  # not init

        metadata.pre_caches_length = forward_meta.pre_caches_length  # not inited

        self.attention_metadata = metadata

        if self.rotary_embs is None:
            self.rotary_embs = metadata.rotary_embs.reshape((-1, self.head_dim))

        # some info for attention
        self.seq_lens_this_time_list = forward_meta.seq_lens_this_time.tolist()  # List[int]
        self.seq_lens_encoder_list = forward_meta.seq_lens_encoder.tolist()  # List[List[int]]
        self.seq_lens_decoder_list = forward_meta.seq_lens_decoder.tolist()  # List[List[int]]
        self.seq_lens_sum = np.sum(self.seq_lens_this_time_list)
        self.max_seq_len_this_time = np.max(self.seq_lens_this_time_list)

        num_seqs = forward_meta.seq_lens_this_time.shape[0]

        self.is_decoder = all(x[0] == 0 for x in self.seq_lens_encoder_list)
        self.is_all_prefill = all(x[0] == 0 for x in self.seq_lens_decoder_list)

        # block_tables and slot_mapping
        if self.all_slot_mapping is None:
            max_num_blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
            total_blocks = max_num_blocks_per_seq * self.max_num_seqs
            self.all_block_tables = (
                np.arange(0, total_blocks, dtype=np.int32)
                .reshape((self.max_num_seqs, max_num_blocks_per_seq))
                .tolist()
            )
            self.all_slot_mapping = (
                np.arange(0, total_blocks * self.block_size, dtype=np.int32).reshape((self.max_num_seqs, -1)).tolist()
            )

        block_tables = []
        slot_mapping = []
        cache_slot_range = []
        cache_lens = []
        position_ids = []
        for seq_idx in range(num_seqs):
            cache_len = None
            if self.seq_lens_encoder_list[seq_idx][0] != 0:  # prefill
                cache_len = 0
            elif self.seq_lens_decoder_list[seq_idx][0] != 0:  # decode
                cache_len = self.seq_lens_decoder_list[seq_idx][0]
            # else:  doesn't have req in this seq_idx

            if cache_len is not None:
                lens_this_time = self.seq_lens_this_time_list[seq_idx]
                start = cache_len
                end = start + lens_this_time
                slot_mapping.extend(self.all_slot_mapping[seq_idx][start:end])
                cache_slot_range.extend(self.all_slot_mapping[seq_idx][0:end])
                cache_lens.append(end)
                block_tables.append(self.all_block_tables[seq_idx])
                position_ids.extend(self.position_ids_base[start:end])

        self.block_tables = paddle.to_tensor(block_tables, dtype="int32")
        self.slot_mapping = paddle.to_tensor(slot_mapping, dtype="int32")
        self.cache_slot_range = paddle.to_tensor(cache_slot_range, dtype="int32")
        self.position_ids = paddle.to_tensor(position_ids, dtype="int32")
        self.position_ids = self.position_ids.reshape_((1, -1))

        if self.enable_monitor:
            logger.info(f"[FD_DEBUG] init_attention_metadata, position_ids:\n{self.position_ids}")

        cu_query_lens_data = [0]
        for seq_idx in range(num_seqs):
            if self.seq_lens_this_time_list[seq_idx] != 0:
                cu_query_lens_data.append(self.seq_lens_this_time_list[seq_idx])
        cu_query_lens = np.array(cu_query_lens_data, dtype=np.int32).cumsum(axis=0)

        self.cu_query_lens = paddle.to_tensor(cu_query_lens, dtype="int32")
        self.seqused_k = paddle.to_tensor(cache_lens, dtype="int32")
        self.max_seqlen_q = self.max_seq_len_this_time
        self.max_seqlen_k = np.max(cache_lens)

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
        # [total_tokens, kv_num_heads, head_dim]
        return (
            max_num_blocks * self.block_size,
            self.kv_num_heads,
            self.head_dim,
        )

    @paddle.no_grad()
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
        """Run a forward for mixed."""
        token_num = qkv.shape[0]
        q_size = self.num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        num_or_sections = [q_size, kv_size, kv_size]
        query, key, value = paddle.split(qkv, num_or_sections=num_or_sections, axis=-1)

        query = query.reshape_((1, -1, self.num_heads, self.head_dim))
        key = key.reshape_((1, -1, self.kv_num_heads, self.head_dim))

        # 1. Rope
        if self.rotary_embs.dtype != query.dtype:
            self.rotary_embs = paddle.cast(self.rotary_embs, query.dtype)

        query, key = fused_rotary_embedding(
            query,
            key,
            self.rotary_embs,
            self.position_ids,
            layer.use_neox_rotary_style,
        )

        # 2. Save kv cache
        # shape: [total_tokens, kv_num_heads, head_dim]
        key = key.reshape_((-1, self.kv_num_heads, self.head_dim))
        value = value.reshape_((-1, self.kv_num_heads, self.head_dim))
        key_caches = forward_meta.caches[2 * layer.layer_id]
        value_caches = forward_meta.caches[2 * layer.layer_id + 1]
        key_caches[self.slot_mapping, :, :] = key
        value_caches[self.slot_mapping, :, :] = value

        # 3. calc attn
        query = query.reshape_((-1, self.num_heads, self.head_dim))
        key_caches = key_caches.reshape((-1, self.block_size, self.kv_num_heads, self.head_dim))
        value_caches = value_caches.reshape((-1, self.block_size, self.kv_num_heads, self.head_dim))
        res = flash_attn_var_len(
            query=query,
            key=key_caches,
            value=value_caches,
            cu_seqlens_q=self.cu_query_lens,
            cu_seqlens_k=None,
            seqused_k=self.seqused_k,
            leftpad_k=None,
            block_table=self.block_tables,
            alibi_slopes=None,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            p_dropout=0.0,
            softmax_scale=self.scaling,
            zero_tensors=False,
            is_causal=self.causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            return_softmax=False,
        )
        res = res.reshape_((token_num, -1))
        return res
