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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.gcu import (
    fused_rotary_embedding,
    mem_efficient_attention,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class GCUMemEfficientAttnMetadata(AttentionMetadata):
    """
    GCUMemEfficientAttnMetadata
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


class GCUMemEfficientAttnBackend(AttentionBackend):
    """
    GCUMemEfficientAttnBackend backend implementation.
    """

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
        GCUMemEfficientAttnBackend __init__
        """
        super().__init__()
        self.attention_metadata: GCUMemEfficientAttnMetadata = None
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
        self.use_paddle_native_sdpa = False

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = GCUMemEfficientAttnMetadata()

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
        query_lens = []
        cached_kv_lens = []
        cached_kv_slot_range = []
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
                query_lens.append(lens_this_time)
                cached_kv_lens.append(end)
                cached_kv_slot_range.append(
                    [
                        self.all_slot_mapping[seq_idx][0],
                        self.all_slot_mapping[seq_idx][end],
                    ]
                )

        self.block_tables = paddle.to_tensor(block_tables, dtype="int32")
        self.slot_mapping = paddle.to_tensor(slot_mapping, dtype="int32")
        self.cache_slot_range = paddle.to_tensor(cache_slot_range, dtype="int32")
        self.position_ids = paddle.to_tensor(position_ids, dtype="int32")
        self.position_ids = self.position_ids.reshape_((1, -1))

        logger.info(f"[FD_DEBUG] init_attention_metadata, self.position_ids:\n{self.position_ids}")

        cu_query_lens_data = [0]
        for seq_idx in range(num_seqs):
            if self.seq_lens_this_time_list[seq_idx] != 0:
                cu_query_lens_data.append(self.seq_lens_this_time_list[seq_idx])
        cu_query_lens = np.array(cu_query_lens_data, dtype=np.int32).cumsum(axis=0)

        self.cu_query_lens = paddle.to_tensor(cu_query_lens, dtype="int32")
        self.seqused_k = paddle.to_tensor(cache_lens, dtype="int32")
        self.max_seqlen_q = self.max_seq_len_this_time
        self.max_seqlen_k = np.max(cache_lens)

        self.query_lens = query_lens
        self.cached_kv_lens = cached_kv_lens
        self.cached_kv_slot_range = cached_kv_slot_range

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

        q_start = 0
        result = paddle.empty_like(query)
        for idx in range(len(self.query_lens)):
            q_end = q_start + self.query_lens[idx]
            kv_start = self.cached_kv_slot_range[idx][0]
            kv_end = self.cached_kv_slot_range[idx][1]

            q_ = query[q_start:q_end, :, :]
            k_ = key_caches[kv_start:kv_end, :, :]
            v_ = value_caches[kv_start:kv_end, :, :]

            if self.use_paddle_native_sdpa:
                res = self.native_sdpa_impl(q_, k_, v_)
            else:
                res = mem_efficient_attention(
                    query=q_.unsqueeze(0),
                    key=k_.unsqueeze(0),
                    value=v_.unsqueeze(0),
                    attn_mask=None,
                    dropout=0.0,
                    softmax_scale=self.scaling,
                    mask_mode=1,
                    seqlens=[0],
                    causal=self.causal,
                )
            result[q_start:q_end, :, :] = res
            q_start = q_end
        result = result.reshape_((token_num, -1))
        return result

    def get_triangle_upper_mask(self, shape, dtype):
        #  [batch_size, 1, q_seq_len, kv_seq_len]
        shape[1] = 1
        q_seq_len = shape[2]
        kv_seq_len = shape[3]
        paddle_dtype = dtype  # paddle.base.data_feeder.convert_dtype(dtype)
        mask = paddle.full(shape, paddle.finfo(paddle_dtype).min, dtype=paddle_dtype)
        mask = paddle.triu(mask, diagonal=kv_seq_len - q_seq_len + 1)
        return mask

    def native_sdpa_impl(self, query, key, value):
        # input shape: [num_tokens, num_heads, head_dim] -> [1, num_tokens, num_heads, head_dim]
        q = query.unsqueeze(0)
        k = key.unsqueeze(0)
        v = value.unsqueeze(0)
        batch, q_seq_len, heads, head_dim = q.shape
        kv_seq_len = k.shape[1]

        #  [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = paddle.transpose(q, [0, 2, 1, 3])
        k = paddle.transpose(k, [0, 2, 1, 3])
        v = paddle.transpose(v, [0, 2, 1, 3])

        # GQA
        if q.shape[1] != k.shape[1]:
            kv_head = k.shape[1]

            k = k.reshape([batch, kv_head, 1, kv_seq_len, head_dim])
            k = paddle.tile(k, [1, 1, heads // kv_head, 1, 1])
            k = k.reshape([batch, heads, kv_seq_len, head_dim])

            v = v.reshape([batch, kv_head, 1, kv_seq_len, head_dim])
            v = paddle.tile(v, [1, 1, heads // kv_head, 1, 1])
            v = v.reshape([batch, heads, kv_seq_len, head_dim])

        # matmul and devide by sqrt(head_dim)
        attn_weights = paddle.matmul(q / math.sqrt(head_dim), k.transpose([0, 1, 3, 2]))

        attention_mask = self.get_triangle_upper_mask([batch, 1, q_seq_len, kv_seq_len], q.dtype)
        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1, dtype="float32").astype(q.dtype)

        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        return attn_output.squeeze(0)
