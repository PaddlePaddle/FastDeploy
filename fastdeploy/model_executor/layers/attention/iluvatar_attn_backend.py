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
from math import sqrt
from typing import TYPE_CHECKING, Optional

import paddle

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.iluvatar import (
    mixed_fused_paged_attention,
    paged_attention,
    prefill_fused_paged_attention,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class IluvatarAttentionMetadata(AttentionMetadata):
    """
    IluvatarAttentionMetadata
    """

    alibi_slopes: Optional[paddle.Tensor] = None
    window_left: int = -1
    window_right: int = -1
    softcap: float = 0.0
    use_cuda_graph: bool = False
    use_sqrt_alibi: bool = False
    prefill_rope_cos: paddle.Tensor = None
    prefill_rope_sin: paddle.Tensor = None
    prefill_cu_seqlens_q: paddle.Tensor = None
    prefill_block_tables: paddle.Tensor = None
    decode_rope_cos: paddle.Tensor = None
    decode_rope_sin: paddle.Tensor = None
    decode_seq_lens: paddle.Tensor = None
    decode_block_tables: paddle.Tensor = None


class IluvatarAttnBackend(AttentionBackend):
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
    ):
        super().__init__()
        self.attention_metadata = IluvatarAttentionMetadata()
        self.block_size = fd_config.cache_config.block_size
        assert self.block_size == 16, "Iluvatar paged attn requires block_size must be 16."
        self.max_context_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)
        self.num_kv_heads = kv_num_heads
        self.num_heads = num_heads
        self.total_num_heads = num_heads + 2 * kv_num_heads
        self.head_dim = head_dim
        self.hidden_dim = fd_config.model_config.hidden_size
        # note: scale need to change if using MLA
        self.scale = 1.0 / sqrt(head_dim)
        self.dtype = paddle.get_default_dtype()
        self.enable_mm = fd_config.model_config.enable_mm
        self.rope_batch_stride = self.max_context_len * self.head_dim if self.enable_mm else 0
        if "paddleocr" in fd_config.model_config.model_type:
            self.is_interleaved_rope_mode = False
        else:
            self.is_interleaved_rope_mode = True

        # enable cuda_graph if qkv is only pure decode stage
        # pre-alloc is for enabling cuda graph
        self.attention_metadata.prefill_rope_cos = paddle.empty(shape=(), dtype="float32")
        self.attention_metadata.prefill_rope_sin = paddle.empty(shape=(), dtype="float32")
        self.attention_metadata.prefill_cu_seqlens_q = paddle.empty(shape=(), dtype="int32")
        self.attention_metadata.prefill_block_tables = paddle.empty(shape=(), dtype="int32")
        self.attention_metadata.decode_rope_cos = paddle.empty(shape=(), dtype="float32")
        self.attention_metadata.decode_rope_sin = paddle.empty(shape=(), dtype="float32")
        self.attention_metadata.decode_seq_lens = paddle.empty(shape=(), dtype="int32")
        self.attention_metadata.decode_block_tables = paddle.empty(shape=(), dtype="int32")

    def split_and_copy_rope(self, batch_ids, forward_meta, stage):
        if self.enable_mm:
            # the num_seqs dim of rotary_embs > 1 (e.g. ernie-vl and paddleocr-vl)
            _1d_batch_ids = batch_ids.unsqueeze(0) if batch_ids.dim() == 0 else batch_ids
            cos = forward_meta.rotary_embs[_1d_batch_ids, 0, 0, :, :, :]
            sin = forward_meta.rotary_embs[_1d_batch_ids, 1, 0, :, :, :]
        else:
            #  the num_seqs dim of rotary_embs = 1 (e.g. ernie-text)
            cos = forward_meta.rotary_embs[0, 0, :, :, :]
            sin = forward_meta.rotary_embs[1, 0, :, :, :]

        if stage == "prefill":
            self.attention_metadata.prefill_rope_cos.copy_(cos)
            self.attention_metadata.prefill_rope_sin.copy_(sin)
        elif stage == "decode":
            self.attention_metadata.decode_rope_cos.copy_(cos)
            self.attention_metadata.decode_rope_sin.copy_(sin)
        else:
            raise ValueError("Only support stage is prefill or decode")

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        self.prefill_info_dict = {}
        self.decode_info_dict = {}
        prefill_batch_ids = paddle.where(forward_meta.seq_lens_encoder)[0]
        decode_batch_ids = paddle.where(forward_meta.seq_lens_decoder)[0]
        self.prefill_len = len(prefill_batch_ids)
        self.decode_len = len(decode_batch_ids)
        # only prefill
        if self.decode_len == 0:
            self.mixed = False
            cu_seq_ids = prefill_batch_ids + 1
            cu_seqlens_q = paddle.concat([forward_meta.cu_seqlens_q[:1], forward_meta.cu_seqlens_q[cu_seq_ids]])
            self.split_and_copy_rope(prefill_batch_ids, forward_meta, "prefill")

            self.attention_metadata.prefill_cu_seqlens_q.copy_(cu_seqlens_q)
            self.attention_metadata.prefill_block_tables.copy_(forward_meta.block_tables[prefill_batch_ids, :])
        # only decode
        elif self.prefill_len == 0:
            self.mixed = False
            self.split_and_copy_rope(decode_batch_ids, forward_meta, "decode")
            self.attention_metadata.decode_seq_lens.copy_(forward_meta.seq_lens_decoder[decode_batch_ids] + 1)
            self.attention_metadata.decode_block_tables.copy_(forward_meta.block_tables[decode_batch_ids, :])

        # both prefill and decode
        else:
            self.mixed = True
            self.split_and_copy_rope(prefill_batch_ids, forward_meta, "prefill")
            self.split_and_copy_rope(decode_batch_ids, forward_meta, "decode")
            self.prefill_num_tokens = paddle.sum(forward_meta.seq_lens_encoder).item()
            cu_seqlens_q = paddle.zeros([self.prefill_len + 1], dtype=forward_meta.cu_seqlens_q.dtype)
            cu_seqlens_q[1:] = forward_meta.seq_lens_encoder[prefill_batch_ids]
            # NOTE: The explicit dtype='int32' is required for Iluvatar hardware compatibility.
            cu_seqlens_q = paddle.cumsum(cu_seqlens_q, dtype="int32")

            self.attention_metadata.prefill_cu_seqlens_q.copy_(cu_seqlens_q)
            self.attention_metadata.prefill_block_tables.copy_(forward_meta.block_tables[prefill_batch_ids, :])
            self.attention_metadata.decode_seq_lens.copy_(forward_meta.seq_lens_decoder[decode_batch_ids] + 1)
            self.attention_metadata.decode_block_tables.copy_(forward_meta.block_tables[decode_batch_ids, :])

            self.tmp_buffer = paddle.zeros(
                [self.prefill_num_tokens + self.decode_len, self.hidden_dim], dtype=self.dtype
            )
            prefill_start, decode_start, start = 0, self.prefill_num_tokens, 0
            non_zeros_ids = paddle.where(forward_meta.seq_lens_this_time)[0]
            non_zeros_seq_lens = forward_meta.seq_lens_this_time[non_zeros_ids]
            end = non_zeros_seq_lens[0]
            if end > 1:
                last_stage = "prefill"
                prefill_end = end
                decode_end = decode_start
            else:
                last_stage = "decode"
                prefill_end = 0
                decode_end = decode_start + end

            self.id_group = []
            self.reverse_id_group = []
            for seq_len in non_zeros_seq_lens[1:]:
                if seq_len > 1:
                    if last_stage == "decode":
                        self.id_group.append((decode_start, decode_end))
                        self.reverse_id_group.append((start, end))
                        decode_start = decode_end
                        start = end
                        last_stage = "prefill"
                    prefill_end += seq_len
                    end += seq_len
                else:
                    if last_stage == "prefill":
                        self.id_group.append((prefill_start, prefill_end))
                        self.reverse_id_group.append((start, end))
                        prefill_start = prefill_end
                        start = end
                        last_stage = "decode"
                    decode_end += seq_len
                    end += seq_len

            if prefill_start < prefill_end:
                self.id_group.append((prefill_start, prefill_end))
                self.reverse_id_group.append((start, end))
            if decode_start < decode_end:
                self.id_group.append((decode_start, decode_end))
                self.reverse_id_group.append((start, end))

    def get_attention_meta(self):
        """get_attention_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim]
        value_cache_shape = [max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim]
        return key_cache_shape, value_cache_shape

    def transpose(self, hidden_states):
        for ids, reverse_ids in zip(self.id_group, self.reverse_id_group):
            self.tmp_buffer[ids[0] : ids[1], :] = hidden_states[reverse_ids[0] : reverse_ids[1], :]
        return self.tmp_buffer

    def reverse_transpose(self, hidden_states):
        for ids, reverse_ids in zip(self.id_group, self.reverse_id_group):
            self.tmp_buffer[reverse_ids[0] : reverse_ids[1], :] = hidden_states[ids[0] : ids[1], :]
        return self.tmp_buffer

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
        """
        forward_mixed
        """
        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1
        k_cache = forward_meta.caches[k_cache_id]
        v_cache = forward_meta.caches[v_cache_id]
        if self.decode_len == 0:
            output = prefill_fused_paged_attention(
                qkv,
                k_cache,
                v_cache,
                block_tables=self.attention_metadata.prefill_block_tables,
                cu_seqlens_qkv=self.attention_metadata.prefill_cu_seqlens_q,
                rope_sin=self.attention_metadata.prefill_rope_sin,
                rope_cos=self.attention_metadata.prefill_rope_cos,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                block_size=self.block_size,
                max_seq_len=self.max_context_len,
                scale=self.scale,
                causal=self.causal,
                q_rope=True,
                k_rope=True,
                v_rope=False,
                is_interleaved_rope_mode=self.is_interleaved_rope_mode,
            )
        elif self.prefill_len == 0:
            output = paged_attention(
                qkv,
                k_cache,
                v_cache,
                block_tables=self.attention_metadata.decode_block_tables,
                seq_lens=self.attention_metadata.decode_seq_lens,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                block_size=self.block_size,
                max_context_len=self.max_context_len,
                alibi_slopes=self.attention_metadata.alibi_slopes,
                causal=self.causal,
                window_left=self.attention_metadata.window_left,
                window_right=self.attention_metadata.window_right,
                softcap=self.attention_metadata.softcap,
                use_cuda_graph=self.attention_metadata.use_cuda_graph,
                use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                merged_qkv=True,
                k=qkv,
                v=qkv,
                rope_sin=self.attention_metadata.decode_rope_sin,
                rope_cos=self.attention_metadata.decode_rope_cos,
                rope_batch_stride=self.rope_batch_stride,
                is_interleaved_rope_mode=self.is_interleaved_rope_mode,
            )
        else:
            output = mixed_fused_paged_attention(
                qkv,
                k_cache,
                v_cache,
                prefill_block_tables=self.attention_metadata.prefill_block_tables,
                decode_block_tables=self.attention_metadata.decode_block_tables,
                cu_seqlens_qkv=self.attention_metadata.prefill_cu_seqlens_q,
                seq_lens=self.attention_metadata.decode_seq_lens,
                prefill_rope_sin=self.attention_metadata.prefill_rope_sin,
                prefill_rope_cos=self.attention_metadata.prefill_rope_cos,
                prefill_num_tokens=self.prefill_num_tokens,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                block_size=self.block_size,
                max_seq_len=self.max_context_len,
                scale=self.scale,
                causal=self.causal,
                q_rope=True,
                k_rope=True,
                v_rope=False,
                window_left=self.attention_metadata.window_left,
                window_right=self.attention_metadata.window_right,
                softcap=self.attention_metadata.softcap,
                use_cuda_graph=self.attention_metadata.use_cuda_graph,
                use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                decode_rope_sin=self.attention_metadata.decode_rope_sin,
                decode_rope_cos=self.attention_metadata.decode_rope_cos,
                rope_batch_stride=self.rope_batch_stride,
                is_interleaved_rope_mode=self.is_interleaved_rope_mode,
            )

        return output
