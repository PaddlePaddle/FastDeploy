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


# qk[seq, h, d], cos/sin [seq, 1, d]
def apply_rope(qk, cos, sin):
    rotate_half = paddle.reshape(
        paddle.stack([-qk[..., 1::2], qk[..., 0::2]], axis=-1),
        paddle.shape(qk),
    )
    out = paddle.add(paddle.multiply(qk, cos), paddle.multiply(rotate_half, sin))
    return paddle.cast(out, qk.dtype)


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
        self.speculate_method = getattr(fd_config.parallel_config, "speculate_method", None)
        self.use_speculate = self.speculate_method is not None
        self.num_kv_heads = kv_num_heads
        self.num_heads = num_heads
        self.total_num_heads = num_heads + 2 * kv_num_heads
        self.head_dim = head_dim
        self.hidden_dim = fd_config.model_config.hidden_size
        # note: scale need to change if using MLA
        self.scale = 1.0 / sqrt(head_dim)
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.dtype = paddle.get_default_dtype()
        self.enable_mm = fd_config.model_config.enable_mm

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        if self.enable_mm:
            # VL: TODO: The first 0 may need to be replaced with batch_id
            # of max_num_seqs when running multiple batch case later
            self.rope_cos = forward_meta.rotary_embs[0, 0, 0, :, :, :]
            self.rope_sin = forward_meta.rotary_embs[0, 1, 0, :, :, :]
        else:
            # text
            self.rope_cos = forward_meta.rotary_embs[0, 0, :, :, :]
            self.rope_sin = forward_meta.rotary_embs[1, 0, :, :, :]
        self.prefill_info_dict = {}
        self.decode_info_dict = {}
        self.prefill_info_dict["batch_ids"] = paddle.where(forward_meta.seq_lens_encoder)[0]
        self.decode_info_dict["batch_ids"] = paddle.where(forward_meta.seq_lens_decoder)[0]
        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])
        # only prefill
        if self.decode_len == 0:
            cu_seq_ids = list(range(self.prefill_len + 1))
            self.prefill_info_dict["cu_seqlens_q"] = forward_meta.cu_seqlens_q[cu_seq_ids]
            self.mixed = False
        # only decode
        elif self.prefill_len == 0:
            self.mixed = False
        # both prefill and decode
        else:
            self.mixed = True
            self.prefill_num_tokens = paddle.sum(forward_meta.seq_lens_encoder).item()
            self.prefill_info_dict["cu_seqlens_q"] = paddle.zeros(
                [self.prefill_len + 1], dtype=forward_meta.cu_seqlens_q.dtype
            )
            self.prefill_info_dict["cu_seqlens_q"][1:] = forward_meta.seq_lens_encoder[
                self.prefill_info_dict["batch_ids"], 0
            ]
            # NOTE: The explicit dtype='int32' is required for Iluvatar hardware compatibility.
            self.prefill_info_dict["cu_seqlens_q"] = paddle.cumsum(
                self.prefill_info_dict["cu_seqlens_q"], dtype="int32"
            )

            self.tmp_buffer = paddle.zeros(
                [self.prefill_num_tokens + self.decode_len, self.hidden_dim], dtype=self.dtype
            )

            prefill_start, decode_start, start = 0, self.prefill_num_tokens, 0
            non_zeros_ids = forward_meta.seq_lens_this_time != 0
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
                block_tables=forward_meta.block_tables[self.prefill_info_dict["batch_ids"], :],
                cu_seqlens_qkv=self.prefill_info_dict["cu_seqlens_q"],
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
                rope_sin=self.rope_sin,
                rope_cos=self.rope_cos,
            )
        elif self.prefill_len == 0:
            output = paged_attention(
                qkv,
                k_cache,
                v_cache,
                block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
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
                rope_sin=self.rope_sin,
                rope_cos=self.rope_cos,
            )
        else:
            output = mixed_fused_paged_attention(
                qkv,
                k_cache,
                v_cache,
                prefill_block_tables=forward_meta.block_tables[self.prefill_info_dict["batch_ids"], :],
                decode_block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                cu_seqlens_qkv=self.prefill_info_dict["cu_seqlens_q"],
                seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
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
                rope_sin=self.rope_sin,
                rope_cos=self.rope_cos,
            )

        return output
