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
from math import sqrt
from typing import TYPE_CHECKING, Optional

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.ops.iluvatar import paged_attention

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class IluvatarAttentionMetadata(AttentionMetadata):
    """
    IluvatarAttentionMetadata
    """

    # flash_attn metadata
    cu_seqlens_q: Optional[paddle.Tensor] = None
    cu_seqlens_k: Optional[paddle.Tensor] = None
    fixed_seed_offset: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    attn_mask_start_row_indices: Optional[paddle.Tensor] = None
    dropout: float = 0.0
    causal: bool = True
    return_softmax: bool = False
    rng_name: str = ""

    # paged_attn metadata
    block_tables: Optional[paddle.Tensor] = None
    seq_lens: Optional[paddle.Tensor] = None
    num_kv_heads: int = 1
    scale: float = 1.0
    block_size: int = 1
    max_context_len: int = 1
    alibi_slopes: Optional[paddle.Tensor] = None
    # causal: bool = True
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

    def __init__(self, fd_config: FDConfig, kv_num_heads: int, num_heads: int, head_dim: int):
        super().__init__()
        self.attention_metadata = IluvatarAttentionMetadata()
        self.attention_metadata.block_size = fd_config.parallel_config.block_size
        assert (
            fd_config.parallel_config.enc_dec_block_num == 0
        ), f"Iluvatar does not support yet, {fd_config.parallel_config.enc_dec_block_num}"
        assert self.attention_metadata.block_size == 16, "Iluvatar paged attn requires block_size must be 16."

        self.attention_metadata.max_context_len = fd_config.parallel_config.max_model_len
        self.attention_metadata.causal = getattr(fd_config.model_config, "causal", True)
        self.speculate_method = getattr(fd_config.parallel_config, "speculate_method", None)
        self.use_speculate = self.speculate_method is not None
        self.attention_metadata.num_kv_heads = kv_num_heads
        self.attention_metadata.dropout = fd_config.model_config.hidden_dropout_prob
        self.num_heads = num_heads
        self.total_num_heads = num_heads + 2 * kv_num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.total_hidden_dim = self.total_num_heads * head_dim
        # note: scale need to change if using MLA
        self.attention_metadata.scale = 1.0 / sqrt(head_dim)
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.dtype = paddle.get_default_dtype()

        self.record_block_table_metadata = {}
        self.enable_fused_attention = int(os.getenv("FD_ILUVATAR_ENABLE_FUSED_ATTN", 1))

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        self.prefill_info_dict = {}
        self.decode_info_dict = {}

        prefill_non_zeros_ids = forward_meta.seq_lens_this_time > 1
        decode_non_zeros_ids = forward_meta.seq_lens_this_time == 1
        self.prefill_info_dict["batch_ids"] = paddle.where(prefill_non_zeros_ids)[0]
        self.decode_info_dict["batch_ids"] = paddle.where(decode_non_zeros_ids)[0]

        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])
        # only prefill
        if self.decode_len == 0:
            cu_seq_ids = list(range(self.prefill_len + 1))
            self.prefill_info_dict["cu_seqlens_q"] = forward_meta.cu_seqlens_q[cu_seq_ids]
        # only decode
        elif self.prefill_len == 0:
            pass
        # both prefill and decode
        else:
            prefill_num_tokens = paddle.sum(forward_meta.seq_lens_this_time[prefill_non_zeros_ids])
            decode_num_tokens = paddle.sum(forward_meta.seq_lens_this_time[decode_non_zeros_ids])

            self.prefill_info_dict["cu_seqlens_q"] = paddle.zeros(
                [self.prefill_len + 1], dtype=forward_meta.cu_seqlens_q.dtype
            )
            self.prefill_info_dict["cu_seqlens_q"][1:] = forward_meta.seq_lens_encoder[
                self.prefill_info_dict["batch_ids"], 0
            ]
            self.prefill_info_dict["cu_seqlens_q"] = paddle.cumsum(self.prefill_info_dict["cu_seqlens_q"])

            self.prefill_qkv = paddle.zeros([prefill_num_tokens, self.total_hidden_dim], dtype=self.dtype)
            self.decode_qkv = paddle.zeros([decode_num_tokens, self.total_hidden_dim], dtype=self.dtype)
            self.merged_output = paddle.zeros(
                [prefill_num_tokens + decode_num_tokens, self.num_heads, self.head_dim], dtype=self.dtype
            )

            prefill_start, decode_start, start = 0, 0, 0
            non_zeros_ids = forward_meta.seq_lens_this_time != 0
            non_zeros_seq_lens = forward_meta.seq_lens_this_time[non_zeros_ids]
            end = non_zeros_seq_lens[0]
            if end > 1:
                last_stage = "prefill"
                prefill_end = end
                decode_end = 0
            else:
                last_stage = "decode"
                prefill_end = 0
                decode_end = end

            self.prefill_info_dict["id_group"] = []
            self.prefill_info_dict["reverse_id_group"] = []
            self.decode_info_dict["id_group"] = []
            self.decode_info_dict["reverse_id_group"] = []
            self.record_stages = []
            for seq_len in non_zeros_seq_lens[1:]:
                if seq_len > 1:
                    if last_stage == "decode":
                        self.record_stages.append((last_stage, len(self.decode_info_dict["id_group"])))
                        self.decode_info_dict["id_group"].append((decode_start, decode_end))
                        self.decode_info_dict["reverse_id_group"].append((start, end))
                        decode_start = decode_end
                        start = end
                        last_stage = "prefill"
                    prefill_end += seq_len
                    end += seq_len
                else:
                    if last_stage == "prefill":
                        self.record_stages.append((last_stage, len(self.prefill_info_dict["id_group"])))
                        self.prefill_info_dict["id_group"].append((prefill_start, prefill_end))
                        self.prefill_info_dict["reverse_id_group"].append((start, end))
                        prefill_start = prefill_end
                        start = end
                        last_stage = "decode"
                    decode_end += seq_len
                    end += seq_len

            if prefill_start < prefill_end:
                self.record_stages.append(("prefill", len(self.prefill_info_dict["id_group"])))
                self.prefill_info_dict["id_group"].append((prefill_start, prefill_end))
                self.prefill_info_dict["reverse_id_group"].append((start, end))
            if decode_start < decode_end:
                self.record_stages.append(("decode", len(self.decode_info_dict["id_group"])))
                self.decode_info_dict["id_group"].append((decode_start, decode_end))
                self.decode_info_dict["reverse_id_group"].append((start, end))

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
        return (
            max_num_blocks,
            self.attention_metadata.num_kv_heads,
            self.attention_metadata.block_size,
            self.head_dim,
        )

    def prefill_update_kv_cache(
        self, k, v, k_cache_id: int, v_cache_id: int, layer_id: int, forward_meta: ForwardMeta, prefill_batch_ids: list
    ):
        # [num_tokens, num_kv_heads, head_dim] -> [num_kv_heads, num_tokens, head_dim]
        trans_k = k.transpose([1, 0, 2]).contiguous()
        trans_v = v.transpose([1, 0, 2]).contiguous()
        tensor_start = 0
        for batch_idx in prefill_batch_ids:
            seq_len = forward_meta.seq_lens_this_time[batch_idx]

            tensor_end = tensor_start + seq_len
            slice_trans_k = trans_k[:, tensor_start:tensor_end, :]
            slice_trans_v = trans_v[:, tensor_start:tensor_end, :]

            cur_block_tables = forward_meta.block_tables[batch_idx]
            cur_used_block_tables = cur_block_tables[cur_block_tables != -1]

            cache_start = 0
            cur_used_num_blocks = cur_used_block_tables.shape[0]
            for i, block_id in enumerate(cur_used_block_tables):
                # last block: seq_len - cache_start <= block_size
                if i == cur_used_num_blocks - 1:
                    cache_end = seq_len - cache_start
                    assert cache_end <= self.attention_metadata.block_size
                    paddle.assign(
                        slice_trans_k[:, cache_start:seq_len, :],
                        output=forward_meta.caches[k_cache_id][block_id, :, 0:cache_end, :],
                    )
                    paddle.assign(
                        slice_trans_v[:, cache_start:seq_len, :],
                        output=forward_meta.caches[v_cache_id][block_id, :, 0:cache_end, :],
                    )
                    if layer_id == self.num_layers - 1:
                        self.record_block_table_metadata[batch_idx] = {
                            "block_id": block_id.item(),
                            "cache_end": cache_end.item(),
                        }
                # non last block: seq_lens_this_time > block_size
                else:
                    assert seq_len > self.attention_metadata.block_size
                    cache_end = cache_start + self.attention_metadata.block_size
                    paddle.assign(
                        slice_trans_k[:, cache_start:cache_end, :], output=forward_meta.caches[k_cache_id][block_id]
                    )
                    paddle.assign(
                        slice_trans_v[:, cache_start:cache_end, :], output=forward_meta.caches[v_cache_id][block_id]
                    )
                    cache_start += self.attention_metadata.block_size

            tensor_start = tensor_end

    def get_splited_qkv(
        self, qkv: paddle.Tensor, forward_meta: ForwardMeta, cu_seqlens_q: paddle.Tensor, batch_ids=None
    ):
        q_end = self.hidden_dim
        k_end = q_end + self.attention_metadata.num_kv_heads * self.head_dim
        v_end = k_end + self.attention_metadata.num_kv_heads * self.head_dim
        assert v_end == qkv.shape[-1], f"Shape mismatch: {v_end} vs {qkv.shape[-1]}"
        assert qkv.shape[0] == cu_seqlens_q[-1], f"Shape mismatch: {qkv.shape[0]} vs {cu_seqlens_q[-1]}"

        if batch_ids is None:
            batch_ids = list(range(forward_meta.seq_lens_this_time.shape[0]))

        q = qkv[..., 0:q_end]
        k = qkv[..., q_end:k_end]
        v = qkv[..., k_end:v_end]
        q = q.view([-1, self.num_heads, self.head_dim])
        k = k.view([-1, self.attention_metadata.num_kv_heads, self.head_dim])
        v = v.view([-1, self.attention_metadata.num_kv_heads, self.head_dim])

        for idx in range(len(cu_seqlens_q) - 1):
            batch_idx = batch_ids[idx]
            seq_len_i = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len_i == 0:
                continue
            cached_kv_len = forward_meta.seq_lens_decoder[batch_idx][0]
            cu_seq_start_q = cu_seqlens_q[idx]
            cu_seq_end_q = cu_seqlens_q[idx + 1]
            # forward_meta.rotary_embs is [2, 1, S, 1, D]
            if forward_meta.rotary_embs is not None:
                cos = forward_meta.rotary_embs[0, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                sin = forward_meta.rotary_embs[1, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                q[cu_seq_start_q:cu_seq_end_q] = apply_rope(q[cu_seq_start_q:cu_seq_end_q], cos, sin)
                k[cu_seq_start_q:cu_seq_end_q] = apply_rope(k[cu_seq_start_q:cu_seq_end_q], cos, sin)

        return q, k, v

    def split_pd_qkv(self, qkv):

        for ids, reverse_ids in zip(self.prefill_info_dict["id_group"], self.prefill_info_dict["reverse_id_group"]):
            self.prefill_qkv[ids[0] : ids[1], :] = qkv[reverse_ids[0] : reverse_ids[1], :]

        for ids, reverse_ids in zip(self.decode_info_dict["id_group"], self.decode_info_dict["reverse_id_group"]):
            self.decode_qkv[ids[0] : ids[1], :] = qkv[reverse_ids[0] : reverse_ids[1], :]

        return self.prefill_qkv, self.decode_qkv

    def merge_pd_output(self, prefill_out, decode_out):
        for stage, idx in self.record_stages:
            if stage == "prefill":
                ids = self.prefill_info_dict["id_group"][idx]
                reverse_ids = self.prefill_info_dict["reverse_id_group"][idx]
                self.merged_output[reverse_ids[0] : reverse_ids[1], :, :] = prefill_out[ids[0] : ids[1], :, :]
            else:
                ids = self.decode_info_dict["id_group"][idx]
                reverse_ids = self.decode_info_dict["reverse_id_group"][idx]
                self.merged_output[reverse_ids[0] : reverse_ids[1], :, :] = decode_out[ids[0] : ids[1], :, :]
        return self.merged_output

    def forward_prefill(self, prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        prefill_q, prefill_k, prefill_v = self.get_splited_qkv(
            prefill_qkv,
            forward_meta,
            self.prefill_info_dict["cu_seqlens_q"],
            batch_ids=self.prefill_info_dict["batch_ids"],
        )

        prefill_out = flash_attn_unpadded(
            prefill_q,
            prefill_k,
            prefill_v,
            cu_seqlens_q=self.prefill_info_dict["cu_seqlens_q"],
            cu_seqlens_k=self.prefill_info_dict["cu_seqlens_q"],
            max_seqlen_q=self.attention_metadata.max_context_len,
            max_seqlen_k=self.attention_metadata.max_context_len,
            scale=self.attention_metadata.scale,
            dropout=self.attention_metadata.dropout,
            causal=self.attention_metadata.causal,
            return_softmax=self.attention_metadata.return_softmax,
        )[0]
        self.prefill_update_kv_cache(
            prefill_k, prefill_v, k_cache_id, v_cache_id, layer_id, forward_meta, self.prefill_info_dict["batch_ids"]
        )

        return prefill_out

    def forward_decode(self, decode_qkv, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        k_cache = forward_meta.caches[k_cache_id]
        v_cache = forward_meta.caches[v_cache_id]
        if self.enable_fused_attention:
            rope_cos = forward_meta.rotary_embs[0, 0, :, :, :]
            rope_sin = forward_meta.rotary_embs[1, 0, :, :, :]
            decode_out = paged_attention(
                decode_qkv.view([-1, self.total_num_heads, self.head_dim]),
                k_cache,
                v_cache,
                block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                num_kv_heads=self.attention_metadata.num_kv_heads,
                scale=self.attention_metadata.scale,
                block_size=self.attention_metadata.block_size,
                max_context_len=self.attention_metadata.max_context_len,
                alibi_slopes=self.attention_metadata.alibi_slopes,
                causal=self.attention_metadata.causal,
                window_left=self.attention_metadata.window_left,
                window_right=self.attention_metadata.window_right,
                softcap=self.attention_metadata.softcap,
                use_cuda_graph=self.attention_metadata.use_cuda_graph,
                use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                merged_qkv=True,
                k=decode_qkv,
                v=decode_qkv,
                rope_sin=rope_sin,
                rope_cos=rope_cos,
            )
        else:
            decode_q, decode_k, decode_v = self.get_splited_qkv(
                decode_qkv,
                forward_meta,
                self.decode_info_dict["cu_seqlens_q"],
                batch_ids=self.decode_info_dict["batch_ids"],
            )

            decode_out = paged_attention(
                decode_q,
                k_cache,
                v_cache,
                block_tables=forward_meta.block_tables[self.decode_info_dict["batch_ids"], :],
                seq_lens=forward_meta.seq_lens_decoder[self.decode_info_dict["batch_ids"], 0] + 1,
                num_kv_heads=self.attention_metadata.num_kv_heads,
                scale=self.attention_metadata.scale,
                block_size=self.attention_metadata.block_size,
                max_context_len=self.attention_metadata.max_context_len,
                alibi_slopes=self.attention_metadata.alibi_slopes,
                causal=self.attention_metadata.causal,
                window_left=self.attention_metadata.window_left,
                window_right=self.attention_metadata.window_right,
                softcap=self.attention_metadata.softcap,
                use_cuda_graph=self.attention_metadata.use_cuda_graph,
                use_sqrt_alibi=self.attention_metadata.use_sqrt_alibi,
                k=decode_k,
                v=decode_v,
            )

        return decode_out

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
        assert not self.use_speculate, "IluvatarAttnBackend cannot support speculate now"
        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1
        q_dim = qkv.dim()
        assert q_dim == 2

        if self.decode_len == 0:
            output = self.forward_prefill(qkv, layer_id, k_cache_id, v_cache_id, forward_meta)

        elif self.prefill_len == 0:
            output = self.forward_decode(qkv, k_cache_id, v_cache_id, forward_meta)
        else:
            prefill_qkv, decode_qkv = self.split_pd_qkv(qkv)
            prefill_output = self.forward_prefill(prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta)
            decode_output = self.forward_decode(decode_qkv, k_cache_id, v_cache_id, forward_meta)
            output = self.merge_pd_output(prefill_output, decode_output)

        output = output.view([-1, self.num_heads * self.head_dim])
        return output
