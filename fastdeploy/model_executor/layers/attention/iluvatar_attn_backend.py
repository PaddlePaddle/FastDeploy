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

    def __init__(
        self,
        llm_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.attention_metadata = IluvatarAttentionMetadata()
        self.attention_metadata.block_size = llm_config.parallel_config.block_size
        assert llm_config.parallel_config.enc_dec_block_num == 0, "Iluvatar does not support yet"

        self.attention_metadata.max_context_len = llm_config.parallel_config.max_model_len
        self.attention_metadata.causal = getattr(llm_config.model_config, "causal", True)
        self.speculate_method = getattr(llm_config.parallel_config, "speculate_method", None)
        self.use_speculate = self.speculate_method is not None
        self.attention_metadata.num_kv_heads = kv_num_heads
        self.attention_metadata.dropout = llm_config.model_config.hidden_dropout_prob
        self.num_heads = num_heads
        self.head_dim = head_dim
        # note: scale need to change if using MLA
        self.attention_metadata.scale = 1.0 / sqrt(head_dim)
        self.num_layers = llm_config.model_config.num_hidden_layers
        self.record_block_table_metadata = {}
        self.only_use_flash_attn = int(os.getenv("FD_ILUVATAR_ONLY_USE_FLASH_ATTN", 0)) == 1
        self.do_check_kv_cache = int(os.getenv("FD_ILUVATAR_CHECK_KV_CACHE_CORRECTNESS", 0)) == 1
        if not self.only_use_flash_attn:
            assert self.attention_metadata.block_size == 16, "Iluvatar paged attn requires block_size must be 16."
        if self.do_check_kv_cache:
            self.record_batched_k = [{} for _ in range(self.num_layers)]
            self.record_batched_v = [{} for _ in range(self.num_layers)]

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        self.attention_metadata.block_tables = forward_meta.block_tables
        self.attention_metadata.attn_mask = forward_meta.attn_mask
        self.attention_metadata.seq_lens = forward_meta.seq_lens_decoder
        self.attention_metadata.cu_seqlens_q = forward_meta.cu_seqlens_q
        self.attention_metadata.cu_seqlens_k = forward_meta.cu_seqlens_k

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
            self.attention_metadata.num_kv_heads,
            self.attention_metadata.block_size,
            self.head_dim,
        )

    def get_new_kv(
        self,
        k,
        v,
        k_cache_id: int,
        v_cache_id: int,
        forward_meta: ForwardMeta,
        debug_paged_attn=False,
    ):
        new_k = []
        new_v = []
        tensor_start = 0
        for batch_idx in range(forward_meta.block_tables.shape[0]):
            seq_len = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len == 0:
                continue

            tensor_end = tensor_start + seq_len
            slice_k = k[tensor_start:tensor_end, :, :]
            slice_v = v[tensor_start:tensor_end, :, :]

            if seq_len > 1:
                # prefill
                new_k.append(slice_k)
                new_v.append(slice_v)
            else:
                # decode
                assert seq_len == 1
                cur_block_tables = forward_meta.block_tables[batch_idx]
                cur_used_block_tables = cur_block_tables[cur_block_tables != -1]
                assert (
                    batch_idx in self.record_block_table_metadata
                ), f"Key error: {batch_idx} vs {self.record_block_table_metadata}."
                cur_block_table_metadata = self.record_block_table_metadata[batch_idx]
                record_last_block_id = cur_block_table_metadata["block_id"]
                assert record_last_block_id != -1
                for block_id in cur_used_block_tables:
                    if block_id == record_last_block_id:
                        cache_end = cur_block_table_metadata["cache_end"]
                        block_k_cache = forward_meta.caches[k_cache_id][block_id, :, 0:cache_end, :]
                        block_v_cache = forward_meta.caches[v_cache_id][block_id, :, 0:cache_end, :]
                    else:
                        block_k_cache = forward_meta.caches[k_cache_id][block_id]
                        block_v_cache = forward_meta.caches[v_cache_id][block_id]

                    # [num_kv_heads, block_size, head_dim] -> [block_size, num_kv_heads, head_dim]
                    new_k.append(block_k_cache.transpose([1, 0, 2]).contiguous())
                    new_v.append(block_v_cache.transpose([1, 0, 2]).contiguous())
                    if block_id == record_last_block_id:
                        break

                # as line 301 show, record_block_table_metadata updates when executing the last layer,
                # so slice_k and slice_v has been updated in block_k_cache and block_v_cache
                if not (debug_paged_attn and (k_cache_id / 2 == self.num_layers - 1)):
                    new_k.append(slice_k)
                    new_v.append(slice_v)

            tensor_start = tensor_end

        if len(new_k) == 1:
            return new_k[0], new_v[0]
        else:
            new_k = paddle.concat(new_k, axis=0)
            new_v = paddle.concat(new_v, axis=0)
            return new_k, new_v

    def update_kv_cache(
        self,
        k,
        v,
        k_cache_id: int,
        v_cache_id: int,
        layer_id: int,
        forward_meta: ForwardMeta,
        specific_batch_ids=None,
        debug_paged_attn=False,
    ):
        # [num_tokens, num_kv_heads, head_dim] -> [num_kv_heads, num_tokens, head_dim]
        trans_k = k.transpose([1, 0, 2]).contiguous()
        trans_v = v.transpose([1, 0, 2]).contiguous()
        tensor_start = 0
        for batch_idx in range(forward_meta.block_tables.shape[0]):
            if specific_batch_ids is not None and batch_idx not in specific_batch_ids:
                continue
            seq_len = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len == 0:
                continue

            tensor_end = tensor_start + seq_len
            slice_trans_k = trans_k[:, tensor_start:tensor_end, :]
            slice_trans_v = trans_v[:, tensor_start:tensor_end, :]

            cur_block_tables = forward_meta.block_tables[batch_idx]
            cur_used_block_tables = cur_block_tables[cur_block_tables != -1]

            # prefill
            if seq_len > 1:
                cache_start = 0
                cur_used_num_blocks = cur_used_block_tables.shape[0]
                for i, block_id in enumerate(cur_used_block_tables):
                    # last block: seq_len - cache_start <= block_size
                    if i == cur_used_num_blocks - 1:
                        cache_end = seq_len - cache_start
                        assert cache_end <= self.attention_metadata.block_size
                        forward_meta.caches[k_cache_id][block_id, :, 0:cache_end, :] = slice_trans_k[
                            :, cache_start:seq_len, :
                        ]
                        forward_meta.caches[v_cache_id][block_id, :, 0:cache_end, :] = slice_trans_v[
                            :, cache_start:seq_len, :
                        ]
                        if layer_id == self.num_layers - 1:
                            self.record_block_table_metadata[batch_idx] = {
                                "block_id": block_id.item(),
                                "cache_end": cache_end,
                            }
                    # non last block: seq_lens_this_time > block_size
                    else:
                        assert seq_len > self.attention_metadata.block_size
                        cache_end = cache_start + self.attention_metadata.block_size
                        forward_meta.caches[k_cache_id][block_id] = slice_trans_k[:, cache_start:cache_end, :]
                        forward_meta.caches[v_cache_id][block_id] = slice_trans_v[:, cache_start:cache_end, :]
                        cache_start += self.attention_metadata.block_size
            else:
                # decode
                assert seq_len == 1
                cur_last_block_id = cur_used_block_tables[-1].item()
                assert cur_last_block_id != -1
                assert (
                    batch_idx in self.record_block_table_metadata
                ), f"Key error: {batch_idx} vs {self.record_block_table_metadata}."
                cur_block_table_metadata = self.record_block_table_metadata[batch_idx]
                record_last_block_id = cur_block_table_metadata["block_id"]

                if cur_last_block_id == record_last_block_id:
                    # not alloc new block in decode stage
                    cache_start = cur_block_table_metadata["cache_end"]
                else:
                    # alloc new block in decode stage
                    cache_start = 0

                cache_end = cache_start + 1
                assert cache_end <= self.attention_metadata.block_size

                # paged attn API will update kv cache with inplace mode
                if not debug_paged_attn:
                    forward_meta.caches[k_cache_id][cur_last_block_id, :, cache_start:cache_end, :] = slice_trans_k
                    forward_meta.caches[v_cache_id][cur_last_block_id, :, cache_start:cache_end, :] = slice_trans_v

                # update record_block_table_metadata
                if layer_id == self.num_layers - 1:
                    self.record_block_table_metadata[batch_idx]["block_id"] = cur_last_block_id
                    self.record_block_table_metadata[batch_idx]["cache_end"] = cache_end

            tensor_start = tensor_end

    def _check_new_kv_correctness(self, k, v, new_k, new_v, layer_id: int, forward_meta: ForwardMeta):
        tensor_start = 0
        for batch_idx, seq_lens_this_time in enumerate(forward_meta.seq_lens_this_time):
            if seq_lens_this_time == 0:
                continue
            # note: the second request will also use the batch_idx 0 instead of 1 in
            # the streaming inference mode, so use seq_lens_this_time > 1 with the same
            # batch_idx represents the second request comes.
            if seq_lens_this_time > 1 and batch_idx in self.record_batched_k[layer_id]:
                print(
                    f"clear self.record_batched_batched_k: "
                    f"layer_id={layer_id}, batch_id={batch_idx}, "
                    f"record_lens={len(self.record_batched_k[layer_id][batch_idx])}"
                )
                self.record_batched_k[layer_id][batch_idx].clear()
                self.record_batched_v[layer_id][batch_idx].clear()
            tensor_end = tensor_start + seq_lens_this_time
            slice_k = k[tensor_start:tensor_end, :, :]
            slice_v = v[tensor_start:tensor_end, :, :]
            if batch_idx not in self.record_batched_k[layer_id]:
                self.record_batched_k[layer_id][batch_idx] = []
                self.record_batched_v[layer_id][batch_idx] = []
            self.record_batched_k[layer_id][batch_idx].append(slice_k)
            self.record_batched_v[layer_id][batch_idx].append(slice_v)
            tensor_start = tensor_end

        ref_k, ref_v = [], []
        for batch_idx, seq_lens_this_time in enumerate(forward_meta.seq_lens_this_time):
            if seq_lens_this_time == 0:
                continue
            bached_k_list = self.record_batched_k[layer_id][batch_idx]
            bached_v_list = self.record_batched_v[layer_id][batch_idx]
            ref_k.extend(bached_k_list)
            ref_v.extend(bached_v_list)

        ref_k = paddle.concat(ref_k, axis=0)
        ref_v = paddle.concat(ref_v, axis=0)
        print(
            f"_check_new_kv_correctness: layer_id={layer_id}, "
            f"k.shape={k.shape}, v.shape={v.shape}, "
            f"ref_k.shape={ref_k.shape}, ref_v.shape={ref_v.shape}, "
            f"new_k.shape={new_k.shape}, new_v.shape={new_v.shape}, "
            f"len(self.record_batched_k[layer_id])={len(self.record_batched_k[layer_id])}, "
            f"len(self.record_batched_k[layer_id][0])={len(self.record_batched_k[layer_id][0])}, "
            f"forward_meta.seq_lens_this_time={forward_meta.seq_lens_this_time}"
            f"ref_k[-2:, 0:2, 0:2]={ref_k[-2:, 0:2, 0:2]}, "
            f"ref_v[-2:, 0:2, 0:2]={ref_v[-2:, 0:2, 0:2]}, "
            f"new_k[-2:, 0:2, 0:2]={new_k[-2:, 0:2, 0:2]}, "
            f"new_v[-2:, 0:2, 0:2]={new_v[-2:, 0:2, 0:2]}"
        )
        assert paddle.allclose(
            ref_k.to("cpu").to(paddle.float32),
            new_k.to("cpu").to(paddle.float32),
        )
        assert paddle.allclose(
            ref_v.to("cpu").to(paddle.float32),
            new_v.to("cpu").to(paddle.float32),
        )

    def get_splited_qkv(self, qkv: paddle.Tensor, forward_meta: ForwardMeta):
        q_end = self.num_heads * self.head_dim
        k_end = q_end + self.attention_metadata.num_kv_heads * self.head_dim
        v_end = k_end + self.attention_metadata.num_kv_heads * self.head_dim
        assert v_end == qkv.shape[-1], f"Shape mistach: {v_end} vs {qkv.shape[-1]}"
        assert qkv.shape[0] == forward_meta.cu_seqlens_q[-1]

        q = qkv[..., 0:q_end]
        k = qkv[..., q_end:k_end]
        v = qkv[..., k_end:v_end]
        q = q.view([-1, self.num_heads, self.head_dim]).contiguous()
        k = k.view([-1, self.attention_metadata.num_kv_heads, self.head_dim]).contiguous()
        v = v.view([-1, self.attention_metadata.num_kv_heads, self.head_dim]).contiguous()
        # forward_meta.seq_lens_this_time [max_batch,]
        for batch_idx in range(forward_meta.seq_lens_this_time.shape[0]):
            seq_len_i = forward_meta.seq_lens_this_time[batch_idx]
            if seq_len_i == 0:
                continue
            cached_kv_len = forward_meta.seq_lens_decoder[batch_idx][0]
            cu_seq_start_q = forward_meta.cu_seqlens_q[batch_idx]
            cu_seq_end_q = forward_meta.cu_seqlens_q[batch_idx + 1]
            # forward_meta.rotary_embs is [2, 1, S, 1, D]
            if forward_meta.rotary_embs is not None:
                cos = forward_meta.rotary_embs[0, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                sin = forward_meta.rotary_embs[1, 0, cached_kv_len : cached_kv_len + seq_len_i, :, :]
                q[cu_seq_start_q:cu_seq_end_q] = apply_rope(q[cu_seq_start_q:cu_seq_end_q], cos, sin)
                k[cu_seq_start_q:cu_seq_end_q] = apply_rope(k[cu_seq_start_q:cu_seq_end_q], cos, sin)

        return q, k, v

    def get_splited_info_by_stage(self, q, k, v, forward_meta: ForwardMeta):
        prefill_info_dict = {"q": [], "k": [], "v": [], "batch_ids": []}
        decode_info_dict = {"q": [], "k": [], "v": [], "batch_ids": []}
        tensor_start = 0
        for batch_idx, seq_lens_this_time in enumerate(forward_meta.seq_lens_this_time):
            if seq_lens_this_time == 0:
                continue
            tensor_end = tensor_start + seq_lens_this_time
            slice_q = q[tensor_start:tensor_end, :, :]
            slice_k = k[tensor_start:tensor_end, :, :]
            slice_v = v[tensor_start:tensor_end, :, :]
            if seq_lens_this_time > 1:
                prefill_info_dict["q"].append(slice_q)
                prefill_info_dict["k"].append(slice_k)
                prefill_info_dict["v"].append(slice_v)
                prefill_info_dict["batch_ids"].append(batch_idx)
            else:
                assert seq_lens_this_time == 1
                decode_info_dict["q"].append(slice_q)
                decode_info_dict["k"].append(slice_k)
                decode_info_dict["v"].append(slice_v)
                decode_info_dict["batch_ids"].append(batch_idx)
            tensor_start = tensor_end

        if len(prefill_info_dict["batch_ids"]) > 0:
            prefill_info_dict["q"] = paddle.concat(prefill_info_dict["q"], axis=0)
            prefill_info_dict["k"] = paddle.concat(prefill_info_dict["k"], axis=0)
            prefill_info_dict["v"] = paddle.concat(prefill_info_dict["v"], axis=0)
            cu_seq_ids = list(map(lambda x: x + 1, prefill_info_dict["batch_ids"]))
            prefill_info_dict["cu_seq_ids"] = [0, *cu_seq_ids]

        if len(decode_info_dict["batch_ids"]) > 0:
            decode_info_dict["q"] = paddle.concat(decode_info_dict["q"], axis=0)
            decode_info_dict["k"] = paddle.concat(decode_info_dict["k"], axis=0)
            decode_info_dict["v"] = paddle.concat(decode_info_dict["v"], axis=0)

        return prefill_info_dict, decode_info_dict

    def merge_output(self, prefill_out, decode_out, forward_meta: ForwardMeta):
        assert not (prefill_out is None and decode_out is None), "prefill and decode output cannot both be None"
        if prefill_out is None:
            return decode_out
        elif decode_out is None:
            return prefill_out
        else:
            merged_output = []
            prefill_tensor_start = 0
            decode_tensor_start = 0
            for seq_lens_this_time in forward_meta.seq_lens_this_time:
                if seq_lens_this_time == 0:
                    continue
                if seq_lens_this_time > 1:
                    tensor_end = prefill_tensor_start + seq_lens_this_time
                    merged_output.append(prefill_out[prefill_tensor_start:tensor_end, :, :])
                    prefill_tensor_start = tensor_end
                else:
                    assert seq_lens_this_time == 1
                    tensor_end = decode_tensor_start + seq_lens_this_time
                    merged_output.append(decode_out[decode_tensor_start:tensor_end, :, :])
                    decode_tensor_start = tensor_end

            assert (
                prefill_tensor_start == prefill_out.shape[0]
            ), f"prefill merged unfinished: {prefill_tensor_start} vs {prefill_out.shape[0]}"
            assert (
                decode_tensor_start == decode_out.shape[0]
            ), f"decode merged unfinished: {decode_tensor_start} vs {decode_out.shape[0]}"
            merged_output = paddle.concat(merged_output, axis=0)
            return merged_output

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

        assert qkv is not None
        q_dim = qkv.dim()
        q, k, v = self.get_splited_qkv(qkv, forward_meta)

        if self.only_use_flash_attn:
            new_k, new_v = self.get_new_kv(k, v, k_cache_id, v_cache_id, forward_meta)
            if self.do_check_kv_cache:
                self._check_new_kv_correctness(k, v, new_k, new_v, layer_id, forward_meta)

            out = flash_attn_unpadded(
                q,
                new_k,
                new_v,
                cu_seqlens_q=self.attention_metadata.cu_seqlens_q,
                cu_seqlens_k=self.attention_metadata.cu_seqlens_k,
                max_seqlen_q=self.attention_metadata.max_context_len,
                max_seqlen_k=self.attention_metadata.max_context_len,
                scale=self.attention_metadata.scale,
                dropout=self.attention_metadata.dropout,
                causal=self.attention_metadata.causal,
                return_softmax=self.attention_metadata.return_softmax,
            )[0]

            self.update_kv_cache(k, v, k_cache_id, v_cache_id, layer_id, forward_meta)
        else:
            prefill_info_dict, decode_info_dict = self.get_splited_info_by_stage(q, k, v, forward_meta)
            prefill_out, decode_out = None, None

            if len(prefill_info_dict["batch_ids"]) > 0:
                prefill_out = flash_attn_unpadded(
                    prefill_info_dict["q"],
                    prefill_info_dict["k"],
                    prefill_info_dict["v"],
                    cu_seqlens_q=forward_meta.cu_seqlens_q[prefill_info_dict["cu_seq_ids"]],
                    cu_seqlens_k=forward_meta.cu_seqlens_k[prefill_info_dict["cu_seq_ids"]],
                    max_seqlen_q=self.attention_metadata.max_context_len,
                    max_seqlen_k=self.attention_metadata.max_context_len,
                    scale=self.attention_metadata.scale,
                    dropout=self.attention_metadata.dropout,
                    causal=self.attention_metadata.causal,
                    return_softmax=self.attention_metadata.return_softmax,
                )[0]
                self.update_kv_cache(
                    prefill_info_dict["k"],
                    prefill_info_dict["v"],
                    k_cache_id,
                    v_cache_id,
                    layer_id,
                    forward_meta,
                    specific_batch_ids=prefill_info_dict["batch_ids"],
                )

            if len(decode_info_dict["batch_ids"]) > 0:
                k_cache = forward_meta.caches[k_cache_id]
                v_cache = forward_meta.caches[v_cache_id]

                decode_out = paged_attention(
                    decode_info_dict["q"],
                    k_cache,
                    v_cache,
                    block_tables=forward_meta.block_tables[decode_info_dict["batch_ids"], :],
                    seq_lens=forward_meta.seq_lens_decoder[decode_info_dict["batch_ids"], 0] + 1,
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
                    k=decode_info_dict["k"],
                    v=decode_info_dict["v"],
                )

                if self.do_check_kv_cache:
                    self.update_kv_cache(
                        decode_info_dict["k"],
                        decode_info_dict["v"],
                        k_cache_id,
                        v_cache_id,
                        layer_id,
                        forward_meta,
                        specific_batch_ids=decode_info_dict["batch_ids"],
                        debug_paged_attn=True,
                    )

            if self.do_check_kv_cache:
                new_k, new_v = self.get_new_kv(
                    k,
                    v,
                    k_cache_id,
                    v_cache_id,
                    forward_meta,
                    debug_paged_attn=True,
                )
                self._check_new_kv_correctness(k, v, new_k, new_v, layer_id, forward_meta)

            out = self.merge_output(prefill_out, decode_out, forward_meta)

        if q_dim == 2:
            out = out.view([-1, self.num_heads * self.head_dim])

        return out
