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

from typing import TYPE_CHECKING

import paddle
from paddle.nn.functional import scaled_dot_product_attention

from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


class PaddleNativeAttnBackend(AttentionBackend):
    """
    The backend class that uses paddle native attention implementation.
    Which is used only for testing purpose.
    """

    def __init__(self) -> None:
        super().__init__()

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: paddle.Tensor,
        output: paddle.Tensor,
        k_cache: paddle.Tensor,
        v_cache: paddle.Tensor,
        req_to_token: paddle.Tensor,
        req_pool_indices: paddle.Tensor,
        seq_lens: paddle.Tensor,
        extend_prefix_lens: paddle.Tensor,
        extend_seq_lens: paddle.Tensor,
        causal: bool = False,
    ) -> paddle.Tensor:
        """Run the extend forward by using paddle native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        # query = query.movedim(0, query.dim() - 2) =>
        query = paddle.transpose(query, perm=[1, 0, 2])

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = paddle.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            # per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            # per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_key = k_cache[per_req_tokens].transpose([query.dim() - 2, 0])
            per_req_value = v_cache[per_req_tokens].transpose([query.dim() - 2, 0])

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    is_causal=causal,
                )
                .squeeze(0)
                .transpose([query.dim() - 2, 0])
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _scaled_dot_product_attention(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        is_causal: bool = False,
    ) -> paddle.Tensor:
        """Paddle implementation of scaled dot-product attention."""
        # query, key, value shape: [batch_size, num_heads, seq_len, head_size]
        d_k = query.shape[-1]
        scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))  # QK^T

        scores = scores / paddle.sqrt(paddle.to_tensor(d_k, dtype=scores.dtype))
        if is_causal:
            # Apply causal mask
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            mask = paddle.triu(paddle.ones((q_len, k_len)) * -1e4, diagonal=1)
            scores += mask.unsqueeze(0).unsqueeze(0)

        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        output = paddle.matmul(attn_weights, value)
        return output

    def _run_sdpa_forward_decode(
        self,
        query: paddle.Tensor,
        output: paddle.Tensor,
        k_cache: paddle.Tensor,
        v_cache: paddle.Tensor,
        req_to_token: paddle.Tensor,
        req_pool_indices: paddle.Tensor,
        seq_lens: paddle.Tensor,
        causal: bool = False,
    ) -> paddle.Tensor:
        """Run the decode forward by using paddle native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.transpose([1, 0, 2])

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]

            # [seq_len_kv, num_heads, head_size] -> [num_heads, seq_len_kv, head_size]
            per_req_key = k_cache[per_req_tokens].transpose([query.dim() - 2, 0])
            per_req_value = v_cache[per_req_tokens].transpose([query.dim() - 2, 0])

            per_req_out = (
                self._scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    is_causal=causal,
                )
                .squeeze(0)
                .transpose([query.dim() - 2, 0])
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
        save_kv_cache: bool = True,
    ) -> paddle.Tensor:
        """
        Run the prefill and extend(prompt cache) attention forward by using paddle native sdpa op.
        """
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.self.num_heads * layer.v_head_dim))
        else:
            o = paddle.empty_like(q)

        if save_kv_cache:
            forward_meta.token_to_kv_pool.set_kv_buffer(layer, forward_meta.out_cache_loc, k, v)

        q_ = q.view([-1, layer.self.num_heads, layer.qk_head_dim])
        o_ = o.view([-1, layer.self.num_heads, layer.v_head_dim])

        causal = True

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_meta.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_meta.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_meta.req_to_token_pool.req_to_token,
            forward_meta.req_pool_indices,
            forward_meta.seq_lens,
            forward_meta.extend_prefix_lens,
            forward_meta.extend_seq_lens,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Run the decoding attention forward by using paddle native sdpa op.
        """
        q = q.reshape([-1, layer.self.num_heads * layer.qk_head_dim])

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.self.num_heads * layer.v_head_dim))
        else:
            o = paddle.empty_like(q)

        forward_meta.token_to_kv_pool.set_kv_buffer(layer, forward_meta.out_cache_loc, k, v)

        q_ = q.view([-1, layer.self.num_heads, layer.qk_head_dim])
        o_ = o.view([-1, layer.self.num_heads, layer.v_head_dim])

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_meta.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_meta.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_meta.req_to_token_pool.req_to_token,
            forward_meta.req_pool_indices,
            forward_meta.seq_lens,
            causal=False,
        )

        return o
