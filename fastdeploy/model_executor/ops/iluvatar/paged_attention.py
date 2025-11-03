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

import paddle

try:
    from fastdeploy.model_executor.ops.iluvatar import (
        mixed_fused_paged_attn,
        paged_attn,
        prefill_fused_paged_attn,
    )
except ImportError:
    paged_attn = None
    prefill_fused_paged_attn = None
    mixed_fused_paged_attn = None


def paged_attention(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    seq_lens: paddle.Tensor,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    scale: float,
    block_size: int,
    max_context_len: int,
    alibi_slopes: paddle.Tensor = None,
    causal: bool = True,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    use_cuda_graph: bool = False,
    use_sqrt_alibi: bool = False,
    merged_qkv: bool = False,
    k: paddle.Tensor = None,
    v: paddle.Tensor = None,
    rope_sin: paddle.Tensor = None,
    rope_cos: paddle.Tensor = None,
):
    return paged_attn(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        alibi_slopes,
        k,
        v,
        rope_sin,
        rope_cos,
        num_heads,
        head_dim,
        num_kv_heads,
        scale,
        block_size,
        max_context_len,
        causal,
        window_left,
        window_right,
        softcap,
        use_cuda_graph,
        use_sqrt_alibi,
        merged_qkv,
    )


def prefill_fused_paged_attention(
    qkv: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    cu_seqlens_qkv: paddle.Tensor,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    scale: float,
    causal: bool = True,
    q_rope: bool = True,
    k_rope: bool = True,
    v_rope: bool = False,
    rope_sin: paddle.Tensor = None,
    rope_cos: paddle.Tensor = None,
):
    return prefill_fused_paged_attn(
        qkv,
        k_cache,
        v_cache,
        block_tables,
        cu_seqlens_qkv,
        rope_sin,
        rope_cos,
        num_heads,
        head_dim,
        num_kv_heads,
        block_size,
        max_seq_len,
        scale,
        causal,
        q_rope,
        k_rope,
        v_rope,
    )


def mixed_fused_paged_attention(
    qkv: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    prefill_block_tables: paddle.Tensor,
    decode_block_tables: paddle.Tensor,
    cu_seqlens_qkv: paddle.Tensor,
    seq_lens: paddle.Tensor,
    prefill_num_tokens: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    block_size: int,
    max_seq_len: int,
    scale: float,
    causal: bool = True,
    q_rope: bool = True,
    k_rope: bool = True,
    v_rope: bool = False,
    window_left: int = -1,
    window_right: int = -1,
    softcap: float = 0.0,
    use_cuda_graph: bool = False,
    use_sqrt_alibi: bool = False,
    rope_sin: paddle.Tensor = None,
    rope_cos: paddle.Tensor = None,
):
    return mixed_fused_paged_attn(
        qkv,
        k_cache,
        v_cache,
        prefill_block_tables,
        decode_block_tables,
        cu_seqlens_qkv,
        seq_lens,
        rope_sin,
        rope_cos,
        prefill_num_tokens,
        num_heads,
        head_dim,
        num_kv_heads,
        block_size,
        max_seq_len,
        scale,
        causal,
        q_rope,
        k_rope,
        v_rope,
        window_left,
        window_right,
        softcap,
        use_cuda_graph,
        use_sqrt_alibi,
    )
