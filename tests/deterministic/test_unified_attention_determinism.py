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
Unit tests for unified extend attention kernel determinism.

Tests that the Triton kernel produces bit-identical output for the same
KV cache content regardless of the prefix/extend split. This isolates
whether non-determinism comes from the attention kernel itself or from
upstream KV cache writing (gqa_rope_write_cache).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_unified_attention_determinism.py -v -s
"""

import os
import sys

import numpy as np
import paddle
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    build_kv_indices_from_block_tables,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
)

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Reference: pure-paddle attention (float32, no approximation)
# ---------------------------------------------------------------------------
def reference_attention(
    q, k_cache, v_cache, block_tables, prefix_len, extend_len, block_size, num_kv_heads, is_causal=True
):
    """
    Pure paddle reference attention for a single sequence.
    q: [extend_len, num_heads, head_dim]
    k_cache/v_cache: [num_blocks, kv_heads, block_size, head_dim]
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    group_size = num_heads // num_kv_heads
    total_len = prefix_len + extend_len
    sm_scale = 1.0 / (head_dim**0.5)

    # Gather all K/V from paged cache
    k_list, v_list = [], []
    for t in range(total_len):
        blk_idx = t // block_size
        blk_off = t % block_size
        phys_blk = int(block_tables[0, blk_idx].item())
        k_list.append(k_cache[phys_blk, :, blk_off, :])  # [kv_heads, head_dim]
        v_list.append(v_cache[phys_blk, :, blk_off, :])

    k_all = paddle.stack(k_list, axis=0).astype("float32")  # [total_len, kv_heads, head_dim]
    v_all = paddle.stack(v_list, axis=0).astype("float32")

    # GQA: repeat KV heads
    if group_size > 1:
        k_all = k_all.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([total_len, num_heads, head_dim])
        v_all = v_all.unsqueeze(2).expand([-1, -1, group_size, -1]).reshape([total_len, num_heads, head_dim])

    q_f32 = q.astype("float32")  # [extend_len, num_heads, head_dim]
    # [extend_len, num_heads, total_len]
    scores = paddle.einsum("qhd,khd->qhk", q_f32, k_all) * sm_scale

    if is_causal:
        for qi in range(extend_len):
            for ki in range(total_len):
                if ki < prefix_len:
                    pass  # prefix always visible
                else:
                    ki_in_extend = ki - prefix_len
                    if ki_in_extend > qi:
                        scores[qi, :, ki] = float("-inf")

    weights = paddle.nn.functional.softmax(scores, axis=-1)
    out = paddle.einsum("qhk,khd->qhd", weights, v_all)
    return out  # [extend_len, num_heads, head_dim], float32


def _build_block_tables_and_cache(total_len, block_size, kv_heads, head_dim, dtype):
    """Create a simple paged KV cache with sequential block mapping."""
    num_blocks = (total_len + block_size - 1) // block_size
    # Sequential block mapping: block i in logical = block i in physical
    block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)  # [1, num_blocks]

    paddle.seed(42)
    k_cache = paddle.randn([num_blocks, kv_heads, block_size, head_dim]).astype(dtype)
    v_cache = paddle.randn([num_blocks, kv_heads, block_size, head_dim]).astype(dtype)
    return block_tables, k_cache, v_cache


def _run_unified_kernel(
    q,
    k_cache,
    v_cache,
    block_tables,
    prefix_len,
    extend_len,
    block_size,
    num_heads,
    kv_heads,
    head_dim,
    is_causal=True,
):
    """Run the unified Triton kernel with given prefix/extend split."""
    bs = 1
    total_len = prefix_len + extend_len

    prefix_lens_t = paddle.to_tensor([prefix_len], dtype="int32")
    extend_lens_t = paddle.to_tensor([extend_len], dtype="int32")
    total_lens_t = paddle.to_tensor([total_len], dtype="int32")

    # Build prefix KV indices
    prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
        block_tables, prefix_lens_t, block_size, bs
    )

    # Build all KV indices
    all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)

    # Build extend KV indices
    extend_start_loc = paddle.zeros([1], dtype="int32")
    extend_kv_indices = paddle.empty([max(extend_len, 1)], dtype="int32")
    if extend_len > 0:
        plen = prefix_len
        src_start = int(all_kv_indptr[0].item()) + plen
        extend_kv_indices[:extend_len] = all_kv_indices[src_start : src_start + extend_len]

    # Build unified indices
    unified_kv_indptr, unified_kv_indices, _ = build_unified_kv_indices(
        prefix_kv_indptr, prefix_kv_indices, extend_start_loc, extend_lens_t, extend_kv_indices, bs
    )

    # QO indptr
    qo_indptr = paddle.to_tensor([0, extend_len], dtype="int32")

    # Run kernel
    o = paddle.zeros([extend_len, num_heads, head_dim], dtype=q.dtype)
    extend_attention_fwd_unified(
        q,
        o,
        k_cache,
        v_cache,
        qo_indptr,
        unified_kv_indptr,
        unified_kv_indices,
        prefix_lens_t,
        num_heads,
        kv_heads,
        head_dim,
        extend_len,
        is_causal,
    )
    return o


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnifiedAttentionDeterminism:
    """Test that unified attention kernel is split-invariant."""

    @pytest.mark.parametrize(
        "total_len,prefix_len",
        [
            (128, 0),  # all extend, no prefix
            (128, 64),  # half prefix, half extend
            (128, 120),  # mostly prefix, small extend
            (825, 0),  # real-world: cache miss
            (825, 768),  # real-world: cache hit (matches the failing test)
        ],
    )
    def test_split_invariance_last_token(self, total_len, prefix_len):
        """
        Core test: for identical KV cache, the output of the LAST query token
        must be bit-identical regardless of prefix/extend split.

        This simulates:
        - Run 0 (cache miss): prefix_len=0, extend_len=total_len
        - Run N (cache hit):  prefix_len=P, extend_len=total_len-P
        """
        block_size = 64
        num_heads = 28
        kv_heads = 4
        head_dim = 128
        dtype = "bfloat16"

        extend_len_miss = total_len  # cache miss: all tokens are extend
        extend_len_hit = total_len - prefix_len  # cache hit

        # Create shared KV cache
        block_tables, k_cache, v_cache = _build_block_tables_and_cache(
            total_len, block_size, kv_heads, head_dim, dtype
        )

        # Q for cache miss: all total_len queries
        paddle.seed(123)
        q_miss = paddle.randn([extend_len_miss, num_heads, head_dim]).astype(dtype)

        # Q for cache hit: only the last extend_len_hit queries
        # These must be the SAME as the last extend_len_hit queries in q_miss
        q_hit = q_miss[-extend_len_hit:].clone()

        # Run kernel: cache miss (prefix=0, extend=total_len)
        o_miss = _run_unified_kernel(
            q_miss,
            k_cache,
            v_cache,
            block_tables,
            prefix_len=0,
            extend_len=extend_len_miss,
            block_size=block_size,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

        # Run kernel: cache hit (prefix=prefix_len, extend=extend_len_hit)
        o_hit = _run_unified_kernel(
            q_hit,
            k_cache,
            v_cache,
            block_tables,
            prefix_len=prefix_len,
            extend_len=extend_len_hit,
            block_size=block_size,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

        # Compare the LAST token's output (most critical for generation)
        o_miss_last = o_miss[-1:].astype("float32").numpy()
        o_hit_last = o_hit[-1:].astype("float32").numpy()

        max_diff = np.abs(o_miss_last - o_hit_last).max()
        print(
            f"\n[TEST] total={total_len} prefix={prefix_len} "
            f"extend_miss={extend_len_miss} extend_hit={extend_len_hit}"
        )
        print(f"[TEST] Last token max_diff = {max_diff}")
        print(f"[TEST] o_miss[-1] md5 = {_md5(o_miss_last)}")
        print(f"[TEST] o_hit[-1]  md5 = {_md5(o_hit_last)}")

        if max_diff == 0.0:
            print("[TEST] PASS: bit-identical")
        else:
            print(f"[TEST] FAIL: max_diff = {max_diff}")

        assert np.array_equal(o_miss_last, o_hit_last), (
            f"Attention output differs for last token: max_diff={max_diff}. "
            f"This means the Triton kernel is NOT split-invariant."
        )

    def test_correctness_vs_reference(self):
        """Verify the Triton kernel matches a pure-paddle reference implementation."""
        block_size = 64
        num_heads = 8
        kv_heads = 2
        head_dim = 128
        total_len = 256
        prefix_len = 192
        extend_len = total_len - prefix_len
        dtype = "bfloat16"

        block_tables, k_cache, v_cache = _build_block_tables_and_cache(
            total_len, block_size, kv_heads, head_dim, dtype
        )

        paddle.seed(456)
        q = paddle.randn([extend_len, num_heads, head_dim]).astype(dtype)

        # Triton kernel
        o_triton = _run_unified_kernel(
            q,
            k_cache,
            v_cache,
            block_tables,
            prefix_len=prefix_len,
            extend_len=extend_len,
            block_size=block_size,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )

        # Reference
        o_ref = reference_attention(q, k_cache, v_cache, block_tables, prefix_len, extend_len, block_size, kv_heads)

        o_triton_f32 = o_triton.astype("float32").numpy()
        o_ref_f32 = o_ref.numpy()

        max_diff = np.abs(o_triton_f32 - o_ref_f32).max()
        cos_sim = np.sum(o_triton_f32 * o_ref_f32) / (np.linalg.norm(o_triton_f32) * np.linalg.norm(o_ref_f32) + 1e-12)

        print(f"\n[TEST] correctness: max_diff={max_diff}, cos_sim={cos_sim}")
        assert max_diff < 0.02, f"Triton vs reference max_diff={max_diff} too large"
        assert cos_sim > 0.999, f"Triton vs reference cos_sim={cos_sim} too low"


def _md5(arr):
    import hashlib

    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
