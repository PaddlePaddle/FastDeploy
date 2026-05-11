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
"""Precision parity tests for MLA fused read+interleave.

Verifies that the Triton single-kernel implementation
``fused_read_cache_and_interleave_triton`` produces bit-level identical
results to the Python/Paddle reference ``fused_read_cache_and_interleave_naive``
across a variety of batch/sequence configurations.
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.mla_attention_backend import (
    fused_read_cache_and_interleave_naive,
    fused_read_cache_and_interleave_triton,
)

# Typical DeepSeek V3 MLA geometry.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
BLOCK_SIZE = 64

pytestmark = pytest.mark.skipif(
    not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0,
    reason="fused_read_cache_and_interleave_triton requires CUDA",
)


def _build_inputs(cached_lens, new_lens, dtype="float16", seed=0):
    """Construct all tensors needed by the fused read+interleave entry."""
    assert len(cached_lens) == len(new_lens)
    bsz = len(cached_lens)
    total_new = int(sum(new_lens))

    # Paged latent cache with enough blocks for all batches.
    max_blocks_per_seq = max(1, max((c + BLOCK_SIZE - 1) // BLOCK_SIZE for c in cached_lens) if bsz else 1)
    # Give extra blocks so block ids are not all sequential.
    num_blocks = max(bsz * max_blocks_per_seq + 7, 8)

    rng = np.random.default_rng(seed)
    latent_np = rng.standard_normal((num_blocks, 1, BLOCK_SIZE, LATENT_DIM)).astype(
        np.float16 if dtype == "float16" else np.float32
    )
    latent_cache = paddle.to_tensor(latent_np).cast(dtype)

    # block_tables: assign a distinct physical block id per (batch, block_idx).
    bt_np = np.zeros((bsz, max_blocks_per_seq), dtype=np.int32)
    free_ids = list(range(num_blocks))
    rng.shuffle(free_ids)
    cursor = 0
    for b in range(bsz):
        nb = (cached_lens[b] + BLOCK_SIZE - 1) // BLOCK_SIZE
        for i in range(nb):
            bt_np[b, i] = free_ids[cursor]
            cursor += 1
    block_tables = paddle.to_tensor(bt_np)

    new_kv_c_np = rng.standard_normal((max(total_new, 1), KV_LORA_RANK)).astype(np.float32)
    new_k_pe_np = rng.standard_normal((max(total_new, 1), QK_ROPE_HEAD_DIM)).astype(np.float32)
    new_compressed_kv = paddle.to_tensor(new_kv_c_np[:total_new] if total_new > 0 else new_kv_c_np[:0]).cast(dtype)
    new_k_pe = paddle.to_tensor(new_k_pe_np[:total_new] if total_new > 0 else new_k_pe_np[:0]).cast(dtype)

    cu_total = np.zeros(bsz + 1, dtype=np.int32)
    cu_new = np.zeros(bsz + 1, dtype=np.int32)
    for i in range(bsz):
        cu_total[i + 1] = cu_total[i] + cached_lens[i] + new_lens[i]
        cu_new[i + 1] = cu_new[i] + new_lens[i]
    cu_seqlens_k = paddle.to_tensor(cu_total)
    cu_seqlens_q = paddle.to_tensor(cu_new)

    return {
        "latent_cache": latent_cache,
        "block_tables": block_tables,
        "new_compressed_kv": new_compressed_kv,
        "new_k_pe": new_k_pe,
        "cu_seqlens_k": cu_seqlens_k,
        "cu_seqlens_q": cu_seqlens_q,
    }


def _run_both(inputs):
    common = dict(
        latent_cache=inputs["latent_cache"],
        block_tables=inputs["block_tables"],
        new_compressed_kv=inputs["new_compressed_kv"],
        new_k_pe=inputs["new_k_pe"],
        cu_seqlens_k=inputs["cu_seqlens_k"],
        cu_seqlens_q=inputs["cu_seqlens_q"],
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_size=BLOCK_SIZE,
    )

    ref_kv, ref_pe = fused_read_cache_and_interleave_naive(**common)
    out_kv, out_pe = fused_read_cache_and_interleave_triton(**common)
    return ref_kv, ref_pe, out_kv, out_pe


def _assert_bitwise_equal(ref_kv, ref_pe, out_kv, out_pe):
    assert tuple(out_kv.shape) == tuple(ref_kv.shape)
    assert tuple(out_pe.shape) == tuple(ref_pe.shape)
    # Both paths do pure copies (no arithmetic), so they should agree bit-for-bit.
    np.testing.assert_array_equal(out_kv.cast("float32").numpy(), ref_kv.cast("float32").numpy())
    np.testing.assert_array_equal(out_pe.cast("float32").numpy(), ref_pe.cast("float32").numpy())


# -----------------------------------------------------------------------------
# Test cases: (name, cached_lens, new_lens)
# -----------------------------------------------------------------------------
_CASES = [
    # Single batch, purely new tokens (no prefix cache).
    ("single_no_cache", [0], [17]),
    # Single batch, purely cached (edge case; triton path should still be correct).
    ("single_all_cache_short", [13], [1]),
    # Single batch spanning multiple cache blocks.
    ("single_multi_block", [BLOCK_SIZE * 3 + 5], [9]),
    # Single batch, cached aligned to exact block boundary.
    ("single_cache_aligned", [BLOCK_SIZE * 2], [7]),
    # Multi batch, uniform small lengths.
    ("multi_uniform_small", [8, 8, 8, 8], [4, 4, 4, 4]),
    # Multi batch, mixed: some with cache, some without.
    ("multi_mixed_cache", [0, 33, 0, BLOCK_SIZE + 1], [5, 7, 11, 3]),
    # Multi batch, long + short mixed.
    ("multi_long_short", [BLOCK_SIZE * 5 + 3, 2, BLOCK_SIZE * 2, 0], [16, 1, 8, 12]),
    # Larger batch with varied lengths.
    (
        "multi_large",
        [0, 17, BLOCK_SIZE, BLOCK_SIZE * 2 + 1, 5, 0, 64, BLOCK_SIZE * 4 + 30],
        [8, 3, 16, 1, 12, 4, 9, 7],
    ),
    # Every cached_len > 0 (no batch without prefix cache).
    ("multi_all_have_cache", [BLOCK_SIZE + 2, 7, BLOCK_SIZE * 2 + 17, 40], [6, 6, 6, 6]),
    # bsz=1, minimal.
    ("minimal", [1], [1]),
]


@pytest.mark.parametrize("name,cached_lens,new_lens", _CASES, ids=[c[0] for c in _CASES])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_triton_matches_naive(name, cached_lens, new_lens, dtype):
    inputs = _build_inputs(cached_lens, new_lens, dtype=dtype, seed=hash(name) & 0xFFFF)
    ref_kv, ref_pe, out_kv, out_pe = _run_both(inputs)
    _assert_bitwise_equal(ref_kv, ref_pe, out_kv, out_pe)


def test_total_tokens_zero():
    """All batches empty -> early return with empty tensors from both paths."""
    inputs = _build_inputs([0, 0], [0, 0], dtype="float16", seed=1)
    ref_kv, ref_pe, out_kv, out_pe = _run_both(inputs)
    assert ref_kv.shape[0] == 0 and out_kv.shape[0] == 0
    assert ref_pe.shape[0] == 0 and out_pe.shape[0] == 0


# ---------------------------------------------------------------------------
# Hand-crafted baseline test: small tensors with deterministic integer values so
# every expected output entry can be written out and compared directly. This
# gives a ground-truth anchor independent of the naive reference, covering the
# real chunk-prefill / prefix-cache layout end-to-end.
#
# Scenario:
#   bsz = 2
#   batch 0: cached=3, new=2  (spans 2 paged blocks: pbid 5 then pbid 2)
#   batch 1: cached=0, new=4
#   block_size=2, kv_lora_rank=4, qk_rope_head_dim=2
#
# Interleaved output layout (per-batch [cached, new]):
#   t0..t2 : batch0 cached 0,1,2  (block_tables[0]=[5,2])
#   t3..t4 : batch0 new    0,1
#   t5..t8 : batch1 new    2,3,4,5
# ---------------------------------------------------------------------------


def _fill_latent_cache(num_blocks, block_size, lora, rope):
    """latent_cache[pbid, 0, off, d] = pbid * 1000 + off * 100 + d."""
    latent_dim = lora + rope
    arr = np.zeros((num_blocks, 1, block_size, latent_dim), dtype=np.float32)
    for p in range(num_blocks):
        for o in range(block_size):
            for d in range(latent_dim):
                arr[p, 0, o, d] = p * 1000 + o * 100 + d
    return arr


def test_manual_baseline():
    lora, rope = 4, 2
    block_size = 2
    num_blocks = 8

    latent_np = _fill_latent_cache(num_blocks, block_size, lora, rope)
    latent_cache = paddle.to_tensor(latent_np).cast("float32")

    # batch 0 cached tokens live in physical blocks (pbid=5 then pbid=2),
    # batch 1 has no cached tokens so its row is unused.
    block_tables = paddle.to_tensor(np.array([[5, 2], [0, 0]], dtype=np.int32))

    # New tokens: 2 (batch 0) + 4 (batch 1) = 6 rows, deterministic values.
    total_new = 6
    new_kv = np.zeros((total_new, lora), dtype=np.float32)
    new_pe = np.zeros((total_new, rope), dtype=np.float32)
    for i in range(total_new):
        for d in range(lora):
            new_kv[i, d] = -(i * 10 + d + 1)  # negatives so they never clash with cache pattern
        for d in range(rope):
            new_pe[i, d] = -(i * 10 + d + 101)
    new_compressed_kv = paddle.to_tensor(new_kv)
    new_k_pe = paddle.to_tensor(new_pe)

    # cu_seqlens_k = cumsum(cached + new) per batch = [0, 5, 9]
    # cu_seqlens_q             = cumsum(new)          per batch = [0, 2, 6]
    cu_k = paddle.to_tensor(np.array([0, 5, 9], dtype=np.int32))
    cu_q = paddle.to_tensor(np.array([0, 2, 6], dtype=np.int32))

    out_kv, out_pe = fused_read_cache_and_interleave_triton(
        latent_cache,
        block_tables,
        new_compressed_kv,
        new_k_pe,
        cu_k,
        cu_q,
        lora,
        rope,
        block_size,
    )

    # Hand-built ground truth: total_tokens=9.
    expected_kv = np.zeros((9, lora), dtype=np.float32)
    expected_pe = np.zeros((9, rope), dtype=np.float32)

    # t0: batch0 cached 0 -> pbid=5, off=0 -> 5000 + 0 + d
    # t1: batch0 cached 1 -> pbid=5, off=1 -> 5000 + 100 + d
    # t2: batch0 cached 2 -> pbid=2, off=0 -> 2000 + 0   + d
    cached_refs = [(5, 0), (5, 1), (2, 0)]
    for t, (pbid, off) in enumerate(cached_refs):
        for d in range(lora):
            expected_kv[t, d] = pbid * 1000 + off * 100 + d
        for d in range(rope):
            expected_pe[t, d] = pbid * 1000 + off * 100 + lora + d

    # t3,t4: batch0 new tokens (src=0,1)
    # t5..t8: batch1 new tokens (src=2,3,4,5)
    new_sources = [0, 1, 2, 3, 4, 5]
    for slot, src in enumerate(new_sources):
        expected_kv[3 + slot] = new_kv[src]
        expected_pe[3 + slot] = new_pe[src]

    np.testing.assert_array_equal(out_kv.numpy(), expected_kv)
    np.testing.assert_array_equal(out_pe.numpy(), expected_pe)


def test_manual_baseline_no_cache():
    """All batches pure prefill (no prefix cache) -> kernel must still copy
    new tokens in order with no off-by-one."""
    lora, rope = 4, 2
    block_size = 2
    # Any latent_cache / block_tables; kernel should never read from them.
    latent_cache = paddle.zeros([4, 1, block_size, lora + rope], dtype="float32")
    block_tables = paddle.zeros([3, 1], dtype="int32")

    new_kv = np.arange(5 * lora, dtype=np.float32).reshape(5, lora)
    new_pe = np.arange(5 * rope, dtype=np.float32).reshape(5, rope) + 1000
    new_compressed_kv = paddle.to_tensor(new_kv)
    new_k_pe = paddle.to_tensor(new_pe)

    # 3 batches, new=[2,1,2], no cache
    cu_k = paddle.to_tensor(np.array([0, 2, 3, 5], dtype=np.int32))
    cu_q = paddle.to_tensor(np.array([0, 2, 3, 5], dtype=np.int32))

    out_kv, out_pe = fused_read_cache_and_interleave_triton(
        latent_cache,
        block_tables,
        new_compressed_kv,
        new_k_pe,
        cu_k,
        cu_q,
        lora,
        rope,
        block_size,
    )
    np.testing.assert_array_equal(out_kv.numpy(), new_kv)
    np.testing.assert_array_equal(out_pe.numpy(), new_pe)


def test_manual_baseline_all_cached():
    """Single batch, all cached (decode-like K build with zero new tokens)."""
    lora, rope = 4, 2
    block_size = 2
    num_blocks = 4
    latent_np = _fill_latent_cache(num_blocks, block_size, lora, rope)
    latent_cache = paddle.to_tensor(latent_np)

    # batch 0 cached=4 spans two blocks; pick pbid=3 then pbid=1.
    block_tables = paddle.to_tensor(np.array([[3, 1]], dtype=np.int32))

    # No new tokens; still need a valid (0-row) tensor.
    new_compressed_kv = paddle.zeros([0, lora], dtype="float32")
    new_k_pe = paddle.zeros([0, rope], dtype="float32")

    cu_k = paddle.to_tensor(np.array([0, 4], dtype=np.int32))
    cu_q = paddle.to_tensor(np.array([0, 0], dtype=np.int32))

    out_kv, out_pe = fused_read_cache_and_interleave_triton(
        latent_cache,
        block_tables,
        new_compressed_kv,
        new_k_pe,
        cu_k,
        cu_q,
        lora,
        rope,
        block_size,
    )

    # expected: pbid=3 off=0,1 then pbid=1 off=0,1
    expected_kv = np.zeros((4, lora), dtype=np.float32)
    expected_pe = np.zeros((4, rope), dtype=np.float32)
    for t, (pbid, off) in enumerate([(3, 0), (3, 1), (1, 0), (1, 1)]):
        for d in range(lora):
            expected_kv[t, d] = pbid * 1000 + off * 100 + d
        for d in range(rope):
            expected_pe[t, d] = pbid * 1000 + off * 100 + lora + d
    np.testing.assert_array_equal(out_kv.numpy(), expected_kv)
    np.testing.assert_array_equal(out_pe.numpy(), expected_pe)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
