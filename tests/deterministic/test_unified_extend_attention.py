"""
Unified extend attention kernel tests — correctness, determinism, and split-invariance.

Correctness verification strategy:
    Two independent, simple Python reference implementations (naive_attention via einsum
    and sdpa_attention_reference via float32 matmul) are first cross-validated against each
    other, then the Triton kernel output is compared against both using max absolute diff
    and cosine similarity thresholds. Broad parametrized coverage spans head configurations
    (MHA/GQA/MQA), data types (float16/bfloat16), sequence lengths, and edge cases.

Test scenarios:
1. Cumsum utility (triton_cumsum_with_zero_prefix): basic, empty, cross-validate vs paddle
2. Index building (build_kv_indices_from_block_tables / build_unified_kv_indices):
   single/multi/empty sequence, non-contiguous blocks, large batch stress (bs=32),
   ref-vs-triton cross-validation, edge cases (bs=0, all-zero extend)
3. Kernel correctness (extend_attention_fwd_unified vs naive_attention):
   MHA/GQA/MQA, head_dim=13/64/80/96/128/256, float16/bfloat16,
   causal/non-causal, with/without prefix, non-contiguous blocks,
   long sequence (4096), large values, custom sm_scale
4. Split invariance (core feature):
   cache miss vs hit produce identical output, GQA variant,
   non-aligned prefix, multiple splits (6 different prefix lengths),
   bfloat16 dtype, Qwen2.5-7B real-world config (28q/4kv, 825 tokens)
5. Determinism: 5-10 runs bitwise identical, with/without prefix, GQA large batch
6. Production-scale correctness: bs=19 SGLang-scale, seq=4096, mixed lengths, prefix
7. Cross-validation: naive vs sdpa reference, triton vs sdpa (triple validation)

Usage:
    source /root/paddlejob/workspace/env_run/gongweibao/archfd/fdarchenv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_unified_extend_attention.py -v
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    build_kv_indices_from_block_tables,
    build_kv_indices_from_block_tables_ref,
    build_unified_kv_indices,
    extend_attention_fwd_unified,
    triton_cumsum_with_zero_prefix,
)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
FP16_ATOL = 1e-2
BF16_ATOL = 5e-2
MQA_ATOL = 5e-2
LARGE_HEAD_ATOL = 0.1
NONSTANDARD_HEAD_ATOL = 0.05
PRIME_HEAD_ATOL = 0.1
LARGE_SCALE_RTOL = 1e-2
LARGE_SCALE_ATOL = 0.02
COSINE_SIM_THRESHOLD = 1 - 1e-4
DETERMINISM_ATOL = 0.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.astype("float32").reshape([-1])
    b_flat = b.astype("float32").reshape([-1])
    dot = float(paddle.sum(a_flat * b_flat).item())
    norm_a = float(paddle.sqrt(paddle.sum(a_flat * a_flat)).item())
    norm_b = float(paddle.sqrt(paddle.sum(b_flat * b_flat)).item())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _expand_kv_for_gqa(k, v, num_heads):
    """Expand KV heads for GQA: [bs, kv_len, kv_heads, dim] -> [bs, kv_len, num_heads, dim]."""
    bs, kv_len, num_kv_heads, head_dim = k.shape
    group_size = num_heads // num_kv_heads
    k = k.unsqueeze(3).expand([-1, -1, -1, group_size, -1]).reshape([bs, kv_len, num_heads, head_dim])
    v = v.unsqueeze(3).expand([-1, -1, -1, group_size, -1]).reshape([bs, kv_len, num_heads, head_dim])
    return k, v


def _build_causal_mask(bs, q_len, kv_len, prefix_lens, is_causal):
    """Build causal mask: [bs, 1, q_len, kv_len] with -inf for masked positions."""
    mask = paddle.zeros([bs, 1, q_len, kv_len], dtype="float32")
    if is_causal:
        qi_idx = paddle.arange(q_len, dtype="int32").reshape([1, 1, q_len, 1])
        ki_idx = paddle.arange(kv_len, dtype="int32").reshape([1, 1, 1, kv_len])
        plens = prefix_lens.reshape([bs, 1, 1, 1]).astype("int32")
        cond = (ki_idx >= plens) & (qi_idx + plens < ki_idx)
        mask = paddle.where(cond, paddle.full_like(mask, float("-inf")), mask)
    return mask


def sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True):
    """Reference implementation using manual SDPA (float32 matmul)."""
    bs, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    k, v = _expand_kv_for_gqa(k, v, num_heads)
    q_t = q.transpose([0, 2, 1, 3]).astype("float32")
    k_t = k.transpose([0, 2, 1, 3]).astype("float32")
    v_t = v.transpose([0, 2, 1, 3]).astype("float32")
    mask = _build_causal_mask(bs, q_len, kv_len, prefix_lens, is_causal)
    scale = 1.0 / (head_dim**0.5)
    scores = paddle.matmul(q_t, k_t.transpose([0, 1, 3, 2])) * scale + mask
    attn = paddle.nn.functional.softmax(scores, axis=-1)
    out = paddle.matmul(attn, v_t)
    return out.transpose([0, 2, 1, 3])


def naive_attention(q, k, v, prefix_lens, is_causal=True):
    """Naive multi-head attention reference using einsum."""
    bs, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    k, v = _expand_kv_for_gqa(k, v, num_heads)
    scale = 1.0 / (head_dim**0.5)
    scores = paddle.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if is_causal:
        qi_idx = paddle.arange(q_len, dtype="int32").reshape([1, 1, q_len, 1])
        ki_idx = paddle.arange(kv_len, dtype="int32").reshape([1, 1, 1, kv_len])
        plens = prefix_lens.reshape([bs, 1, 1, 1])
        # mask: ki >= plen AND qi < ki - plen  =>  qi + plen < ki
        mask = (ki_idx >= plens) & (qi_idx + plens < ki_idx)
        scores = paddle.where(mask, paddle.full_like(scores, float("-inf")), scores)
    attn = paddle.nn.functional.softmax(scores, axis=-1)
    return paddle.einsum("bhqk,bkhd->bqhd", attn, v)


def _build_paged_kv_cache(k_flat, v_flat, block_size):
    """Pack flat KV tensors into paged cache format [num_blocks, heads, block_size, dim]."""
    total_tokens, num_kv_heads, head_dim = k_flat.shape
    num_blocks = (total_tokens + block_size - 1) // block_size
    cache_k = paddle.zeros([num_blocks, num_kv_heads, block_size, head_dim], dtype=k_flat.dtype)
    cache_v = paddle.zeros([num_blocks, num_kv_heads, block_size, head_dim], dtype=v_flat.dtype)
    for t in range(total_tokens):
        block_id = t // block_size
        offset = t % block_size
        cache_k[block_id, :, offset, :] = k_flat[t]
        cache_v[block_id, :, offset, :] = v_flat[t]
    return cache_k, cache_v


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _run_single_seq_kernel(q_flat, k_flat, v_flat, block_size, num_q, num_kv, dim, is_causal=True, sm_scale=None):
    """Run Triton kernel on a single sequence (bs=1, no prefix). Returns output tensor."""
    seq_len = q_flat.shape[0]
    cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)
    qo_indptr = paddle.to_tensor([0, seq_len], dtype="int32")
    kv_indptr = paddle.to_tensor([0, k_flat.shape[0]], dtype="int32")
    kv_indices = paddle.arange(k_flat.shape[0], dtype="int32")
    prefix_lens = paddle.zeros([1], dtype="int32")
    o = paddle.zeros_like(q_flat)
    kwargs = {"sm_scale": sm_scale} if sm_scale is not None else {}
    return extend_attention_fwd_unified(
        q_flat,
        o,
        cache_k,
        cache_v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        num_q,
        num_kv,
        dim,
        seq_len,
        is_causal,
        **kwargs,
    )


def _run_kernel_vs_ref(
    bs,
    q_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_size,
    prefix_len=0,
    is_causal=True,
    dtype="float16",
    sm_scale=None,
    ref_fn=None,
):
    """Run Triton kernel and compare against reference. Returns (max_diff, cos_sim)."""
    if ref_fn is None:
        ref_fn = naive_attention

    kv_len = prefix_len + q_len
    paddle.seed(42)
    q_batched = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype(dtype)
    k_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype(dtype)
    v_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype(dtype)

    prefix_lens_ref = paddle.full([bs], prefix_len, dtype="int32")
    ref_out = ref_fn(q_batched, k_batched, v_batched, prefix_lens_ref, is_causal)

    k_flat = k_batched.reshape([-1, num_kv_heads, head_dim])
    v_flat = v_batched.reshape([-1, num_kv_heads, head_dim])
    cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

    total_q = bs * q_len
    q_flat = q_batched.reshape([total_q, num_q_heads, head_dim])
    o_flat = paddle.zeros_like(q_flat)

    q_lens = paddle.full([bs], q_len, dtype="int32")
    qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(q_lens).astype("int32")])
    kv_lens = paddle.full([bs], kv_len, dtype="int32")
    kv_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(kv_lens).astype("int32")])
    kv_indices = paddle.arange(bs * kv_len, dtype="int32")
    prefix_lens_t = paddle.full([bs], prefix_len, dtype="int32")

    kwargs = {"sm_scale": sm_scale} if sm_scale is not None else {}

    o_flat = extend_attention_fwd_unified(
        q_flat,
        o_flat,
        cache_k,
        cache_v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens_t,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_len,
        is_causal,
        **kwargs,
    )

    triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
    ref_fp32 = ref_out.astype("float32")
    triton_fp32 = triton_out.astype("float32")
    max_diff = float(paddle.max(paddle.abs(ref_fp32 - triton_fp32)).item())
    cos_sim = cosine_similarity(ref_fp32, triton_fp32)
    return max_diff, cos_sim


def _run_determinism_check(
    bs,
    q_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_size,
    prefix_len=0,
    num_runs=10,
):
    """Run kernel multiple times and verify bitwise identical results."""
    kv_len = prefix_len + q_len
    actual_bs = 1 if prefix_len > 0 else bs
    total_q = actual_bs * q_len if prefix_len == 0 else q_len
    total_kv = actual_bs * kv_len if prefix_len == 0 else kv_len

    paddle.seed(999)
    q_flat = paddle.randn([total_q, num_q_heads, head_dim]).astype("float16")
    k_flat = paddle.randn([total_kv, num_kv_heads, head_dim]).astype("float16")
    v_flat = paddle.randn([total_kv, num_kv_heads, head_dim]).astype("float16")
    cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

    if prefix_len == 0:
        seq_lens = paddle.full([actual_bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(total_kv, dtype="int32")
        prefix_lens_t = paddle.zeros([actual_bs], dtype="int32")
    else:
        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, kv_len], dtype="int32")
        kv_indices = paddle.arange(kv_len, dtype="int32")
        prefix_lens_t = paddle.to_tensor([prefix_len], dtype="int32")

    results = []
    for _ in range(num_runs):
        o = paddle.zeros_like(q_flat)
        o = extend_attention_fwd_unified(
            q_flat,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens_t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )
        results.append(o.astype("float32").numpy())

    for i in range(1, len(results)):
        assert (results[0] == results[i]).all(), f"Run 0 vs run {i} differ"


# ===========================================================================
# 1. Cumsum utility tests
# ===========================================================================


class TestTritonCumsumWithZeroPrefix:

    def test_basic(self):
        x = paddle.to_tensor([3, 1, 4, 1, 5], dtype="int32")
        assert triton_cumsum_with_zero_prefix(x).tolist() == [0, 3, 4, 8, 9, 14]

    def test_single_element(self):
        x = paddle.to_tensor([7], dtype="int32")
        assert triton_cumsum_with_zero_prefix(x).tolist() == [0, 7]

    def test_empty(self):
        x = paddle.to_tensor([], dtype="int32")
        assert triton_cumsum_with_zero_prefix(x, n=0).tolist() == [0]

    def test_all_zeros(self):
        x = paddle.zeros([4], dtype="int32")
        assert triton_cumsum_with_zero_prefix(x).tolist() == [0, 0, 0, 0, 0]

    def test_with_explicit_n(self):
        x = paddle.to_tensor([2, 3, 5, 7, 11], dtype="int32")
        assert triton_cumsum_with_zero_prefix(x, n=3).tolist() == [0, 2, 5, 10]

    def test_matches_paddle_reference(self):
        np.random.seed(42)
        for length in [1, 7, 32, 64, 127, 128, 255]:
            x_np = np.random.randint(0, 100, size=length).astype(np.int32)
            x = paddle.to_tensor(x_np, dtype="int32")
            triton_result = triton_cumsum_with_zero_prefix(x)
            paddle_result = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(x).astype("int32")])
            assert triton_result.tolist() == paddle_result.tolist(), f"Mismatch at length={length}"


# ===========================================================================
# 2. Index building tests (block tables, unified indices, deterministic dispatch)
# ===========================================================================


class TestBuildKvIndices:
    """Tests for build_kv_indices_from_block_tables, build_unified_kv_indices,
    and _deterministic_build_triton_indices."""

    # --- build_kv_indices_from_block_tables ---

    def test_single_sequence(self):
        block_tables = paddle.to_tensor([[2, 5]], dtype="int32")
        seq_lens = paddle.to_tensor([6], dtype="int32")
        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, 4, bs=1)
        assert kv_indptr.tolist() == [0, 6]
        expected = [8, 9, 10, 11, 20, 21]
        assert kv_indices.tolist() == expected

    def test_multiple_sequences(self):
        block_tables = paddle.to_tensor([[1, 3], [0, 0]], dtype="int32")
        seq_lens = paddle.to_tensor([3, 2], dtype="int32")
        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, 2, bs=2)
        assert kv_indptr.tolist() == [0, 3, 5]
        assert kv_indices.tolist() == [2, 3, 6, 0, 1]

    def test_empty_sequence(self):
        block_tables = paddle.to_tensor([[0]], dtype="int32")
        seq_lens = paddle.to_tensor([0], dtype="int32")
        kv_indptr, _ = build_kv_indices_from_block_tables(block_tables, seq_lens, 4, bs=1)
        assert kv_indptr.tolist() == [0, 0]

    def test_large_sequence(self):
        block_size, seq_len = 64, 500
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks_needed, dtype="int32").unsqueeze(0)
        seq_lens = paddle.to_tensor([seq_len], dtype="int32")
        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs=1)
        assert kv_indptr.tolist() == [0, seq_len]
        for t in range(seq_len):
            expected = (t // block_size) * block_size + t % block_size
            assert kv_indices[t].item() == expected, f"Mismatch at t={t}"

    def test_non_contiguous_blocks(self):
        block_size, seq_len = 4, 10
        block_tables = paddle.to_tensor([[5, 2, 8]], dtype="int32")
        seq_lens = paddle.to_tensor([seq_len], dtype="int32")
        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs=1)
        expected = []
        for t in range(seq_len):
            bid = [5, 2, 8][t // block_size]
            expected.append(bid * block_size + t % block_size)
        assert kv_indices.tolist() == expected

    def test_ref_vs_triton_cross_validation(self):
        rng = np.random.RandomState(123)
        for bs in [1, 4, 16]:
            block_size = 16
            seq_lens_np = rng.randint(0, 100, size=bs).astype(np.int32)
            max_seq = int(seq_lens_np.max()) if bs > 0 else 0
            max_blocks = max((max_seq + block_size - 1) // block_size, 1)
            block_tables_np = rng.randint(0, 50, size=(bs, max_blocks)).astype(np.int32)
            block_tables = paddle.to_tensor(block_tables_np)
            seq_lens = paddle.to_tensor(seq_lens_np)
            indptr_triton, indices_triton = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs)
            indptr_ref, indices_ref = build_kv_indices_from_block_tables_ref(block_tables, seq_lens, block_size, bs)
            assert indptr_triton.tolist() == indptr_ref.tolist(), f"indptr mismatch at bs={bs}"
            total_len = int(indptr_ref[-1].item())
            assert (
                indices_triton[:total_len].tolist() == indices_ref[:total_len].tolist()
            ), f"indices mismatch at bs={bs}"

    def test_bs_zero(self):
        block_tables = paddle.zeros([0, 4], dtype="int32")
        seq_lens = paddle.zeros([0], dtype="int32")
        kv_indptr, _ = build_kv_indices_from_block_tables(block_tables, seq_lens, 4, bs=0)
        assert kv_indptr.tolist() == [0]

    # --- build_unified_kv_indices ---

    def test_unified_basic_merge(self):
        prefix_kv_indptr = paddle.to_tensor([0, 2, 3], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 20], dtype="int32")
        extend_seq_lens = paddle.to_tensor([3, 2], dtype="int32")
        extend_start_loc = paddle.to_tensor([0, 3], dtype="int32")
        extend_kv_indices = paddle.to_tensor([100, 101, 102, 200, 201], dtype="int32")
        unified_indptr, unified_indices, prefix_lens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs=2,
        )
        assert prefix_lens.tolist() == [2, 1]
        assert unified_indptr.tolist() == [0, 5, 8]
        assert unified_indices[:8].tolist() == [10, 11, 100, 101, 102, 20, 200, 201]

    def test_unified_large_bs(self):
        bs = 8
        prefix_lens_list = [10, 20, 5, 0, 15, 8, 30, 12]
        extend_lens_list = [5, 10, 3, 7, 2, 6, 4, 8]
        prefix_indptr = [0]
        for p in prefix_lens_list:
            prefix_indptr.append(prefix_indptr[-1] + p)
        prefix_kv_indptr = paddle.to_tensor(prefix_indptr, dtype="int32")
        prefix_kv_indices = paddle.arange(sum(prefix_lens_list), dtype="int32") + 1000
        extend_seq_lens = paddle.to_tensor(extend_lens_list, dtype="int32")
        extend_start_loc_list = [0]
        for e in extend_lens_list[:-1]:
            extend_start_loc_list.append(extend_start_loc_list[-1] + e)
        extend_start_loc = paddle.to_tensor(extend_start_loc_list, dtype="int32")
        extend_kv_indices = paddle.arange(sum(extend_lens_list), dtype="int32") + 2000
        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )
        assert plens.tolist() == prefix_lens_list
        expected_indptr = [0]
        for p, e in zip(prefix_lens_list, extend_lens_list):
            expected_indptr.append(expected_indptr[-1] + p + e)
        assert unified_indptr.tolist() == expected_indptr
        for s in range(bs):
            start = expected_indptr[s]
            end = expected_indptr[s + 1]
            plen = prefix_lens_list[s]
            seq_indices = unified_indices[start:end].tolist()
            p_start = prefix_indptr[s]
            expected_prefix = list(range(1000 + p_start, 1000 + p_start + plen))
            e_start = extend_start_loc_list[s]
            elen = extend_lens_list[s]
            expected_extend = list(range(2000 + e_start, 2000 + e_start + elen))
            assert seq_indices == expected_prefix + expected_extend, f"Seq {s} mismatch"

    def test_unified_some_prefix_zero(self):
        prefix_kv_indptr = paddle.to_tensor([0, 0, 10, 10], dtype="int32")
        prefix_kv_indices = paddle.arange(10, dtype="int32") + 500
        extend_seq_lens = paddle.to_tensor([5, 3, 8], dtype="int32")
        extend_start_loc = paddle.to_tensor([0, 5, 8], dtype="int32")
        extend_kv_indices = paddle.arange(16, dtype="int32") + 800
        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs=3,
        )
        assert plens.tolist() == [0, 10, 0]
        assert unified_indices[0:5].tolist() == [800, 801, 802, 803, 804]
        assert unified_indices[5:18].tolist() == list(range(500, 510)) + [805, 806, 807]

    def test_unified_extend_one(self):
        bs, block_size = 2, 4
        prefix_lens = [10, 5]
        extend_lens = [1, 1]
        block_tables = paddle.to_tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="int32")
        prefix_lens_t = paddle.to_tensor(prefix_lens, dtype="int32")
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )
        total_lens_t = paddle.to_tensor([p + e for p, e in zip(prefix_lens, extend_lens)], dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)
        extend_seq_lens = paddle.to_tensor(extend_lens, dtype="int32")
        extend_start_loc = paddle.to_tensor([0, 1], dtype="int32")
        extend_kv_indices = paddle.empty([sum(extend_lens)], dtype="int32")
        for s in range(bs):
            src_start = int(all_kv_indptr[s].item()) + prefix_lens[s]
            dst_start = int(extend_start_loc[s].item())
            extend_kv_indices[dst_start : dst_start + extend_lens[s]] = all_kv_indices[
                src_start : src_start + extend_lens[s]
            ]
        unified_indptr, _, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )
        assert plens.tolist() == prefix_lens
        assert unified_indptr.tolist() == [0, 11, 17]

    @pytest.mark.parametrize("bs_mode", ["bs1", "bs3"])
    def test_unified_extend_start_loc(self, bs_mode):
        if bs_mode == "bs1":
            prefix_kv_indptr = paddle.to_tensor([0, 3], dtype="int32")
            prefix_kv_indices = paddle.to_tensor([10, 11, 12], dtype="int32")
            extend_seq_lens = paddle.to_tensor([2], dtype="int32")
            extend_start_loc = paddle.to_tensor([0], dtype="int32")
            extend_kv_indices = paddle.to_tensor([20, 21], dtype="int32")
            unified_indptr, unified_indices, _ = build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs=1,
            )
            assert unified_indptr.tolist() == [0, 5]
            assert unified_indices[:5].tolist() == [10, 11, 12, 20, 21]
        else:
            prefix_kv_indptr = paddle.to_tensor([0, 2, 5, 5], dtype="int32")
            prefix_kv_indices = paddle.to_tensor([10, 11, 20, 21, 22], dtype="int32")
            extend_seq_lens = paddle.to_tensor([3, 2, 4], dtype="int32")
            extend_start_loc = paddle.to_tensor([0, 3, 5], dtype="int32")
            extend_kv_indices = paddle.to_tensor([100, 101, 102, 200, 201, 300, 301, 302, 303], dtype="int32")
            unified_indptr, unified_indices, plens = build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs=3,
            )
            assert plens.tolist() == [2, 3, 0]
            assert unified_indptr.tolist() == [0, 5, 10, 14]
            assert unified_indices[:5].tolist() == [10, 11, 100, 101, 102]
            assert unified_indices[5:10].tolist() == [20, 21, 22, 200, 201]
            assert unified_indices[10:14].tolist() == [300, 301, 302, 303]

    def test_unified_large_batch_stress(self):
        bs, block_size = 32, 16
        rng = np.random.RandomState(42)
        prefix_lens_list = rng.randint(0, 64, size=bs).tolist()
        extend_lens_list = rng.randint(1, 32, size=bs).tolist()
        max_total = max(p + e for p, e in zip(prefix_lens_list, extend_lens_list))
        max_blocks_per_seq = (max_total + block_size - 1) // block_size
        block_tables_np = np.zeros([bs, max_blocks_per_seq], dtype=np.int32)
        next_block = 0
        for s in range(bs):
            total = prefix_lens_list[s] + extend_lens_list[s]
            n_blocks = (total + block_size - 1) // block_size
            for b in range(n_blocks):
                block_tables_np[s, b] = next_block
                next_block += 1
        block_tables = paddle.to_tensor(block_tables_np, dtype="int32")
        prefix_lens_t = paddle.to_tensor(prefix_lens_list, dtype="int32")
        extend_seq_lens = paddle.to_tensor(extend_lens_list, dtype="int32")
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )
        total_lens = [p + e for p, e in zip(prefix_lens_list, extend_lens_list)]
        total_lens_t = paddle.to_tensor(total_lens, dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)
        extend_start_loc_list = [0]
        for e in extend_lens_list[:-1]:
            extend_start_loc_list.append(extend_start_loc_list[-1] + e)
        extend_start_loc = paddle.to_tensor(extend_start_loc_list, dtype="int32")
        total_extend = sum(extend_lens_list)
        extend_kv_indices = paddle.empty([max(total_extend, 1)], dtype="int32")
        for s in range(bs):
            plen = prefix_lens_list[s]
            elen = extend_lens_list[s]
            if elen == 0:
                continue
            src_start = int(all_kv_indptr[s].item()) + plen
            dst_start = int(extend_start_loc[s].item())
            extend_kv_indices[dst_start : dst_start + elen] = all_kv_indices[src_start : src_start + elen]
        unified_indptr, _, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )
        assert plens.tolist() == prefix_lens_list
        expected_indptr = [0]
        for p, e in zip(prefix_lens_list, extend_lens_list):
            expected_indptr.append(expected_indptr[-1] + p + e)
        assert unified_indptr.tolist() == expected_indptr

    def test_unified_all_zero_extend(self):
        prefix_kv_indptr = paddle.to_tensor([0, 3, 5], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 12, 20, 21], dtype="int32")
        extend_seq_lens = paddle.zeros([2], dtype="int32")
        extend_start_loc = paddle.zeros([2], dtype="int32")
        extend_kv_indices = paddle.zeros([0], dtype="int32")
        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs=2,
        )
        assert plens.tolist() == [3, 2]
        assert unified_indptr.tolist() == [0, 3, 5]
        assert unified_indices[:5].tolist() == [10, 11, 12, 20, 21]


# ===========================================================================
# 3. Kernel correctness tests (parametrized)
# ===========================================================================


class TestKernelCorrectness:
    """Unified kernel correctness against naive reference, covering all parameter combinations."""

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "num_q,num_kv,dim,bs,q_len,blk,prefix,causal,dtype,atol,label",
        [
            # Original basic tests
            (4, 4, 64, 2, 8, 4, 0, True, "float16", FP16_ATOL, "MHA-d64"),
            (4, 4, 128, 2, 8, 4, 0, True, "float16", FP16_ATOL, "MHA-d128"),
            (8, 2, 64, 2, 8, 4, 0, True, "float16", FP16_ATOL, "GQA4:1-d64"),
            (8, 2, 128, 2, 8, 4, 0, True, "float16", FP16_ATOL, "GQA4:1-d128"),
            # With prefix
            (4, 4, 64, 1, 3, 4, 4, True, "float16", FP16_ATOL, "prefix-4"),
            # MQA
            (32, 1, 64, 1, 8, 8, 0, True, "float16", MQA_ATOL, "MQA"),
            # Large head_dim
            (4, 4, 256, 1, 8, 8, 0, True, "float16", LARGE_HEAD_ATOL, "d256"),
            # bfloat16
            (4, 4, 64, 2, 8, 4, 0, True, "bfloat16", BF16_ATOL, "bf16"),
            # Large batch
            (4, 4, 64, 8, 16, 16, 0, True, "float16", MQA_ATOL, "bs8"),
            # Non-causal
            (4, 4, 64, 1, 8, 4, 0, False, "float16", FP16_ATOL * 2, "non-causal"),
            # Non-standard head_dim
            (4, 4, 96, 2, 16, 16, 0, True, "float16", NONSTANDARD_HEAD_ATOL, "d96"),
            (4, 4, 80, 2, 16, 16, 0, True, "float16", NONSTANDARD_HEAD_ATOL, "d80"),
            (4, 4, 13, 1, 8, 4, 0, True, "float16", PRIME_HEAD_ATOL, "d13-prime"),
            (12, 4, 96, 2, 16, 16, 0, True, "float16", NONSTANDARD_HEAD_ATOL, "d96-GQA"),
            (4, 4, 80, 1, 8, 8, 16, True, "float16", NONSTANDARD_HEAD_ATOL, "d80-prefix"),
            # Boundary: minimal input
            (4, 4, 64, 1, 1, 4, 0, True, "float16", FP16_ATOL, "bs1-ext1"),
            # Boundary: q_len not aligned to block_size
            (4, 4, 64, 1, 7, 4, 0, True, "float16", FP16_ATOL, "q7-blk4"),
        ],
    )
    def test_correctness(self, num_q, num_kv, dim, bs, q_len, blk, prefix, causal, dtype, atol, label):
        max_diff, cos_sim = _run_kernel_vs_ref(bs, q_len, num_q, num_kv, dim, blk, prefix, causal, dtype)
        print(f"\n[{label}] max_diff={max_diff:.6e}, cos_sim={cos_sim:.10f}")
        assert max_diff < atol, f"[{label}] max_diff={max_diff} exceeds {atol}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"[{label}] low cosine sim: {cos_sim}"

    @pytest.mark.gpu
    def test_explicit_sm_scale(self):
        """Custom sm_scale instead of default 1/sqrt(head_dim)."""
        max_diff, _ = _run_kernel_vs_ref(1, 8, 4, 4, 64, 4, sm_scale=0.125)
        assert max_diff < FP16_ATOL, f"Custom sm_scale failed: max_diff={max_diff}"

    @pytest.mark.gpu
    def test_non_contiguous_blocks(self):
        """Non-contiguous physical block IDs must produce same result as sequential."""
        q_len, num_q, num_kv, dim, blk = 8, 4, 4, 64, 4
        total_blocks = 20
        paddle.seed(42)
        q_flat = paddle.randn([q_len, num_q, dim]).astype("float16")
        k_flat = paddle.randn([q_len, num_kv, dim]).astype("float16")
        v_flat = paddle.randn([q_len, num_kv, dim]).astype("float16")
        num_blocks_needed = (q_len + blk - 1) // blk
        np.random.seed(42)
        physical_blocks = sorted(np.random.choice(total_blocks, num_blocks_needed, replace=False))
        cache_k = paddle.zeros([total_blocks, num_kv, blk, dim], dtype="float16")
        cache_v = paddle.zeros([total_blocks, num_kv, blk, dim], dtype="float16")
        for t in range(q_len):
            pb = physical_blocks[t // blk]
            off = t % blk
            cache_k[pb, :, off, :] = k_flat[t]
            cache_v[pb, :, off, :] = v_flat[t]
        kv_indices = paddle.to_tensor(
            [int(physical_blocks[t // blk]) * blk + t % blk for t in range(q_len)], dtype="int32"
        )
        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")
        o = paddle.zeros([q_len, num_q, dim], dtype="float16")
        o = extend_attention_fwd_unified(
            q_flat,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q,
            num_kv,
            dim,
            q_len,
            True,
        )
        # Compare with sequential layout using shared helper
        o_ref = _run_single_seq_kernel(q_flat, k_flat, v_flat, blk, num_q, num_kv, dim)
        max_diff = float(paddle.max(paddle.abs(o.astype("float32") - o_ref.astype("float32"))).item())
        assert max_diff < 1e-5, f"Non-contiguous blocks differ: max_diff={max_diff}"

    @pytest.mark.gpu
    def test_long_sequence_4096_no_nan(self):
        """Long sequence (4096) should not produce NaN/Inf."""
        seq_len, num_q, num_kv, dim, blk = 4096, 4, 4, 64, 64
        paddle.seed(42)
        q_flat = paddle.randn([seq_len, num_q, dim]).astype("float16")
        k_flat = paddle.randn([seq_len, num_kv, dim]).astype("float16")
        v_flat = paddle.randn([seq_len, num_kv, dim]).astype("float16")
        o = _run_single_seq_kernel(q_flat, k_flat, v_flat, blk, num_q, num_kv, dim)
        assert not paddle.any(paddle.isnan(o)).item(), "NaN in output"
        assert not paddle.any(paddle.isinf(o)).item(), "Inf in output"
        assert float(paddle.abs(o).mean().item()) > 1e-4, "Output is degenerate"

    @pytest.mark.gpu
    def test_large_values_no_nan(self):
        """Input near fp16 range should not produce NaN/Inf."""
        q_len, num_q, num_kv, dim, blk = 8, 4, 4, 64, 4
        paddle.seed(42)
        q_flat = (paddle.randn([q_len, num_q, dim]) * 10.0).astype("float16")
        k_flat = (paddle.randn([q_len, num_kv, dim]) * 10.0).astype("float16")
        v_flat = (paddle.randn([q_len, num_kv, dim]) * 10.0).astype("float16")
        o = _run_single_seq_kernel(q_flat, k_flat, v_flat, blk, num_q, num_kv, dim)
        assert not paddle.any(paddle.isnan(o)).item(), "NaN with large values"
        assert not paddle.any(paddle.isinf(o)).item(), "Inf with large values"


# ===========================================================================
# 4. Split invariance tests (core feature)
# ===========================================================================


class TestSplitInvariance:
    """
    Core test: the unified kernel must produce the same attention output
    regardless of how the sequence is split into prefix (cached) and extend (new).
    """

    def _run_with_split(
        self, q_all, k_all, v_all, total_len, prefix_len, block_size, num_q_heads, num_kv_heads, head_dim
    ):
        extend_len = total_len - prefix_len
        bs = 1
        q_extend = q_all[prefix_len:total_len]
        cache_k, cache_v = _build_paged_kv_cache(k_all[:total_len], v_all[:total_len], block_size)
        num_blocks = (total_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)
        prefix_lens_t = paddle.to_tensor([prefix_len], dtype="int32")
        extend_seq_lens = paddle.to_tensor([extend_len], dtype="int32")
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )
        total_lens_t = paddle.to_tensor([total_len], dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)
        extend_start_loc = paddle.zeros([1], dtype="int32")
        extend_kv_indices = all_kv_indices[prefix_len : prefix_len + extend_len].clone()
        unified_kv_indptr, unified_kv_indices, _ = build_unified_kv_indices(
            prefix_kv_indptr, prefix_kv_indices, extend_start_loc, extend_seq_lens, extend_kv_indices, bs
        )
        qo_indptr = paddle.to_tensor([0, extend_len], dtype="int32")
        o = paddle.zeros([extend_len, num_q_heads, head_dim], dtype=q_extend.dtype)
        o = extend_attention_fwd_unified(
            q_extend,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens_t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            extend_len,
            True,
        )
        return o

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "total_len,prefix_a,prefix_b,num_q,num_kv,dim,blk,seed,dtype",
        [
            (400, 0, 384, 4, 4, 128, 64, 42, "float16"),  # basic: cache miss vs hit
            (256, 0, 192, 8, 2, 128, 64, 123, "float16"),  # GQA
            (32, 0, 5, 4, 4, 64, 4, 42, "float16"),  # non-aligned prefix
            (825, 0, 768, 28, 4, 128, 64, 123, "bfloat16"),  # Qwen2.5-7B real-world cache hit
            (128, 0, 64, 28, 4, 128, 64, 42, "bfloat16"),  # bf16 half prefix
            (128, 0, 120, 28, 4, 128, 64, 42, "bfloat16"),  # bf16 mostly prefix
        ],
    )
    def test_split_invariance_pairwise(self, total_len, prefix_a, prefix_b, num_q, num_kv, dim, blk, seed, dtype):
        paddle.seed(seed)
        q_all = paddle.randn([total_len, num_q, dim]).astype(dtype)
        k_all = paddle.randn([total_len, num_kv, dim]).astype(dtype)
        v_all = paddle.randn([total_len, num_kv, dim]).astype(dtype)
        out_a = self._run_with_split(q_all, k_all, v_all, total_len, prefix_a, blk, num_q, num_kv, dim)
        out_a_tail = out_a[prefix_b:]
        out_b = self._run_with_split(q_all, k_all, v_all, total_len, prefix_b, blk, num_q, num_kv, dim)
        out_a_f32 = out_a_tail.astype("float32").numpy()
        out_b_f32 = out_b.astype("float32").numpy()
        assert np.array_equal(out_a_f32, out_b_f32), (
            f"Split invariance FAILED: not bit-identical, " f"max_diff={np.abs(out_a_f32 - out_b_f32).max()}"
        )

    @pytest.mark.gpu
    def test_split_invariance_multiple_splits(self):
        """Multiple different splits all produce the same result for the last 16 tokens."""
        total_len, extend_len = 128, 16
        num_q, num_kv, dim, blk = 4, 4, 64, 16
        paddle.seed(777)
        q_all = paddle.randn([total_len, num_q, dim]).astype("float16")
        k_all = paddle.randn([total_len, num_kv, dim]).astype("float16")
        v_all = paddle.randn([total_len, num_kv, dim]).astype("float16")
        prefix_lens_to_test = [0, 16, 48, 64, 96, 112]
        results = []
        for plen in prefix_lens_to_test:
            out = self._run_with_split(q_all, k_all, v_all, total_len, plen, blk, num_q, num_kv, dim)
            assert out.shape[0] >= extend_len
            results.append(out[-extend_len:].astype("float32").numpy())
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), (
                f"prefix={prefix_lens_to_test[i]} vs 0: not bit-identical, "
                f"max_diff={np.abs(results[0] - results[i]).max()}"
            )


# ===========================================================================
# 5. Determinism tests (parametrized)
# ===========================================================================


class TestDeterminism:
    """Verify kernel produces bitwise identical results across multiple runs."""

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "bs,q_len,num_q,num_kv,dim,blk,prefix,runs",
        [
            (2, 8, 8, 4, 64, 4, 0, 5),  # original basic
            (2, 16, 8, 4, 128, 16, 0, 10),  # strengthened no-prefix
            (1, 8, 8, 4, 64, 16, 32, 10),  # with prefix
            (4, 16, 16, 2, 128, 16, 0, 10),  # GQA large batch
        ],
    )
    def test_determinism(self, bs, q_len, num_q, num_kv, dim, blk, prefix, runs):
        _run_determinism_check(bs, q_len, num_q, num_kv, dim, blk, prefix, runs)


# ===========================================================================
# 6. Production-scale correctness
# ===========================================================================


class TestProductionScaleCorrectness:

    def _run_large_scale_attention(
        self,
        bs,
        seq_lens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_size,
        prefix_lens_list=None,
        dtype="float16",
    ):
        if prefix_lens_list is None:
            prefix_lens_list = [0] * bs
        extend_lens = [s - p for s, p in zip(seq_lens, prefix_lens_list)]
        paddle.seed(42)
        all_ref_outputs, all_q_flat, all_k_flat, all_v_flat = [], [], [], []
        for b in range(bs):
            q_len_b, kv_len_b = extend_lens[b], seq_lens[b]
            q_b = paddle.randn([1, q_len_b, num_q_heads, head_dim]).astype(dtype)
            k_b = paddle.randn([1, kv_len_b, num_kv_heads, head_dim]).astype(dtype)
            v_b = paddle.randn([1, kv_len_b, num_kv_heads, head_dim]).astype(dtype)
            prefix_t = paddle.to_tensor([prefix_lens_list[b]], dtype="int32")
            ref_out_b = naive_attention(q_b, k_b, v_b, prefix_t, is_causal=True)
            all_ref_outputs.append(ref_out_b.reshape([q_len_b, num_q_heads, head_dim]))
            all_q_flat.append(q_b.reshape([q_len_b, num_q_heads, head_dim]))
            all_k_flat.append(k_b.reshape([kv_len_b, num_kv_heads, head_dim]))
            all_v_flat.append(v_b.reshape([kv_len_b, num_kv_heads, head_dim]))
        k_all_flat = paddle.concat(all_k_flat, axis=0)
        v_all_flat = paddle.concat(all_v_flat, axis=0)
        cache_k, cache_v = _build_paged_kv_cache(k_all_flat, v_all_flat, block_size)
        q_flat = paddle.concat(all_q_flat, axis=0)
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr_list, kv_indptr_list, kv_indices_list, prefix_lens_t_list = [0], [0], [], []
        kv_offset = 0
        for b in range(bs):
            qo_indptr_list.append(qo_indptr_list[-1] + extend_lens[b])
            kv_indptr_list.append(kv_indptr_list[-1] + seq_lens[b])
            for t in range(seq_lens[b]):
                kv_indices_list.append(kv_offset + t)
            kv_offset += seq_lens[b]
            prefix_lens_t_list.append(prefix_lens_list[b])
        qo_indptr = paddle.to_tensor(qo_indptr_list, dtype="int32")
        kv_indptr = paddle.to_tensor(kv_indptr_list, dtype="int32")
        kv_indices = paddle.to_tensor(kv_indices_list, dtype="int32")
        prefix_lens_t = paddle.to_tensor(prefix_lens_t_list, dtype="int32")
        o_flat = extend_attention_fwd_unified(
            q_flat,
            o_flat,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens_t,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max(extend_lens),
            True,
        )
        ref_concat = paddle.concat(all_ref_outputs, axis=0)
        ref_fp32 = ref_concat.astype("float32")
        triton_fp32 = o_flat.astype("float32")
        max_diff = float(paddle.max(paddle.abs(ref_fp32 - triton_fp32)).item())
        cos_sim = cosine_similarity(ref_fp32, triton_fp32)
        assert not paddle.any(paddle.isnan(o_flat)).item(), "Output contains NaN"
        assert not paddle.any(paddle.isinf(o_flat)).item(), "Output contains Inf"
        return max_diff, cos_sim

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "bs,seq_lens,num_q,num_kv,dim,blk,prefix_lens,label",
        [
            (19, None, 12, 4, 128, 64, None, "bs19-sglang-scale"),
            (1, [4096], 4, 4, 64, 64, None, "seq4096"),
            (8, [32, 256, 64, 1024, 16, 512, 128, 2048], 8, 2, 128, 64, None, "mixed-len"),
            (12, None, 8, 4, 128, 64, "random", "bs12-prefix"),
        ],
    )
    def test_production_scale(self, bs, seq_lens, num_q, num_kv, dim, blk, prefix_lens, label):
        rng = np.random.RandomState(42 if label != "bs12-prefix" else 123)
        if seq_lens is None:
            seq_lens = rng.randint(64, 512, size=bs).tolist()
        if prefix_lens == "random":
            prefix_lens = rng.randint(0, 256, size=bs).tolist()
            seq_lens = [max(s, p + 1) for s, p in zip(seq_lens, prefix_lens)]
        elif prefix_lens is None:
            prefix_lens = None
        max_diff, cos_sim = self._run_large_scale_attention(bs, seq_lens, num_q, num_kv, dim, blk, prefix_lens)
        print(f"\n[{label}] max_diff={max_diff:.6e}, cos_sim={cos_sim:.10f}")
        assert max_diff < LARGE_SCALE_ATOL, f"[{label}] max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"[{label}] low cosine sim: {cos_sim}"


# ===========================================================================
# 7. Cross-validation (reference implementations + triton vs sdpa)
# ===========================================================================


class TestCrossValidation:

    @pytest.mark.parametrize("num_kv,num_q", [(4, 4), (2, 8), (1, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_refs_match_no_prefix(self, num_kv, num_q, head_dim):
        bs, q_len = 2, 16
        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv, head_dim]).astype("float16")
        prefix_lens = paddle.zeros([bs], dtype="int32")
        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=True)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True)
        max_diff = float(paddle.max(paddle.abs(out_naive.astype("float32") - out_sdpa.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Reference mismatch: max_diff={max_diff}"

    @pytest.mark.parametrize("prefix_len", [4, 16, 32])
    def test_refs_match_with_prefix(self, prefix_len):
        bs, q_len = 1, 8
        kv_len = prefix_len + q_len
        paddle.seed(123)
        q = paddle.randn([bs, q_len, 4, 64]).astype("float16")
        k = paddle.randn([bs, kv_len, 4, 64]).astype("float16")
        v = paddle.randn([bs, kv_len, 4, 64]).astype("float16")
        prefix_lens = paddle.to_tensor([prefix_len], dtype="int32")
        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=True)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True)
        max_diff = float(paddle.max(paddle.abs(out_naive.astype("float32") - out_sdpa.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Reference mismatch w/ prefix={prefix_len}: max_diff={max_diff}"

    def test_refs_match_non_causal(self):
        bs, q_len = 2, 8
        paddle.seed(42)
        q = paddle.randn([bs, q_len, 4, 64]).astype("float16")
        k = paddle.randn([bs, q_len, 4, 64]).astype("float16")
        v = paddle.randn([bs, q_len, 4, 64]).astype("float16")
        prefix_lens = paddle.zeros([bs], dtype="int32")
        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=False)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=False)
        max_diff = float(paddle.max(paddle.abs(out_naive.astype("float32") - out_sdpa.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Non-causal ref mismatch: max_diff={max_diff}"

    @pytest.mark.gpu
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_triton_matches_sdpa(self, head_dim):
        """Triple validation: triton vs sdpa reference."""
        max_diff, cos_sim = _run_kernel_vs_ref(
            2,
            16,
            8,
            4,
            head_dim,
            16,
            ref_fn=sdpa_attention_reference,
        )
        assert max_diff < FP16_ATOL, f"Triton vs SDPA mismatch: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
