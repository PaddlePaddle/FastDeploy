"""
Tests for unified extend attention kernel.
Verifies correctness against naive attention and determinism across runs.
"""

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
# fp16 attention: standard O(1e-3) error from fused softmax + dot
FP16_ATOL = 1e-2
# bfloat16: 8-bit mantissa → ~2x looser than fp16
BF16_ATOL = 5e-2
# MQA (kv_heads=1, large group_size): GQA broadcast amplifies rounding
MQA_ATOL = 5e-2
# head_dim>=256: larger dot products → more fp16 accumulation error
LARGE_HEAD_ATOL = 0.1
# Non-standard head_dim (e.g., 96, 80): slightly looser due to non-power-of-2
NONSTANDARD_HEAD_ATOL = 0.05
# Prime head_dim (e.g., 13): very small dim, accumulation errors amplified
PRIME_HEAD_ATOL = 0.1
# Large-scale correctness (long sequences): relaxed rtol for cumulative error
LARGE_SCALE_RTOL = 1e-2
LARGE_SCALE_ATOL = 0.02
# Cosine similarity threshold (1 - eps): for long sequence validation
COSINE_SIM_THRESHOLD = 1 - 1e-4
# Determinism: must be bitwise identical
DETERMINISM_ATOL = 0.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two tensors (flattened).

    Args:
        a, b: paddle tensors of same shape
    Returns:
        float: cosine similarity in [0, 1]
    """
    a_flat = a.astype("float32").reshape([-1])
    b_flat = b.astype("float32").reshape([-1])
    dot = float(paddle.sum(a_flat * b_flat).item())
    norm_a = float(paddle.sqrt(paddle.sum(a_flat * a_flat)).item())
    norm_b = float(paddle.sqrt(paddle.sum(b_flat * b_flat)).item())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True):
    """
    Second reference implementation using paddle's scaled_dot_product_attention.

    This provides cross-validation against naive_attention to catch bugs
    in either reference implementation.

    Args:
        q: [bs, q_len, num_heads, head_dim]
        k: [bs, kv_len, num_kv_heads, head_dim]
        v: [bs, kv_len, num_kv_heads, head_dim]
        prefix_lens: [bs], prefix length per sequence
        is_causal: whether to apply causal mask on extend part
    Returns:
        output: [bs, q_len, num_heads, head_dim]
    """
    bs, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    num_kv_heads = k.shape[2]
    group_size = num_heads // num_kv_heads

    # Expand KV for GQA
    k = k.unsqueeze(3).expand([-1, -1, -1, group_size, -1])
    k = k.reshape([bs, kv_len, num_heads, head_dim])
    v = v.unsqueeze(3).expand([-1, -1, -1, group_size, -1])
    v = v.reshape([bs, kv_len, num_heads, head_dim])

    # Transpose to [bs, num_heads, seq_len, head_dim] for SDPA
    q_t = q.transpose([0, 2, 1, 3]).astype("float32")
    k_t = k.transpose([0, 2, 1, 3]).astype("float32")
    v_t = v.transpose([0, 2, 1, 3]).astype("float32")

    # Build attention mask: [bs, 1, q_len, kv_len]
    # prefix region: no mask (all attend); extend region: causal
    mask = paddle.zeros([bs, 1, q_len, kv_len], dtype="float32")
    if is_causal:
        for b in range(bs):
            plen = int(prefix_lens[b].item())
            for qi in range(q_len):
                for ki in range(kv_len):
                    if ki >= plen:
                        k_in_extend = ki - plen
                        if qi < k_in_extend:
                            mask[b, :, qi, ki] = float("-inf")

    # Manual SDPA: softmax(Q@K^T / sqrt(d) + mask) @ V
    scale = 1.0 / (head_dim**0.5)
    scores = paddle.matmul(q_t, k_t.transpose([0, 1, 3, 2])) * scale + mask
    attn = paddle.nn.functional.softmax(scores, axis=-1)
    out = paddle.matmul(attn, v_t)

    # Transpose back to [bs, q_len, num_heads, head_dim]
    return out.transpose([0, 2, 1, 3])


def naive_attention(q, k, v, prefix_lens, is_causal=True):
    """
    Naive multi-head attention reference implementation.

    Args:
        q: [bs, q_len, num_heads, head_dim]
        k: [bs, kv_len, num_kv_heads, head_dim]
        v: [bs, kv_len, num_kv_heads, head_dim]
        prefix_lens: [bs], prefix length per sequence
        is_causal: whether to apply causal mask on extend part
    Returns:
        output: [bs, q_len, num_heads, head_dim]
    """
    bs, q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[1]
    num_kv_heads = k.shape[2]
    group_size = num_heads // num_kv_heads

    # Expand KV for GQA
    k = k.unsqueeze(3).expand([-1, -1, -1, group_size, -1])
    k = k.reshape([bs, kv_len, num_heads, head_dim])
    v = v.unsqueeze(3).expand([-1, -1, -1, group_size, -1])
    v = v.reshape([bs, kv_len, num_heads, head_dim])

    # [bs, num_heads, q_len, kv_len]
    scale = 1.0 / (head_dim**0.5)
    scores = paddle.einsum("bqhd,bkhd->bhqk", q, k) * scale

    if is_causal:
        for b in range(bs):
            plen = int(prefix_lens[b].item())
            for qi in range(q_len):
                for ki in range(kv_len):
                    if ki >= plen:
                        # extend region: causal mask
                        k_in_extend = ki - plen
                        if qi < k_in_extend:
                            scores[b, :, qi, ki] = float("-inf")

    attn = paddle.nn.functional.softmax(scores, axis=-1)
    out = paddle.einsum("bhqk,bkhd->bqhd", attn, v)
    return out


def _build_paged_kv_cache(k_flat, v_flat, block_size):
    """
    Pack flat KV tensors into paged cache format.

    Args:
        k_flat: [total_tokens, num_kv_heads, head_dim]
        v_flat: [total_tokens, num_kv_heads, head_dim]
        block_size: tokens per block
    Returns:
        cache_k: [num_blocks, num_kv_heads, block_size, head_dim]
        cache_v: same shape
        token_to_cache_idx: [total_tokens] mapping token -> flat cache position
    """
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


class TestTritonCumsumWithZeroPrefix:
    """Direct tests for triton_cumsum_with_zero_prefix."""

    def test_basic(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        x = paddle.to_tensor([3, 1, 4, 1, 5], dtype="int32")
        result = triton_cumsum_with_zero_prefix(x)
        assert result.tolist() == [0, 3, 4, 8, 9, 14]

    def test_single_element(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        x = paddle.to_tensor([7], dtype="int32")
        result = triton_cumsum_with_zero_prefix(x)
        assert result.tolist() == [0, 7]

    def test_empty(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        x = paddle.to_tensor([], dtype="int32")
        result = triton_cumsum_with_zero_prefix(x, n=0)
        assert result.tolist() == [0]

    def test_all_zeros(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        x = paddle.zeros([4], dtype="int32")
        result = triton_cumsum_with_zero_prefix(x)
        assert result.tolist() == [0, 0, 0, 0, 0]

    def test_with_explicit_n(self):
        """Use n < x.shape[0] to only cumsum the first n elements."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        x = paddle.to_tensor([2, 3, 5, 7, 11], dtype="int32")
        result = triton_cumsum_with_zero_prefix(x, n=3)
        assert result.tolist() == [0, 2, 5, 10]

    def test_matches_paddle_reference(self):
        """Cross-validate against paddle.cumsum for random input."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            triton_cumsum_with_zero_prefix,
        )

        np.random.seed(42)
        for length in [1, 7, 32, 64, 127, 128, 255]:
            x_np = np.random.randint(0, 100, size=length).astype(np.int32)
            x = paddle.to_tensor(x_np, dtype="int32")
            triton_result = triton_cumsum_with_zero_prefix(x)
            paddle_result = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(x).astype("int32")])
            assert triton_result.tolist() == paddle_result.tolist(), f"Mismatch at length={length}"


class TestBuildKvIndicesFromBlockTables:
    """Test block_tables -> flat kv_indices conversion."""

    def test_single_sequence(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_size = 4
        # seq of length 6: needs 2 blocks, block 0 -> phys 2, block 1 -> phys 5
        block_tables = paddle.to_tensor([[2, 5]], dtype="int32")
        seq_lens = paddle.to_tensor([6], dtype="int32")

        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs=1)

        assert kv_indptr.tolist() == [0, 6]
        expected = [
            2 * 4 + 0,
            2 * 4 + 1,
            2 * 4 + 2,
            2 * 4 + 3,  # block 2, offsets 0-3
            5 * 4 + 0,
            5 * 4 + 1,  # block 5, offsets 0-1
        ]
        assert kv_indices.tolist() == expected

    def test_multiple_sequences(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_size = 2
        # seq0: len=3, blocks=[1, 3], seq1: len=2, blocks=[0, 0]
        block_tables = paddle.to_tensor([[1, 3], [0, 0]], dtype="int32")
        seq_lens = paddle.to_tensor([3, 2], dtype="int32")

        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs=2)

        assert kv_indptr.tolist() == [0, 3, 5]
        expected_seq0 = [1 * 2 + 0, 1 * 2 + 1, 3 * 2 + 0]  # block1: t0,t1; block3: t2
        expected_seq1 = [0 * 2 + 0, 0 * 2 + 1]  # block0: t0,t1
        assert kv_indices.tolist() == expected_seq0 + expected_seq1

    def test_empty_sequence(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_size = 4
        block_tables = paddle.to_tensor([[0]], dtype="int32")
        seq_lens = paddle.to_tensor([0], dtype="int32")

        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs=1)

        assert kv_indptr.tolist() == [0, 0]

    def test_ref_vs_triton_cross_validation(self):
        """Cross-validate Triton version against Python reference implementation."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
            build_kv_indices_from_block_tables_ref,
        )

        rng = np.random.RandomState(123)
        for bs in [1, 4, 16]:
            block_size = 16
            seq_lens_np = rng.randint(0, 100, size=bs).astype(np.int32)
            max_seq = int(seq_lens_np.max()) if bs > 0 else 0
            max_blocks = (max_seq + block_size - 1) // block_size
            max_blocks = max(max_blocks, 1)
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


class TestBuildUnifiedKvIndices:
    """Test merging prefix + extend into unified indices."""

    def test_basic_merge(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        # prefix: seq0 has 2 tokens [10,11], seq1 has 1 token [20]
        prefix_kv_indptr = paddle.to_tensor([0, 2, 3], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 20], dtype="int32")

        # extend: seq0 has 3 tokens [100,101,102], seq1 has 2 tokens [200,201]
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
        # seq0: [10,11, 100,101,102], seq1: [20, 200,201]
        assert unified_indices[:8].tolist() == [10, 11, 100, 101, 102, 20, 200, 201]


class TestUnifiedExtendAttentionKernel:
    """Test the Triton attention kernel correctness against naive implementation."""

    @pytest.mark.parametrize("num_kv_heads,num_q_heads", [(4, 4), (2, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_correctness_no_prefix(self, num_kv_heads, num_q_heads, head_dim):
        """Test with no prefix (all extend) - should match naive causal attention."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 2, 8
        block_size = 4

        paddle.seed(42)
        q_batched = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k_batched = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v_batched = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")

        # Reference: naive attention
        prefix_lens_ref = paddle.zeros([bs], dtype="int32")
        ref_out = naive_attention(q_batched, k_batched, v_batched, prefix_lens_ref, is_causal=True)
        # [bs, q_len, num_heads, head_dim]

        # Build paged KV cache
        k_flat = k_batched.reshape([-1, num_kv_heads, head_dim])
        v_flat = v_batched.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        # Build indices
        total_tokens = bs * q_len
        q_flat = q_batched.reshape([total_tokens, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)

        seq_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])

        # All tokens are "extend" (no prefix)
        prefix_lens = paddle.zeros([bs], dtype="int32")

        # KV indices: sequential since we packed them in order
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(total_tokens, dtype="int32")

        o_flat = extend_attention_fwd_unified(
            q_flat,
            o_flat,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        ref_out_fp32 = ref_out.astype("float32")
        triton_out_fp32 = triton_out.astype("float32")

        max_diff = float(paddle.max(paddle.abs(ref_out_fp32 - triton_out_fp32)).item())
        assert max_diff < FP16_ATOL, f"Max diff {max_diff} exceeds tolerance {FP16_ATOL}"

    def test_correctness_with_prefix(self):
        """Test with prefix tokens - prefix should have no causal mask."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs = 1
        prefix_len = 4
        extend_len = 3
        kv_len = prefix_len + extend_len
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(123)
        q = paddle.randn([bs, extend_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")

        prefix_lens_t = paddle.to_tensor([prefix_len], dtype="int32")
        ref_out = naive_attention(q, k, v, prefix_lens_t, is_causal=True)

        # Build paged cache
        k_flat = k.reshape([-1, num_kv_heads, head_dim])
        v_flat = v.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        # Build indices
        q_flat = q.reshape([extend_len, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr = paddle.to_tensor([0, extend_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, kv_len], dtype="int32")
        kv_indices = paddle.arange(kv_len, dtype="int32")

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
            extend_len,
            True,
        )

        triton_out = o_flat.reshape([bs, extend_len, num_q_heads, head_dim])
        max_diff = float(paddle.max(paddle.abs(ref_out.astype("float32") - triton_out.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Max diff {max_diff} with prefix"

    def test_determinism(self):
        """Run the kernel multiple times, verify results are identical."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 2, 8
        num_q_heads, num_kv_heads, head_dim = 8, 4, 64
        block_size = 4

        paddle.seed(999)
        q_flat = paddle.randn([bs * q_len, num_q_heads, head_dim]).astype("float16")

        k_flat = paddle.randn([bs * q_len, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([bs * q_len, num_kv_heads, head_dim]).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        seq_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(bs * q_len, dtype="int32")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        results = []
        for _ in range(5):
            o = paddle.zeros_like(q_flat)
            o = extend_attention_fwd_unified(
                q_flat,
                o,
                cache_k,
                cache_v,
                qo_indptr,
                kv_indptr,
                kv_indices,
                prefix_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q_len,
                True,
            )
            results.append(o.astype("float32").numpy())

        for i in range(1, len(results)):
            assert (results[0] == results[i]).all(), f"Run 0 vs run {i} differ"


class TestSplitInvariance:
    """
    Core test: the unified kernel must produce the same attention output
    for the SAME logical sequence regardless of how it is split into
    prefix (cached) and extend (new) parts.

    This is the entire purpose of the unified kernel -- if this fails,
    cache hit vs miss will produce different results.
    """

    def _run_with_split(
        self, q_all, k_all, v_all, total_len, prefix_len, block_size, num_q_heads, num_kv_heads, head_dim
    ):
        """
        Run unified attention with a specific prefix/extend split.
        Returns output for the extend tokens only.
        """
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
            build_unified_kv_indices,
            extend_attention_fwd_unified,
        )

        extend_len = total_len - prefix_len
        bs = 1

        # Q is only for extend tokens
        q_extend = q_all[prefix_len:total_len]  # [extend_len, num_q_heads, head_dim]

        # Build paged KV cache from all K/V tokens
        cache_k, cache_v = _build_paged_kv_cache(k_all[:total_len], v_all[:total_len], block_size)

        # block_tables: simple sequential allocation
        num_blocks = (total_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks, dtype="int32").unsqueeze(0)  # [1, num_blocks]

        # Build indices
        prefix_lens_t = paddle.to_tensor([prefix_len], dtype="int32")
        extend_seq_lens = paddle.to_tensor([extend_len], dtype="int32")

        # prefix kv indices
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )

        # all kv indices
        total_lens_t = paddle.to_tensor([total_len], dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)

        # extend kv indices (the part after prefix in all_kv_indices)
        extend_start_loc = paddle.zeros([1], dtype="int32")
        extend_kv_indices = all_kv_indices[prefix_len : prefix_len + extend_len].clone()

        # unified indices
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

    def test_split_invariance_basic(self):
        """
        400 tokens total. Compare:
          Case A: prefix=0, extend=400 (cache miss)
          Case B: prefix=384, extend=16 (cache hit)
        The last 16 tokens' output should be identical.
        """
        total_len = 400
        prefix_len_a = 0
        prefix_len_b = 384
        num_q_heads, num_kv_heads, head_dim = 4, 4, 128
        block_size = 64

        paddle.seed(42)
        q_all = paddle.randn([total_len, num_q_heads, head_dim]).astype("float16")
        k_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")
        v_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")

        # Case A: all extend (cache miss)
        out_a = self._run_with_split(
            q_all, k_all, v_all, total_len, prefix_len_a, block_size, num_q_heads, num_kv_heads, head_dim
        )
        # Only take the last 16 tokens (same as Case B's extend)
        out_a_last16 = out_a[prefix_len_b:]  # [16, heads, dim]

        # Case B: 384 prefix + 16 extend (cache hit)
        out_b = self._run_with_split(
            q_all, k_all, v_all, total_len, prefix_len_b, block_size, num_q_heads, num_kv_heads, head_dim
        )

        diff = paddle.abs(out_a_last16.astype("float32") - out_b.astype("float32"))
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        print(f"\n[split_invariance_basic] max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        assert max_diff < FP16_ATOL, f"Split invariance FAILED: max_diff={max_diff}"

    def test_split_invariance_gqa(self):
        """Same test but with GQA (num_q_heads != num_kv_heads)."""
        total_len = 256
        prefix_len_a = 0
        prefix_len_b = 192
        num_q_heads, num_kv_heads, head_dim = 8, 2, 128
        block_size = 64

        paddle.seed(123)
        q_all = paddle.randn([total_len, num_q_heads, head_dim]).astype("float16")
        k_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")
        v_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")

        out_a = self._run_with_split(
            q_all, k_all, v_all, total_len, prefix_len_a, block_size, num_q_heads, num_kv_heads, head_dim
        )
        out_a_tail = out_a[prefix_len_b:]

        out_b = self._run_with_split(
            q_all, k_all, v_all, total_len, prefix_len_b, block_size, num_q_heads, num_kv_heads, head_dim
        )

        diff = paddle.abs(out_a_tail.astype("float32") - out_b.astype("float32"))
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        print(f"\n[split_invariance_gqa] max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        assert max_diff < FP16_ATOL, f"Split invariance FAILED (GQA): max_diff={max_diff}"

    def test_split_invariance_multiple_splits(self):
        """Test multiple different splits all produce the same result for the last 16 tokens."""
        total_len = 128
        extend_len = 16
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 16

        paddle.seed(777)
        q_all = paddle.randn([total_len, num_q_heads, head_dim]).astype("float16")
        k_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")
        v_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")

        prefix_lens_to_test = [0, 16, 48, 64, 96, 112]  # all extend=16 for the last 16 tokens
        results = []
        for plen in prefix_lens_to_test:
            out = self._run_with_split(
                q_all, k_all, v_all, total_len, plen, block_size, num_q_heads, num_kv_heads, head_dim
            )
            # out has shape [total_len - plen, heads, dim] (extend tokens only).
            # We always want the output for the last `extend_len` positions
            # (original positions [112:128]), which is the tail of every extend output.
            assert (
                out.shape[0] >= extend_len
            ), f"extend output too short: {out.shape[0]} < {extend_len} for prefix={plen}"
            out_last16 = out[-extend_len:]
            results.append(out_last16.astype("float32"))

        baseline = results[0]
        for i, res in enumerate(results[1:], 1):
            max_diff = float(paddle.max(paddle.abs(baseline - res)).item())
            print(f"\n[multi_split] prefix={prefix_lens_to_test[i]} vs prefix=0: max_diff={max_diff:.6e}")
            assert (
                max_diff < FP16_ATOL
            ), f"Split invariance FAILED: prefix={prefix_lens_to_test[i]} vs 0, max_diff={max_diff}"


# ===========================================================================
# ACTION-6: Index building tests (C1-C8)
# ===========================================================================


class TestBuildKvIndicesExtended:
    """Extended tests for build_kv_indices_from_block_tables (C1, C5, C6)."""

    # C1: build_unified_kv_indices large scale (bs=8)
    def test_c1_unified_large_bs(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        bs = 8
        prefix_lens_list = [10, 20, 5, 0, 15, 8, 30, 12]
        extend_lens_list = [5, 10, 3, 7, 2, 6, 4, 8]

        prefix_indptr = [0]
        for p in prefix_lens_list:
            prefix_indptr.append(prefix_indptr[-1] + p)
        prefix_kv_indptr = paddle.to_tensor(prefix_indptr, dtype="int32")

        total_prefix = sum(prefix_lens_list)
        prefix_kv_indices = paddle.arange(total_prefix, dtype="int32") + 1000  # arbitrary offset

        extend_seq_lens = paddle.to_tensor(extend_lens_list, dtype="int32")
        total_extend = sum(extend_lens_list)
        extend_start_loc_list = [0]
        for e in extend_lens_list[:-1]:
            extend_start_loc_list.append(extend_start_loc_list[-1] + e)
        extend_start_loc = paddle.to_tensor(extend_start_loc_list, dtype="int32")
        extend_kv_indices = paddle.arange(total_extend, dtype="int32") + 2000

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

        # Verify each sequence's unified indices = prefix_indices + extend_indices
        for s in range(bs):
            start = expected_indptr[s]
            end = expected_indptr[s + 1]
            plen = prefix_lens_list[s]
            seq_indices = unified_indices[start:end].tolist()
            # prefix part
            p_start = prefix_indptr[s]
            expected_prefix = list(range(1000 + p_start, 1000 + p_start + plen))
            # extend part
            e_start = extend_start_loc_list[s]
            elen = extend_lens_list[s]
            expected_extend = list(range(2000 + e_start, 2000 + e_start + elen))
            assert (
                seq_indices == expected_prefix + expected_extend
            ), f"Seq {s}: got {seq_indices}, expected {expected_prefix + expected_extend}"

    # C2: Some sequences prefix=0
    def test_c2_some_prefix_zero(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        bs = 3
        prefix_lens_list = [0, 10, 0]
        extend_lens_list = [5, 3, 8]

        prefix_indptr = [0]
        for p in prefix_lens_list:
            prefix_indptr.append(prefix_indptr[-1] + p)
        prefix_kv_indptr = paddle.to_tensor(prefix_indptr, dtype="int32")
        prefix_kv_indices = paddle.arange(sum(prefix_lens_list), dtype="int32") + 500

        extend_seq_lens = paddle.to_tensor(extend_lens_list, dtype="int32")
        extend_start_loc_list = [0]
        for e in extend_lens_list[:-1]:
            extend_start_loc_list.append(extend_start_loc_list[-1] + e)
        extend_start_loc = paddle.to_tensor(extend_start_loc_list, dtype="int32")
        extend_kv_indices = paddle.arange(sum(extend_lens_list), dtype="int32") + 800

        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )

        assert plens.tolist() == prefix_lens_list
        # seq0: no prefix, 5 extend → 5 unified indices (all extend)
        seq0 = unified_indices[0:5].tolist()
        assert seq0 == [800, 801, 802, 803, 804]
        # seq1: 10 prefix + 3 extend → 13 unified indices
        seq1 = unified_indices[5:18].tolist()
        assert seq1 == list(range(500, 510)) + [805, 806, 807]

    # C3: extend=1 (decode scenario)
    def test_c3_extend_one(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
            build_unified_kv_indices,
        )

        bs = 2
        block_size = 4
        prefix_lens = [10, 5]
        extend_lens = [1, 1]

        block_tables = paddle.to_tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            dtype="int32",
        )

        prefix_lens_t = paddle.to_tensor(prefix_lens, dtype="int32")
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )

        total_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        total_lens_t = paddle.to_tensor(total_lens, dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)

        extend_seq_lens = paddle.to_tensor(extend_lens, dtype="int32")
        extend_start_loc = paddle.to_tensor([0, 1], dtype="int32")

        # Extract extend indices
        extend_kv_indices = paddle.empty([sum(extend_lens)], dtype="int32")
        for s in range(bs):
            plen = prefix_lens[s]
            elen = extend_lens[s]
            src_start = int(all_kv_indptr[s].item()) + plen
            dst_start = int(extend_start_loc[s].item())
            extend_kv_indices[dst_start : dst_start + elen] = all_kv_indices[src_start : src_start + elen]

        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )

        assert plens.tolist() == prefix_lens
        assert unified_indptr.tolist() == [0, 11, 17]

    # C5: Sequences exceeding BLOCK_SIZE=128
    def test_c5_large_sequence(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_size = 64
        seq_len = 500  # exceeds multiple blocks
        bs = 1
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        block_tables = paddle.arange(num_blocks_needed, dtype="int32").unsqueeze(0)
        seq_lens = paddle.to_tensor([seq_len], dtype="int32")

        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs)

        assert kv_indptr.tolist() == [0, seq_len]
        assert len(kv_indices.tolist()) == seq_len
        # Verify each index
        for t in range(seq_len):
            expected = (t // block_size) * block_size + t % block_size
            assert kv_indices[t].item() == expected, f"Mismatch at t={t}"

    # C6: Non-contiguous block_table
    def test_c6_non_contiguous_blocks(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_size = 4
        bs = 1
        seq_len = 10
        # Physical blocks: [5, 2, 8] (non-contiguous)
        block_tables = paddle.to_tensor([[5, 2, 8]], dtype="int32")
        seq_lens = paddle.to_tensor([seq_len], dtype="int32")

        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs)

        expected = []
        for t in range(seq_len):
            bid = [5, 2, 8][t // block_size]
            expected.append(bid * block_size + t % block_size)
        assert kv_indices.tolist() == expected

    # C7: extend_start_loc bs=1 vs bs>1 branch
    def test_c7_extend_start_loc_bs1(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        # bs=1
        prefix_kv_indptr = paddle.to_tensor([0, 3], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 12], dtype="int32")
        extend_seq_lens = paddle.to_tensor([2], dtype="int32")
        extend_start_loc = paddle.to_tensor([0], dtype="int32")
        extend_kv_indices = paddle.to_tensor([20, 21], dtype="int32")

        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs=1,
        )
        assert unified_indptr.tolist() == [0, 5]
        assert unified_indices[:5].tolist() == [10, 11, 12, 20, 21]

    def test_c7_extend_start_loc_bs_gt1(self):
        """bs>1 branch: extend_start_loc built via cumsum."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        # bs=3
        prefix_kv_indptr = paddle.to_tensor([0, 2, 5, 5], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 20, 21, 22], dtype="int32")
        extend_seq_lens = paddle.to_tensor([3, 2, 4], dtype="int32")
        # cumsum of extend: [0, 3, 5]
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
        # seq0: prefix[10,11] + extend[100,101,102] = 5
        # seq1: prefix[20,21,22] + extend[200,201] = 5
        # seq2: prefix[] + extend[300,301,302,303] = 4
        assert unified_indptr.tolist() == [0, 5, 10, 14]
        assert unified_indices[:5].tolist() == [10, 11, 100, 101, 102]
        assert unified_indices[5:10].tolist() == [20, 21, 22, 200, 201]
        assert unified_indices[10:14].tolist() == [300, 301, 302, 303]

    # C8: Large batch stress test (bs=32)
    def test_c8_large_batch_stress(self):
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
            build_unified_kv_indices,
        )

        bs = 32
        block_size = 16
        rng = np.random.RandomState(42)
        prefix_lens_list = rng.randint(0, 64, size=bs).tolist()
        extend_lens_list = rng.randint(1, 32, size=bs).tolist()

        max_total = max(p + e for p, e in zip(prefix_lens_list, extend_lens_list))
        max_blocks_per_seq = (max_total + block_size - 1) // block_size

        # Assign physical blocks
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

        # Build prefix indices
        prefix_kv_indptr, prefix_kv_indices = build_kv_indices_from_block_tables(
            block_tables, prefix_lens_t, block_size, bs
        )

        # Build all indices
        total_lens = [p + e for p, e in zip(prefix_lens_list, extend_lens_list)]
        total_lens_t = paddle.to_tensor(total_lens, dtype="int32")
        all_kv_indptr, all_kv_indices = build_kv_indices_from_block_tables(block_tables, total_lens_t, block_size, bs)

        # Build extend start loc
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
        print(f"\n[C8] bs=32 stress test passed, total unified len={expected_indptr[-1]}")


class TestDeterministicBuildTritonIndices:
    """C4: Direct test of _deterministic_build_triton_indices with mock forward_meta."""

    def _make_mock_backend(self, block_size):
        from fastdeploy.model_executor.layers.attention.append_attn_backend import (
            AppendAttentionBackend,
        )

        backend = object.__new__(AppendAttentionBackend)
        backend.block_size = block_size
        return backend

    def _make_mock_forward_meta(self, seq_lens_this_time, prefix_lens, block_tables):
        from types import SimpleNamespace

        meta = SimpleNamespace()
        meta.seq_lens_this_time = seq_lens_this_time
        meta.prefix_lens = prefix_lens
        meta.block_tables = block_tables
        return meta

    def test_c4_basic(self):
        """Basic test with 2 sequences: one prefill, one decode."""
        block_size = 4
        backend = self._make_mock_backend(block_size)

        seq_lens_this_time = paddle.to_tensor([8, 1, 0, 0], dtype="int32")
        prefix_lens = paddle.to_tensor([0, 5, 0, 0], dtype="int32")

        # seq0: 8 new tokens, no prefix → total 8 tokens → 2 blocks
        # seq1: 1 new token, prefix=5 → total 6 tokens → 2 blocks
        block_tables = paddle.to_tensor(
            [
                [0, 1, 0, 0],
                [2, 3, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype="int32",
        )

        meta = self._make_mock_forward_meta(seq_lens_this_time, prefix_lens, block_tables)
        qo_indptr, unified_kv_indptr, unified_kv_indices, plens, bs, max_extend = (
            backend._deterministic_build_triton_indices(meta)
        )

        assert bs == 2
        assert max_extend == 8
        assert qo_indptr.tolist() == [0, 8, 9]
        assert plens.tolist() == [0, 5]
        # seq0: 0 prefix + 8 extend = 8 total KV
        # seq1: 5 prefix + 1 extend = 6 total KV
        assert unified_kv_indptr.tolist() == [0, 8, 14]

    def test_c4_all_decode(self):
        """All sequences are decode (extend_len=1)."""
        block_size = 4
        backend = self._make_mock_backend(block_size)

        seq_lens_this_time = paddle.to_tensor([1, 1, 1, 0], dtype="int32")
        prefix_lens = paddle.to_tensor([10, 20, 5, 0], dtype="int32")

        block_tables = paddle.to_tensor(
            [
                [0, 1, 2, 0],
                [3, 4, 5, 6],
                [7, 8, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype="int32",
        )

        meta = self._make_mock_forward_meta(seq_lens_this_time, prefix_lens, block_tables)
        qo_indptr, unified_kv_indptr, unified_kv_indices, plens, bs, max_extend = (
            backend._deterministic_build_triton_indices(meta)
        )

        assert bs == 3
        assert max_extend == 1
        assert qo_indptr.tolist() == [0, 1, 2, 3]
        assert plens.tolist() == [10, 20, 5]
        assert unified_kv_indptr.tolist() == [0, 11, 32, 38]

    def test_c4_mixed_prefill_decode(self):
        """Mixed: seq0 prefill (extend=16), seq1 decode (extend=1)."""
        block_size = 8
        backend = self._make_mock_backend(block_size)

        seq_lens_this_time = paddle.to_tensor([16, 1, 0], dtype="int32")
        prefix_lens = paddle.to_tensor([32, 24, 0], dtype="int32")

        num_blocks = 10
        block_tables = paddle.zeros([3, num_blocks], dtype="int32")
        for i in range(num_blocks):
            block_tables[0, i] = i
        for i in range(5):
            block_tables[1, i] = num_blocks + i

        meta = self._make_mock_forward_meta(seq_lens_this_time, prefix_lens, block_tables)
        qo_indptr, unified_kv_indptr, unified_kv_indices, plens, bs, max_extend = (
            backend._deterministic_build_triton_indices(meta)
        )

        assert bs == 2
        assert max_extend == 16
        # seq0: prefix=32 + extend=16 = 48 total KV
        # seq1: prefix=24 + extend=1 = 25 total KV
        assert unified_kv_indptr.tolist() == [0, 48, 73]


class TestDeterministicBuildTritonIndicesCrossValidation:
    """Cross-validate _deterministic_build_triton_indices vs _ref."""

    def _make_mock_backend(self, block_size):
        from fastdeploy.model_executor.layers.attention.append_attn_backend import (
            AppendAttentionBackend,
        )

        backend = object.__new__(AppendAttentionBackend)
        backend.block_size = block_size
        return backend

    def _make_mock_forward_meta(self, seq_lens_this_time, prefix_lens, block_tables):
        from types import SimpleNamespace

        meta = SimpleNamespace()
        meta.seq_lens_this_time = seq_lens_this_time
        meta.prefix_lens = prefix_lens
        meta.block_tables = block_tables

        # Pre-compute CPU scalars expected by the Triton path
        bs = int((seq_lens_this_time > 0).sum().item())
        meta.deter_bs = bs
        extend_lens = seq_lens_this_time[:bs]
        meta.deter_total_extend_len = int(paddle.sum(extend_lens).item())
        meta.deter_max_extend_len = int(paddle.max(extend_lens).item()) if bs > 0 else 0
        meta.deter_total_prefix_len = int(paddle.sum(prefix_lens[:bs]).item())
        return meta

    @pytest.mark.parametrize(
        "seq_lens,prefix",
        [
            ([8, 1, 0, 0], [0, 5, 0, 0]),
            ([1, 1, 1, 0], [10, 20, 5, 0]),
            ([16, 1, 0], [32, 24, 0]),
            ([4, 0], [0, 0]),
            ([1, 1, 1, 1], [3, 7, 15, 0]),
        ],
    )
    def test_triton_matches_ref(self, seq_lens, prefix):
        """Triton and ref implementations must produce identical indices."""
        block_size = 4
        backend = self._make_mock_backend(block_size)

        seq_lens_t = paddle.to_tensor(seq_lens, dtype="int32")
        prefix_t = paddle.to_tensor(prefix, dtype="int32")

        max_total = max(s + p for s, p in zip(seq_lens, prefix))
        max_blocks = (max_total + block_size - 1) // block_size
        max_blocks = max(max_blocks, 1)
        n_seqs = len(seq_lens)
        rng = np.random.RandomState(42)
        block_tables = paddle.to_tensor(rng.randint(0, 20, size=(n_seqs, max_blocks)).astype(np.int32))

        meta = self._make_mock_forward_meta(seq_lens_t, prefix_t, block_tables)

        qo_triton, kv_indptr_triton, kv_indices_triton, plens_triton, bs_triton, max_ext_triton = (
            backend._deterministic_build_triton_indices(meta)
        )
        qo_ref, kv_indptr_ref, kv_indices_ref, plens_ref, bs_ref, max_ext_ref = (
            backend._deterministic_build_triton_indices_ref(meta)
        )

        assert bs_triton == bs_ref, f"bs mismatch: {bs_triton} vs {bs_ref}"
        assert max_ext_triton == max_ext_ref, "max_extend mismatch"
        assert qo_triton.tolist() == qo_ref.tolist(), "qo_indptr mismatch"
        assert kv_indptr_triton.tolist() == kv_indptr_ref.tolist(), "kv_indptr mismatch"
        total_kv = int(kv_indptr_ref[-1].item())
        assert kv_indices_triton[:total_kv].tolist() == kv_indices_ref[:total_kv].tolist(), "kv_indices mismatch"
        assert plens_triton.tolist() == plens_ref.tolist(), "prefix_lens mismatch"


# ===========================================================================
# ACTION-7: extend_attention_fwd_unified parameter space expansion
# ===========================================================================


class TestUnifiedAttentionExtended:
    """Extended parameter space for extend_attention_fwd_unified."""

    def _run_attention(
        self, bs, q_len, num_q_heads, num_kv_heads, head_dim, block_size, prefix_len=0, is_causal=True, dtype="float16"
    ):
        """Helper to run kernel and compare against naive reference."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        kv_len = prefix_len + q_len
        paddle.seed(42)
        q_batched = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype(dtype)
        k_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype(dtype)
        v_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype(dtype)

        # Reference
        prefix_lens_ref = paddle.full([bs], prefix_len, dtype="int32")
        ref_out = naive_attention(q_batched, k_batched, v_batched, prefix_lens_ref, is_causal)

        # Build paged cache
        k_flat = k_batched.reshape([-1, num_kv_heads, head_dim])
        v_flat = v_batched.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        # Build indices
        total_q_tokens = bs * q_len
        q_flat = q_batched.reshape([total_q_tokens, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)

        # QO indptr
        q_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(q_lens).astype("int32"),
            ]
        )

        # KV indptr and indices
        kv_lens = paddle.full([bs], kv_len, dtype="int32")
        kv_indptr = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(kv_lens).astype("int32"),
            ]
        )
        total_kv = bs * kv_len
        kv_indices = paddle.arange(total_kv, dtype="int32")
        prefix_lens_t = paddle.full([bs], prefix_len, dtype="int32")

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
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        ref_fp32 = ref_out.astype("float32")
        triton_fp32 = triton_out.astype("float32")
        max_diff = float(paddle.max(paddle.abs(ref_fp32 - triton_fp32)).item())
        return max_diff

    # MQA (kv_heads=1)
    def test_mqa_kv1(self):
        """MQA: kv_heads=1, q_heads=32, kv_group_num=32."""
        max_diff = self._run_attention(bs=1, q_len=8, num_q_heads=32, num_kv_heads=1, head_dim=64, block_size=8)
        print(f"\n[MQA] max_diff={max_diff:.6e}")
        assert max_diff < MQA_ATOL, f"MQA failed: max_diff={max_diff}"

    # head_dim=256 (triggers BLOCK_M=64, BLOCK_N=64)
    def test_head_dim_256(self):
        """head_dim=256: triggers different block size selection."""
        max_diff = self._run_attention(bs=1, q_len=8, num_q_heads=4, num_kv_heads=4, head_dim=256, block_size=8)
        print(f"\n[head_dim=256] max_diff={max_diff:.6e}")
        assert max_diff < LARGE_HEAD_ATOL, f"head_dim=256 failed: max_diff={max_diff}"

    # bfloat16
    def test_bfloat16(self):
        """bfloat16 precision test."""
        max_diff = self._run_attention(
            bs=2, q_len=8, num_q_heads=4, num_kv_heads=4, head_dim=64, block_size=4, dtype="bfloat16"
        )
        print(f"\n[bfloat16] max_diff={max_diff:.6e}")
        assert max_diff < BF16_ATOL, f"bfloat16 failed: max_diff={max_diff}"

    # Long sequence 4096+
    def test_long_sequence(self):
        """Long sequence (4096) to detect numerical overflow."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        seq_len = 4096
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 64

        paddle.seed(42)
        q_flat = paddle.randn([seq_len, num_q_heads, head_dim]).astype("float16")
        k_flat = paddle.randn([seq_len, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([seq_len, num_kv_heads, head_dim]).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        qo_indptr = paddle.to_tensor([0, seq_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, seq_len], dtype="int32")
        kv_indices = paddle.arange(seq_len, dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")

        o = paddle.zeros_like(q_flat)
        o = extend_attention_fwd_unified(
            q_flat,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            True,
        )

        # Check no NaN or Inf
        assert not paddle.any(paddle.isnan(o)).item(), "Output contains NaN"
        assert not paddle.any(paddle.isinf(o)).item(), "Output contains Inf"
        # Output should sum to approximately non-zero (not degenerate)
        assert float(paddle.abs(o).mean().item()) > 1e-4, "Output is degenerate (near zero)"
        print(f"\n[long_seq] 4096 tokens passed, mean abs={float(paddle.abs(o).mean().item()):.4f}")

    # Large batch bs=8
    def test_large_batch_bs8(self):
        """Large batch (bs=8) multi-sequence parallel causal attention."""
        max_diff = self._run_attention(bs=8, q_len=16, num_q_heads=4, num_kv_heads=4, head_dim=64, block_size=16)
        print(f"\n[bs=8] max_diff={max_diff:.6e}")
        assert max_diff < MQA_ATOL, f"bs=8 failed: max_diff={max_diff}"

    # Non-contiguous block IDs
    def test_non_contiguous_blocks(self):
        """Non-contiguous block IDs (shuffled allocation)."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        q_len = 8
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4
        total_blocks = 20

        paddle.seed(42)
        q_flat = paddle.randn([q_len, num_q_heads, head_dim]).astype("float16")
        k_flat = paddle.randn([q_len, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([q_len, num_kv_heads, head_dim]).astype("float16")

        # Pack into non-contiguous blocks
        num_blocks_needed = (q_len + block_size - 1) // block_size
        np.random.seed(42)
        physical_blocks = np.random.choice(total_blocks, num_blocks_needed, replace=False)
        physical_blocks.sort()  # sort for stable test

        cache_k = paddle.zeros([total_blocks, num_kv_heads, block_size, head_dim], dtype="float16")
        cache_v = paddle.zeros([total_blocks, num_kv_heads, block_size, head_dim], dtype="float16")
        for t in range(q_len):
            logical_block = t // block_size
            phys_block = physical_blocks[logical_block]
            offset = t % block_size
            cache_k[phys_block, :, offset, :] = k_flat[t]
            cache_v[phys_block, :, offset, :] = v_flat[t]

        # Build indices using physical blocks
        kv_indices_list = []
        for t in range(q_len):
            phys_block = int(physical_blocks[t // block_size])
            kv_indices_list.append(phys_block * block_size + t % block_size)
        kv_indices = paddle.to_tensor(kv_indices_list, dtype="int32")

        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")

        o = paddle.zeros([q_len, num_q_heads, head_dim], dtype="float16")
        o = extend_attention_fwd_unified(
            q_flat,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )

        # Compare with sequential allocation reference
        cache_k_ref, cache_v_ref = _build_paged_kv_cache(k_flat, v_flat, block_size)
        kv_indices_ref = paddle.arange(q_len, dtype="int32")
        o_ref = paddle.zeros([q_len, num_q_heads, head_dim], dtype="float16")
        o_ref = extend_attention_fwd_unified(
            q_flat,
            o_ref,
            cache_k_ref,
            cache_v_ref,
            qo_indptr,
            kv_indptr,
            kv_indices_ref,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )

        max_diff = float(paddle.max(paddle.abs(o.astype("float32") - o_ref.astype("float32"))).item())
        print(f"\n[non_contiguous] max_diff={max_diff:.6e}")
        assert max_diff < 1e-5, f"Non-contiguous blocks differ from sequential: max_diff={max_diff}"

    # is_causal=False
    def test_non_causal(self):
        """is_causal=False: all tokens attend to all KV positions."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 1, 8
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")

        # Non-causal naive reference
        prefix_lens_ref = paddle.zeros([bs], dtype="int32")
        ref_out = naive_attention(q, k, v, prefix_lens_ref, is_causal=False)

        k_flat = k.reshape([-1, num_kv_heads, head_dim])
        v_flat = v.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        q_flat = q.reshape([q_len, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indices = paddle.arange(q_len, dtype="int32")
        prefix_lens_t = paddle.zeros([1], dtype="int32")

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
            False,  # is_causal=False
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        max_diff = float(paddle.max(paddle.abs(ref_out.astype("float32") - triton_out.astype("float32"))).item())
        print(f"\n[non_causal] max_diff={max_diff:.6e}")
        assert max_diff < FP16_ATOL * 2, f"Non-causal failed: max_diff={max_diff}"


# ===========================================================================
# Boundary edge case tests
# ===========================================================================


class TestBoundaryEdgeCases:
    """Boundary / edge-case scenarios not covered by the main test classes."""

    def test_minimal_input_bs1_extend1_no_prefix(self):
        """Smallest valid input: bs=1, extend_len=1, prefix=0."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(42)
        q = paddle.randn([1, 1, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([1, 1, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([1, 1, num_kv_heads, head_dim]).astype("float16")

        # With a single token, causal attention = softmax(q·k/sqrt(d)) * v = v (scaled)
        prefix_lens_ref = paddle.zeros([1], dtype="int32")
        ref_out = naive_attention(q, k, v, prefix_lens_ref, is_causal=True)

        k_flat = k.reshape([-1, num_kv_heads, head_dim])
        v_flat = v.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        q_flat = q.reshape([1, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr = paddle.to_tensor([0, 1], dtype="int32")
        kv_indptr = paddle.to_tensor([0, 1], dtype="int32")
        kv_indices = paddle.to_tensor([0], dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")

        o_flat = extend_attention_fwd_unified(
            q_flat,
            o_flat,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            1,
            True,
        )

        triton_out = o_flat.reshape([1, 1, num_q_heads, head_dim])
        max_diff = float(paddle.max(paddle.abs(ref_out.astype("float32") - triton_out.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Minimal input failed: max_diff={max_diff}"

    def test_q_len_not_aligned_to_block_size(self):
        """q_len=7, block_size=4: not a multiple of block_size."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 1, 7
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")

        prefix_lens_ref = paddle.zeros([bs], dtype="int32")
        ref_out = naive_attention(q, k, v, prefix_lens_ref, is_causal=True)

        k_flat = k.reshape([-1, num_kv_heads, head_dim])
        v_flat = v.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        q_flat = q.reshape([q_len, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indices = paddle.arange(q_len, dtype="int32")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        o_flat = extend_attention_fwd_unified(
            q_flat,
            o_flat,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        max_diff = float(paddle.max(paddle.abs(ref_out.astype("float32") - triton_out.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Non-aligned q_len failed: max_diff={max_diff}"

    def test_prefix_not_aligned_to_block_size(self):
        """prefix_len=5 (not aligned to block_size=4) split invariance."""
        total_len = 32
        prefix_a = 0
        prefix_b = 5  # not aligned to block_size
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(42)
        q_all = paddle.randn([total_len, num_q_heads, head_dim]).astype("float16")
        k_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")
        v_all = paddle.randn([total_len, num_kv_heads, head_dim]).astype("float16")

        # Reuse TestSplitInvariance._run_with_split
        split_test = TestSplitInvariance()
        out_a = split_test._run_with_split(
            q_all, k_all, v_all, total_len, prefix_a, block_size, num_q_heads, num_kv_heads, head_dim
        )
        # out_a covers all 32 tokens; take last (32-5)=27 tokens to compare
        extend_b = total_len - prefix_b
        out_a_tail = out_a[-extend_b:]

        out_b = split_test._run_with_split(
            q_all, k_all, v_all, total_len, prefix_b, block_size, num_q_heads, num_kv_heads, head_dim
        )

        max_diff = float(paddle.max(paddle.abs(out_a_tail.astype("float32") - out_b.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Non-aligned prefix split invariance failed: max_diff={max_diff}"

    def test_explicit_sm_scale(self):
        """Pass explicit sm_scale instead of default 1/sqrt(head_dim)."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 1, 8
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4
        custom_sm_scale = 0.1  # intentionally different from 1/sqrt(64)=0.125

        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")

        # Naive reference with custom scale
        scale = custom_sm_scale
        k_exp = k  # MHA: no GQA expansion needed
        scores = paddle.einsum("bqhd,bkhd->bhqk", q, k_exp) * scale
        # causal mask (no prefix)
        for qi in range(q_len):
            for ki in range(q_len):
                if qi < ki:
                    scores[:, :, qi, ki] = float("-inf")
        attn = paddle.nn.functional.softmax(scores, axis=-1)
        ref_out = paddle.einsum("bhqk,bkhd->bqhd", attn, v)

        k_flat = k.reshape([-1, num_kv_heads, head_dim])
        v_flat = v.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        q_flat = q.reshape([q_len, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)
        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indices = paddle.arange(q_len, dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")

        o_flat = extend_attention_fwd_unified(
            q_flat,
            o_flat,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
            sm_scale=custom_sm_scale,
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        max_diff = float(paddle.max(paddle.abs(ref_out.astype("float32") - triton_out.astype("float32"))).item())
        assert max_diff < FP16_ATOL, f"Custom sm_scale failed: max_diff={max_diff}"


# ===========================================================================
# Strengthened determinism tests
# ===========================================================================


class TestDeterminismStrengthened:
    """Extended determinism verification (more runs, with prefix)."""

    def test_determinism_10_runs_no_prefix(self):
        """Run kernel 10 times, verify bitwise identical."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 2, 16
        num_q_heads, num_kv_heads, head_dim = 8, 4, 128
        block_size = 16

        paddle.seed(999)
        q_flat = paddle.randn([bs * q_len, num_q_heads, head_dim]).astype("float16")
        k_flat = paddle.randn([bs * q_len, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([bs * q_len, num_kv_heads, head_dim]).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        seq_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(bs * q_len, dtype="int32")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        results = []
        for _ in range(10):
            o = paddle.zeros_like(q_flat)
            o = extend_attention_fwd_unified(
                q_flat,
                o,
                cache_k,
                cache_v,
                qo_indptr,
                kv_indptr,
                kv_indices,
                prefix_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q_len,
                True,
            )
            results.append(o.astype("float32").numpy())

        for i in range(1, len(results)):
            assert (results[0] == results[i]).all(), f"Run 0 vs run {i} differ (no prefix)"

    def test_determinism_with_prefix(self):
        """Run kernel 10 times WITH prefix, verify bitwise identical."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        # bs = 1
        prefix_len = 32
        extend_len = 8
        kv_len = prefix_len + extend_len
        num_q_heads, num_kv_heads, head_dim = 8, 4, 64
        block_size = 16

        paddle.seed(42)
        q_flat = paddle.randn([extend_len, num_q_heads, head_dim]).astype("float16")
        k_flat = paddle.randn([kv_len, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([kv_len, num_kv_heads, head_dim]).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        qo_indptr = paddle.to_tensor([0, extend_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, kv_len], dtype="int32")
        kv_indices = paddle.arange(kv_len, dtype="int32")
        prefix_lens = paddle.to_tensor([prefix_len], dtype="int32")

        results = []
        for _ in range(10):
            o = paddle.zeros([extend_len, num_q_heads, head_dim], dtype="float16")
            o = extend_attention_fwd_unified(
                q_flat,
                o,
                cache_k,
                cache_v,
                qo_indptr,
                kv_indptr,
                kv_indices,
                prefix_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                extend_len,
                True,
            )
            results.append(o.astype("float32").numpy())

        for i in range(1, len(results)):
            assert (results[0] == results[i]).all(), f"Run 0 vs run {i} differ (with prefix)"

    def test_determinism_gqa_large_batch(self):
        """Determinism with GQA and bs=4."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 4, 16
        num_q_heads, num_kv_heads, head_dim = 16, 2, 128
        block_size = 16

        paddle.seed(314)
        total_q = bs * q_len
        q_flat = paddle.randn([total_q, num_q_heads, head_dim]).astype("float16")
        k_flat = paddle.randn([total_q, num_kv_heads, head_dim]).astype("float16")
        v_flat = paddle.randn([total_q, num_kv_heads, head_dim]).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        seq_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(total_q, dtype="int32")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        results = []
        for _ in range(10):
            o = paddle.zeros_like(q_flat)
            o = extend_attention_fwd_unified(
                q_flat,
                o,
                cache_k,
                cache_v,
                qo_indptr,
                kv_indptr,
                kv_indices,
                prefix_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q_len,
                True,
            )
            results.append(o.astype("float32").numpy())

        for i in range(1, len(results)):
            assert (results[0] == results[i]).all(), f"Run 0 vs run {i} differ (GQA large batch)"


# ===========================================================================
# Error / robustness tests
# ===========================================================================


class TestErrorHandling:
    """Tests for invalid input handling and robustness."""

    def test_build_kv_indices_bs_zero(self):
        """bs=0 should produce an empty result without crashing."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_kv_indices_from_block_tables,
        )

        block_tables = paddle.zeros([0, 4], dtype="int32")
        seq_lens = paddle.zeros([0], dtype="int32")
        kv_indptr, kv_indices = build_kv_indices_from_block_tables(block_tables, seq_lens, 4, bs=0)
        assert kv_indptr.tolist() == [0]

    def test_build_unified_all_zero_extend(self):
        """All extend_lens=0 (edge case that should not crash)."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            build_unified_kv_indices,
        )

        bs = 2
        prefix_kv_indptr = paddle.to_tensor([0, 3, 5], dtype="int32")
        prefix_kv_indices = paddle.to_tensor([10, 11, 12, 20, 21], dtype="int32")
        extend_seq_lens = paddle.zeros([bs], dtype="int32")
        extend_start_loc = paddle.zeros([bs], dtype="int32")
        extend_kv_indices = paddle.zeros([0], dtype="int32")

        unified_indptr, unified_indices, plens = build_unified_kv_indices(
            prefix_kv_indptr,
            prefix_kv_indices,
            extend_start_loc,
            extend_seq_lens,
            extend_kv_indices,
            bs,
        )

        assert plens.tolist() == [3, 2]
        assert unified_indptr.tolist() == [0, 3, 5]
        assert unified_indices[:5].tolist() == [10, 11, 12, 20, 21]

    def test_output_no_nan_inf_with_large_values(self):
        """Input near fp16 range limit should not produce NaN/Inf."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        q_len = 8
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 4

        paddle.seed(42)
        # Use values near fp16 range but not overflowing
        q_flat = (paddle.randn([q_len, num_q_heads, head_dim]) * 10.0).astype("float16")
        k_flat = (paddle.randn([q_len, num_kv_heads, head_dim]) * 10.0).astype("float16")
        v_flat = (paddle.randn([q_len, num_kv_heads, head_dim]) * 10.0).astype("float16")
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        qo_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indptr = paddle.to_tensor([0, q_len], dtype="int32")
        kv_indices = paddle.arange(q_len, dtype="int32")
        prefix_lens = paddle.zeros([1], dtype="int32")

        o = paddle.zeros([q_len, num_q_heads, head_dim], dtype="float16")
        o = extend_attention_fwd_unified(
            q_flat,
            o,
            cache_k,
            cache_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            prefix_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_len,
            True,
        )

        assert not paddle.any(paddle.isnan(o)).item(), "Output contains NaN with large values"
        assert not paddle.any(paddle.isinf(o)).item(), "Output contains Inf with large values"


# ===========================================================================
# Production-scale correctness tests (SGLang parity: bs=19, N_CTX=12331)
# ===========================================================================


class TestProductionScaleCorrectness:
    """
    Large-scale correctness tests matching SGLang's production-level data sizes.
    SGLang uses B=19, N_CTX=12331 — we test comparable scales.
    """

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
        """Run attention at production scale with variable-length sequences."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        if prefix_lens_list is None:
            prefix_lens_list = [0] * bs

        extend_lens = [s - p for s, p in zip(seq_lens, prefix_lens_list)]
        paddle.seed(42)

        all_ref_outputs = []
        all_q_flat = []
        all_k_flat = []
        all_v_flat = []

        for b in range(bs):
            q_len_b = extend_lens[b]
            kv_len_b = seq_lens[b]
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

        qo_indptr_list = [0]
        kv_indptr_list = [0]
        kv_indices_list = []
        prefix_lens_t_list = []

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
        max_extend = max(extend_lens)

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
            max_extend,
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
    def test_large_batch_bs19(self):
        """SGLang-scale: bs=19 with variable-length sequences."""
        rng = np.random.RandomState(42)
        bs = 19
        seq_lens = rng.randint(64, 512, size=bs).tolist()
        num_q_heads, num_kv_heads, head_dim = 12, 4, 128
        block_size = 64

        max_diff, cos_sim = self._run_large_scale_attention(
            bs, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size
        )
        print(f"\n[bs=19] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < LARGE_SCALE_ATOL, f"Large batch failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine similarity: {cos_sim}"

    @pytest.mark.gpu
    def test_long_sequence_4096_correctness(self):
        """Long single sequence with correctness validation (not just NaN check)."""
        bs = 1
        seq_lens = [4096]
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        block_size = 64

        max_diff, cos_sim = self._run_large_scale_attention(
            bs, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size
        )
        print(f"\n[seq=4096] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < LARGE_SCALE_ATOL, f"Long seq failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine similarity: {cos_sim}"

    @pytest.mark.gpu
    def test_mixed_length_sequences(self):
        """Mixed short and long sequences in same batch."""
        bs = 8
        seq_lens = [32, 256, 64, 1024, 16, 512, 128, 2048]
        num_q_heads, num_kv_heads, head_dim = 8, 2, 128
        block_size = 64

        max_diff, cos_sim = self._run_large_scale_attention(
            bs, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size
        )
        print(f"\n[mixed_len] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < LARGE_SCALE_ATOL, f"Mixed length failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine similarity: {cos_sim}"

    @pytest.mark.gpu
    def test_large_batch_with_prefix(self):
        """Large batch with variable prefix lengths."""
        rng = np.random.RandomState(123)
        bs = 12
        prefix_lens = rng.randint(0, 256, size=bs).tolist()
        extend_lens = rng.randint(1, 128, size=bs).tolist()
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        num_q_heads, num_kv_heads, head_dim = 8, 4, 128
        block_size = 64

        max_diff, cos_sim = self._run_large_scale_attention(
            bs,
            seq_lens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            prefix_lens_list=prefix_lens,
        )
        print(f"\n[bs=12,prefix] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < LARGE_SCALE_ATOL, f"Large batch w/ prefix failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine similarity: {cos_sim}"


# ===========================================================================
# Non-standard head_dim tests (96, 80, 13 — non-power-of-2)
# ===========================================================================


class TestNonStandardHeadDim:
    """
    Test non-standard head dimensions that are NOT powers of 2.
    SGLang tests head_dim=96, 80, 13 — we need the same coverage.
    """

    def _run_attention_with_head_dim(
        self,
        head_dim,
        bs=2,
        q_len=16,
        num_q_heads=4,
        num_kv_heads=4,
        block_size=16,
        prefix_len=0,
    ):
        """Run kernel with specific head_dim and validate against naive reference."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        kv_len = prefix_len + q_len
        paddle.seed(42)
        q_batched = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")
        v_batched = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")

        prefix_lens_ref = paddle.full([bs], prefix_len, dtype="int32")
        ref_out = naive_attention(q_batched, k_batched, v_batched, prefix_lens_ref, is_causal=True)

        k_flat = k_batched.reshape([-1, num_kv_heads, head_dim])
        v_flat = v_batched.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        total_q_tokens = bs * q_len
        q_flat = q_batched.reshape([total_q_tokens, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)

        q_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(q_lens).astype("int32")])
        kv_lens = paddle.full([bs], kv_len, dtype="int32")
        kv_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(kv_lens).astype("int32")])
        kv_indices = paddle.arange(bs * kv_len, dtype="int32")
        prefix_lens_t = paddle.full([bs], prefix_len, dtype="int32")

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
            True,
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        ref_fp32 = ref_out.astype("float32")
        triton_fp32 = triton_out.astype("float32")
        max_diff = float(paddle.max(paddle.abs(ref_fp32 - triton_fp32)).item())
        cos_sim = cosine_similarity(ref_fp32, triton_fp32)

        assert not paddle.any(paddle.isnan(o_flat)).item(), f"NaN at head_dim={head_dim}"
        assert not paddle.any(paddle.isinf(o_flat)).item(), f"Inf at head_dim={head_dim}"
        return max_diff, cos_sim

    @pytest.mark.gpu
    def test_head_dim_96(self):
        """head_dim=96: common in GPT-NeoX variants."""
        max_diff, cos_sim = self._run_attention_with_head_dim(96)
        print(f"\n[head_dim=96] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < NONSTANDARD_HEAD_ATOL, f"head_dim=96 failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"

    @pytest.mark.gpu
    def test_head_dim_80(self):
        """head_dim=80: used in StarCoder and some code models."""
        max_diff, cos_sim = self._run_attention_with_head_dim(80)
        print(f"\n[head_dim=80] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < NONSTANDARD_HEAD_ATOL, f"head_dim=80 failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"

    @pytest.mark.gpu
    def test_head_dim_13_prime(self):
        """head_dim=13: prime number, stress tests non-aligned memory access."""
        max_diff, cos_sim = self._run_attention_with_head_dim(
            13,
            bs=1,
            q_len=8,
            num_q_heads=4,
            num_kv_heads=4,
            block_size=4,
        )
        print(f"\n[head_dim=13] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < PRIME_HEAD_ATOL, f"head_dim=13 failed: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"

    @pytest.mark.gpu
    def test_head_dim_96_with_gqa(self):
        """head_dim=96 + GQA (q_heads=12, kv_heads=4)."""
        max_diff, cos_sim = self._run_attention_with_head_dim(
            96,
            bs=2,
            q_len=16,
            num_q_heads=12,
            num_kv_heads=4,
            block_size=16,
        )
        print(f"\n[head_dim=96,GQA] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < NONSTANDARD_HEAD_ATOL, f"head_dim=96 GQA failed: max_diff={max_diff}"

    @pytest.mark.gpu
    def test_head_dim_80_with_prefix(self):
        """head_dim=80 with prefix tokens."""
        max_diff, cos_sim = self._run_attention_with_head_dim(
            80,
            bs=1,
            q_len=8,
            num_q_heads=4,
            num_kv_heads=4,
            block_size=8,
            prefix_len=16,
        )
        print(f"\n[head_dim=80,prefix] max_diff={max_diff:.6e}, cosine_sim={cos_sim:.10f}")
        assert max_diff < NONSTANDARD_HEAD_ATOL, f"head_dim=80 prefix failed: max_diff={max_diff}"


# ===========================================================================
# Cross-validation: naive_attention vs sdpa_attention_reference
# ===========================================================================


class TestReferenceImplCrossValidation:
    """
    Cross-validate naive_attention against sdpa_attention_reference.
    If either reference has a bug, these tests will catch it.
    """

    @pytest.mark.parametrize("num_kv_heads,num_q_heads", [(4, 4), (2, 8), (1, 8)])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_references_match_no_prefix(self, num_kv_heads, num_q_heads, head_dim):
        """Both references should agree on no-prefix causal attention."""
        bs, q_len = 2, 16
        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=True)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True)

        naive_fp32 = out_naive.astype("float32")
        sdpa_fp32 = out_sdpa.astype("float32")
        max_diff = float(paddle.max(paddle.abs(naive_fp32 - sdpa_fp32)).item())
        cos_sim = cosine_similarity(naive_fp32, sdpa_fp32)
        print(
            f"\n[ref_xval] heads={num_q_heads}/{num_kv_heads}, dim={head_dim}: "
            f"max_diff={max_diff:.6e}, cos_sim={cos_sim:.10f}"
        )
        assert max_diff < FP16_ATOL, f"Reference mismatch: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"

    @pytest.mark.parametrize("prefix_len", [4, 16, 32])
    def test_references_match_with_prefix(self, prefix_len):
        """Both references should agree on prefix + extend causal attention."""
        bs, q_len = 1, 8
        kv_len = prefix_len + q_len
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        paddle.seed(123)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, kv_len, num_kv_heads, head_dim]).astype("float16")
        prefix_lens = paddle.to_tensor([prefix_len], dtype="int32")

        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=True)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=True)

        naive_fp32 = out_naive.astype("float32")
        sdpa_fp32 = out_sdpa.astype("float32")
        max_diff = float(paddle.max(paddle.abs(naive_fp32 - sdpa_fp32)).item())
        assert max_diff < FP16_ATOL, f"Reference mismatch w/ prefix={prefix_len}: max_diff={max_diff}"

    def test_references_match_non_causal(self):
        """Both references should agree on non-causal attention."""
        bs, q_len = 2, 8
        num_q_heads, num_kv_heads, head_dim = 4, 4, 64
        paddle.seed(42)
        q = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        prefix_lens = paddle.zeros([bs], dtype="int32")

        out_naive = naive_attention(q, k, v, prefix_lens, is_causal=False)
        out_sdpa = sdpa_attention_reference(q, k, v, prefix_lens, is_causal=False)

        naive_fp32 = out_naive.astype("float32")
        sdpa_fp32 = out_sdpa.astype("float32")
        max_diff = float(paddle.max(paddle.abs(naive_fp32 - sdpa_fp32)).item())
        assert max_diff < FP16_ATOL, f"Non-causal ref mismatch: max_diff={max_diff}"


# ===========================================================================
# Triton kernel vs SDPA reference (triple validation)
# ===========================================================================


class TestTritonVsSDPACrossValidation:
    """
    Triple validation: Triton kernel output vs SDPA reference.
    This catches bugs that both naive_attention and the kernel share.
    """

    @pytest.mark.gpu
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_triton_matches_sdpa(self, head_dim):
        """Triton kernel output should match SDPA reference."""
        from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
            extend_attention_fwd_unified,
        )

        bs, q_len = 2, 16
        num_q_heads, num_kv_heads = 8, 4
        block_size = 16

        paddle.seed(42)
        q_batched = paddle.randn([bs, q_len, num_q_heads, head_dim]).astype("float16")
        k_batched = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")
        v_batched = paddle.randn([bs, q_len, num_kv_heads, head_dim]).astype("float16")

        prefix_lens_ref = paddle.zeros([bs], dtype="int32")
        sdpa_out = sdpa_attention_reference(
            q_batched,
            k_batched,
            v_batched,
            prefix_lens_ref,
            is_causal=True,
        )

        k_flat = k_batched.reshape([-1, num_kv_heads, head_dim])
        v_flat = v_batched.reshape([-1, num_kv_heads, head_dim])
        cache_k, cache_v = _build_paged_kv_cache(k_flat, v_flat, block_size)

        total_tokens = bs * q_len
        q_flat = q_batched.reshape([total_tokens, num_q_heads, head_dim])
        o_flat = paddle.zeros_like(q_flat)

        seq_lens = paddle.full([bs], q_len, dtype="int32")
        qo_indptr = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens).astype("int32")])
        kv_indptr = qo_indptr.clone()
        kv_indices = paddle.arange(total_tokens, dtype="int32")
        prefix_lens_t = paddle.zeros([bs], dtype="int32")

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
            True,
        )

        triton_out = o_flat.reshape([bs, q_len, num_q_heads, head_dim])
        sdpa_fp32 = sdpa_out.astype("float32")
        triton_fp32 = triton_out.astype("float32")
        max_diff = float(paddle.max(paddle.abs(sdpa_fp32 - triton_fp32)).item())
        cos_sim = cosine_similarity(sdpa_fp32, triton_fp32)
        print(f"\n[triton_vs_sdpa] head_dim={head_dim}: " f"max_diff={max_diff:.6e}, cos_sim={cos_sim:.10f}")
        assert max_diff < FP16_ATOL, f"Triton vs SDPA mismatch: max_diff={max_diff}"
        assert cos_sim > COSINE_SIM_THRESHOLD, f"Low cosine sim: {cos_sim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
