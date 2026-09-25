"""
Triton decode attention kernel tests — correctness, determinism, and edge cases.

Tests for compute_num_kv_splits and decode_attention_fwd from
fastdeploy/model_executor/layers/attention/triton_ops/decode_attention.py

Correctness verification strategy:
    A naive Python reference implementation (using float32 matmul + softmax) is
    compared against the Triton kernel output using max absolute diff and cosine
    similarity thresholds. Broad parametrized coverage spans head configurations
    (MHA/GQA/MQA), data types (float16/bfloat16), sequence lengths, and edge cases.

Test scenarios:
1. compute_num_kv_splits: basic, edge cases, max capping
2. decode_attention_fwd correctness: MHA/GQA/MQA, various head_dim, float16/bfloat16
3. Determinism: multiple runs produce identical results
4. Edge cases: single token seq, large seq, multiple batches

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_triton_decode_attention.py -v
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.decode_attention import (
    compute_num_kv_splits,
    decode_attention_fwd,
)

# ---------------------------------------------------------------------------
# Skip if no CUDA
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0,
    reason="Triton decode attention requires CUDA",
)

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
FP16_ATOL = 2e-2
BF16_ATOL = 5e-2
COSINE_SIM_THRESHOLD = 1 - 1e-3


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
# Reference implementation: naive decode attention (no paging)
# ---------------------------------------------------------------------------
def naive_decode_attention_ref(q, k_pages, v_pages, kv_indptr, kv_indices, sm_scale, kv_block_size):
    """
    Naive Python reference for decode attention with paged KV cache.

    Args:
        q: [batch, num_heads, Lk]
        k_pages: [num_blocks, kv_heads, block_size, Lk]
        v_pages: [num_blocks, kv_heads, block_size, Lv]
        kv_indptr: [batch+1] CSR pointers
        kv_indices: [total_kv_len] flat indices (block_id * block_size + offset)
        sm_scale: float
        kv_block_size: int

    Returns:
        o: [batch, num_heads, Lv]
    """
    batch = q.shape[0]
    num_heads = q.shape[1]
    Lk = q.shape[2]
    kv_heads = k_pages.shape[1]
    Lv = v_pages.shape[-1]
    group_size = num_heads // kv_heads

    q_np = q.astype("float32").numpy()
    k_np = k_pages.astype("float32").numpy()
    v_np = v_pages.astype("float32").numpy()
    indptr_np = kv_indptr.numpy()
    indices_np = kv_indices.numpy()

    o_np = np.zeros([batch, num_heads, Lv], dtype=np.float32)

    for b in range(batch):
        start = indptr_np[b]
        end = indptr_np[b + 1]
        seq_len = end - start
        if seq_len == 0:
            continue

        # Gather K and V for this batch from paged cache
        k_gathered = np.zeros([kv_heads, seq_len, Lk], dtype=np.float32)
        v_gathered = np.zeros([kv_heads, seq_len, Lv], dtype=np.float32)

        for t in range(seq_len):
            flat_idx = indices_np[start + t]
            block_id = flat_idx // kv_block_size
            offset = flat_idx % kv_block_size
            k_gathered[:, t, :] = k_np[block_id, :, offset, :]
            v_gathered[:, t, :] = v_np[block_id, :, offset, :]

        # Expand KV for GQA
        for h in range(num_heads):
            kv_h = h // group_size
            # q_h: [Lk], k_h: [seq_len, Lk], v_h: [seq_len, Lv]
            q_h = q_np[b, h, :]
            k_h = k_gathered[kv_h]  # [seq_len, Lk]
            v_h = v_gathered[kv_h]  # [seq_len, Lv]

            # Attention scores
            scores = (q_h @ k_h.T) * sm_scale  # [seq_len]
            scores -= np.max(scores)
            attn_weights = np.exp(scores)
            attn_weights /= np.sum(attn_weights) + 1e-12

            o_np[b, h, :] = attn_weights @ v_h

    return paddle.to_tensor(o_np)


# ---------------------------------------------------------------------------
# Helper: build paged KV cache test data
# ---------------------------------------------------------------------------
def build_decode_test_data(
    batch_size,
    num_heads,
    kv_heads,
    head_dim_k,
    head_dim_v,
    seq_lens,
    block_size=16,
    dtype="float16",
    seed=42,
):
    """
    Build test data for decode attention.

    Returns dict with all tensors needed for decode_attention_fwd and the reference.
    """
    np.random.seed(seed)
    paddle.seed(seed)

    num_blocks_needed = sum((s + block_size - 1) // block_size for s in seq_lens)
    num_blocks = max(num_blocks_needed + 4, 8)

    # Paged K/V cache
    k_pages = paddle.randn([num_blocks, kv_heads, block_size, head_dim_k]).astype(dtype)
    v_pages = paddle.randn([num_blocks, kv_heads, block_size, head_dim_v]).astype(dtype)

    # Allocate blocks sequentially for simplicity
    block_cursor = 0
    kv_indptr_list = [0]
    kv_indices_list = []

    for b in range(batch_size):
        sl = seq_lens[b]
        for t in range(sl):
            blk_idx_in_seq = t // block_size
            offset = t % block_size
            actual_block_id = block_cursor + blk_idx_in_seq
            kv_indices_list.append(actual_block_id * block_size + offset)
        block_cursor += (sl + block_size - 1) // block_size
        kv_indptr_list.append(kv_indptr_list[-1] + sl)

    kv_indptr = paddle.to_tensor(kv_indptr_list, dtype="int32")
    kv_indices = paddle.to_tensor(kv_indices_list, dtype="int32")

    # Query
    q = paddle.randn([batch_size, num_heads, head_dim_k]).astype(dtype)

    # Compute num_kv_splits
    seq_lens_tensor = paddle.to_tensor(seq_lens, dtype="int32")
    max_kv_splits = 32
    num_kv_splits = compute_num_kv_splits(seq_lens_tensor, batch_size, max_kv_splits)

    # Pre-allocate intermediate buffers
    Lv = head_dim_v
    attn_logits = paddle.empty([batch_size, num_heads, max_kv_splits, Lv], dtype="float32")
    attn_lse = paddle.empty([batch_size, num_heads, max_kv_splits], dtype="float32")
    o = paddle.empty([batch_size, num_heads, Lv], dtype=dtype)

    sm_scale = head_dim_k**-0.5

    return {
        "q": q,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "o": o,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "attn_logits": attn_logits,
        "attn_lse": attn_lse,
        "num_kv_splits": num_kv_splits,
        "max_kv_splits": max_kv_splits,
        "sm_scale": sm_scale,
        "block_size": block_size,
        "seq_lens": seq_lens,
    }


# ===========================================================================
# Tests for compute_num_kv_splits
# ===========================================================================
class TestComputeNumKvSplits:
    """Tests for the compute_num_kv_splits utility."""

    def test_basic(self):
        """Short sequences should get 1 split."""
        seq_lens = paddle.to_tensor([100, 200, 50], dtype="int32")
        splits = compute_num_kv_splits(seq_lens, 3, max_kv_splits=32)
        splits_np = splits[:3].numpy()
        # (100+255)//256 = 1, (200+255)//256 = 1, (50+255)//256 = 1
        np.testing.assert_array_equal(splits_np, [1, 1, 1])

    def test_long_sequences(self):
        """Longer sequences should get more splits."""
        seq_lens = paddle.to_tensor([512, 1024, 2048], dtype="int32")
        splits = compute_num_kv_splits(seq_lens, 3, max_kv_splits=32)
        splits_np = splits[:3].numpy()
        expected = [min((s + 255) // 256, 32) for s in [512, 1024, 2048]]
        np.testing.assert_array_equal(splits_np, expected)

    def test_max_capping(self):
        """Splits should be capped at max_kv_splits."""
        seq_lens = paddle.to_tensor([100000], dtype="int32")
        splits = compute_num_kv_splits(seq_lens, 1, max_kv_splits=16)
        assert splits[0].item() == 16

    def test_single_token(self):
        """Single-token sequence should get 1 split."""
        seq_lens = paddle.to_tensor([1], dtype="int32")
        splits = compute_num_kv_splits(seq_lens, 1, max_kv_splits=32)
        assert splits[0].item() == 1

    def test_out_buf(self):
        """Pre-allocated output buffer should be respected."""
        seq_lens = paddle.to_tensor([512, 1024], dtype="int32")
        out_buf = paddle.zeros([2], dtype="int32")
        result = compute_num_kv_splits(seq_lens, 2, max_kv_splits=32, out_buf=out_buf)
        # Result should be the same object (same data pointer)
        assert result.data_ptr() == out_buf.data_ptr()
        assert result[0].item() == 2  # (512+255)//256 = 2
        assert result[1].item() == 4  # (1024+255)//256 = 4

    def test_empty(self):
        """num_seq=0 should return without error."""
        seq_lens = paddle.empty([0], dtype="int32")
        result = compute_num_kv_splits(seq_lens, 0, max_kv_splits=32)
        assert result.shape[0] == 0


# ===========================================================================
# Tests for decode_attention_fwd
# ===========================================================================

# MLA typical configs: Lk = kv_lora_rank + qk_rope_head_dim (e.g. 512+64=576)
_DECODE_CASES = [
    # (name, batch, num_heads, kv_heads, Lk, Lv, seq_lens, block_size)
    ("mla_basic_bs1", 1, 16, 1, 576, 512, [64], 16),
    ("mla_basic_bs4", 4, 16, 1, 576, 512, [32, 64, 128, 48], 16),
    ("mla_long_seq", 1, 16, 1, 576, 512, [1024], 16),
    ("mla_short_seq", 2, 16, 1, 576, 512, [1, 3], 16),
    ("mla_bs8_mixed", 8, 8, 1, 576, 512, [16, 32, 64, 128, 256, 48, 96, 512], 16),
    ("gqa_basic", 2, 16, 4, 128, 128, [64, 128], 16),
    ("mha_basic", 2, 8, 8, 128, 128, [32, 64], 16),
    ("mla_288", 2, 16, 1, 288, 256, [64, 128], 16),
    ("mla_block32", 2, 16, 1, 576, 512, [64, 128], 32),
]


@pytest.mark.parametrize(
    "name,batch,num_heads,kv_heads,Lk,Lv,seq_lens,block_size",
    _DECODE_CASES,
    ids=[c[0] for c in _DECODE_CASES],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_decode_attention_correctness(name, batch, num_heads, kv_heads, Lk, Lv, seq_lens, block_size, dtype):
    """Triton decode attention output should match naive reference."""
    data = build_decode_test_data(
        batch_size=batch,
        num_heads=num_heads,
        kv_heads=kv_heads,
        head_dim_k=Lk,
        head_dim_v=Lv,
        seq_lens=seq_lens,
        block_size=block_size,
        dtype=dtype,
    )

    # Run triton kernel
    decode_attention_fwd(
        data["q"],
        data["k_pages"],
        data["v_pages"],
        data["o"],
        data["kv_indptr"],
        data["kv_indices"],
        data["attn_logits"],
        data["attn_lse"],
        data["num_kv_splits"],
        data["max_kv_splits"],
        data["sm_scale"],
        data["block_size"],
    )
    triton_out = data["o"].astype("float32")

    # Run reference
    ref_out = naive_decode_attention_ref(
        data["q"],
        data["k_pages"],
        data["v_pages"],
        data["kv_indptr"],
        data["kv_indices"],
        data["sm_scale"],
        data["block_size"],
    )

    max_diff = float(paddle.max(paddle.abs(triton_out - ref_out)).item())
    cos_sim = cosine_similarity(triton_out, ref_out)

    atol = BF16_ATOL if dtype == "bfloat16" else FP16_ATOL
    assert max_diff < atol, f"[{name}/{dtype}] max_diff={max_diff:.6f} exceeds atol={atol}"
    assert (
        cos_sim > COSINE_SIM_THRESHOLD
    ), f"[{name}/{dtype}] cos_sim={cos_sim:.6f} below threshold={COSINE_SIM_THRESHOLD}"


# ===========================================================================
# Determinism test
# ===========================================================================
def test_decode_attention_determinism():
    """Multiple runs should produce bitwise identical results."""
    data = build_decode_test_data(
        batch_size=4,
        num_heads=16,
        kv_heads=1,
        head_dim_k=576,
        head_dim_v=512,
        seq_lens=[64, 128, 32, 256],
        block_size=16,
        dtype="float16",
    )

    results = []
    for _ in range(5):
        o = paddle.empty_like(data["o"])
        decode_attention_fwd(
            data["q"],
            data["k_pages"],
            data["v_pages"],
            o,
            data["kv_indptr"],
            data["kv_indices"],
            data["attn_logits"],
            data["attn_lse"],
            data["num_kv_splits"],
            data["max_kv_splits"],
            data["sm_scale"],
            data["block_size"],
        )
        results.append(o.astype("float32").numpy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i], err_msg=f"Run 0 vs run {i} differ — non-deterministic!")


# ===========================================================================
# Edge case: all sequences same length
# ===========================================================================
def test_decode_attention_uniform_seqlens():
    """Uniform sequence lengths should produce correct results."""
    batch = 4
    seq_len = 128
    data = build_decode_test_data(
        batch_size=batch,
        num_heads=16,
        kv_heads=1,
        head_dim_k=576,
        head_dim_v=512,
        seq_lens=[seq_len] * batch,
        block_size=16,
        dtype="float16",
    )

    decode_attention_fwd(
        data["q"],
        data["k_pages"],
        data["v_pages"],
        data["o"],
        data["kv_indptr"],
        data["kv_indices"],
        data["attn_logits"],
        data["attn_lse"],
        data["num_kv_splits"],
        data["max_kv_splits"],
        data["sm_scale"],
        data["block_size"],
    )
    triton_out = data["o"].astype("float32")

    ref_out = naive_decode_attention_ref(
        data["q"],
        data["k_pages"],
        data["v_pages"],
        data["kv_indptr"],
        data["kv_indices"],
        data["sm_scale"],
        data["block_size"],
    )

    max_diff = float(paddle.max(paddle.abs(triton_out - ref_out)).item())
    assert max_diff < FP16_ATOL, f"max_diff={max_diff:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
