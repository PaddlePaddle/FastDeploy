"""
Test script for V100 Triton attention kernels.

Tests numerical correctness by comparing Triton kernel outputs against
Python reference implementations. Also provides basic performance benchmarks.

Usage:
    # Run on a V100 GPU:
    python tests/model_executor/ops/triton_ops/test_v100_attn_kernels.py

    # Run specific test:
    python -m pytest tests/model_executor/ops/triton_ops/test_v100_attn_kernels.py::TestComputePositions -v

    # Run with benchmark timing:
    python tests/model_executor/ops/triton_ops/test_v100_attn_kernels.py --benchmark
"""

import sys
import time
import unittest

import numpy as np
import paddle


def skip_if_no_gpu(func):
    """Skip test if no GPU available."""

    def wrapper(*args, **kwargs):
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
            raise unittest.SkipTest("No GPU available")
        return func(*args, **kwargs)

    return wrapper


def skip_if_no_triton(func):
    """Skip test if Triton is not available."""

    def wrapper(*args, **kwargs):
        try:
            import triton  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Triton not available")
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Reference Python implementations for comparison
# ---------------------------------------------------------------------------


def ref_compute_positions(batch_id_per_token, cu_seqlens_q, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time):
    """Python reference: compute per-token positions."""
    num_tokens = batch_id_per_token.shape[0]
    positions = []
    batch_token_counts = {}

    for token_idx in range(num_tokens):
        batch_id = int(batch_id_per_token[token_idx].item())
        if batch_id not in batch_token_counts:
            batch_token_counts[batch_id] = 0

        encoder_len = int(seq_lens_encoder[batch_id].item())
        decoder_len = int(seq_lens_decoder[batch_id].item())
        this_time_len = int(seq_lens_this_time[batch_id].item())

        is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)

        if is_prefill:
            pos = batch_token_counts[batch_id]
        else:
            pos = encoder_len + decoder_len + batch_token_counts[batch_id]

        positions.append(pos)
        batch_token_counts[batch_id] += 1

    return paddle.to_tensor(positions, dtype="int64")


def ref_apply_rope_neox(q, k, cos, sin, positions):
    """Python reference: neox-style RoPE."""
    head_dim = q.shape[2]
    half_dim = head_dim // 2

    cos_vals = cos[positions]  # [num_tokens, rotary_dim]
    sin_vals = sin[positions]
    cos_exp = cos_vals.unsqueeze(1)[:, :, :half_dim]
    sin_exp = sin_vals.unsqueeze(1)[:, :, :half_dim]

    q1, q2 = q[:, :, :half_dim], q[:, :, half_dim:]
    k1, k2 = k[:, :, :half_dim], k[:, :, half_dim:]

    q_out = paddle.concat([q1 * cos_exp - q2 * sin_exp, q2 * cos_exp + q1 * sin_exp], axis=-1)
    k_out = paddle.concat([k1 * cos_exp - k2 * sin_exp, k2 * cos_exp + k1 * sin_exp], axis=-1)
    return q_out, k_out


def ref_apply_rope_interleaved(q, k, cos, sin, positions):
    """Python reference: interleaved-style RoPE."""
    num_tokens, num_heads, head_dim = q.shape
    kv_num_heads = k.shape[1]

    cos_vals = cos[positions].unsqueeze(1)
    sin_vals = sin[positions].unsqueeze(1)

    q_even, q_odd = q[:, :, 0::2], q[:, :, 1::2]
    k_even, k_odd = k[:, :, 0::2], k[:, :, 1::2]

    q_out = paddle.stack(
        [q_even * cos_vals - q_odd * sin_vals, q_odd * cos_vals + q_even * sin_vals], axis=-1
    ).reshape([num_tokens, num_heads, head_dim])

    k_out = paddle.stack(
        [k_even * cos_vals - k_odd * sin_vals, k_odd * cos_vals + k_even * sin_vals], axis=-1
    ).reshape([num_tokens, kv_num_heads, head_dim])

    return q_out, k_out


def ref_write_kv_cache(k, v, key_cache, value_cache, block_tables, positions, batch_id_per_token, block_size):
    """Python reference: write KV to block cache."""
    num_tokens = k.shape[0]
    for token_idx in range(num_tokens):
        pos = int(positions[token_idx].item())
        batch_id = int(batch_id_per_token[token_idx].item())
        block_idx = pos // block_size
        block_offset = pos % block_size
        physical_block = int(block_tables[batch_id, block_idx].item())
        key_cache[physical_block, :, block_offset, :] = k[token_idx]
        value_cache[physical_block, :, block_offset, :] = v[token_idx]


def ref_attention(q, k, v, is_causal=True):
    """Python reference: standard scaled dot-product attention."""
    # q: [q_len, num_heads, head_dim]
    # k: [kv_len, num_heads, head_dim]
    # v: [kv_len, num_heads, head_dim]
    q_len, num_heads, head_dim = q.shape
    kv_len = k.shape[0]
    scale = head_dim**-0.5

    q_t = q.transpose([1, 0, 2]).cast("float32")  # [num_heads, q_len, head_dim]
    k_t = k.transpose([1, 0, 2]).cast("float32")
    v_t = v.transpose([1, 0, 2]).cast("float32")

    scores = paddle.matmul(q_t, k_t.transpose([0, 2, 1])) * scale

    if is_causal:
        mask = paddle.zeros([q_len, kv_len], dtype="float32")
        for i in range(q_len):
            pos = kv_len - q_len + i
            if pos + 1 < kv_len:
                mask[i, pos + 1 :] = float("-inf")
        scores = scores + mask.unsqueeze(0)

    attn = paddle.nn.functional.softmax(scores, axis=-1)
    out = paddle.matmul(attn, v_t)
    return out.transpose([1, 0, 2]).cast(q.dtype)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestComputePositions(unittest.TestCase):
    """Test v100_compute_positions kernel."""

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_prefill_only(self):
        """Test positions for a pure prefill batch."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_compute_positions,
        )

        # Batch of 2 prefill sequences: lengths 4 and 3
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, 4, 7], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([4, 3], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([4, 3], dtype="int32")

        result = v100_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        expected = ref_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        np.testing.assert_array_equal(result.numpy(), expected.numpy())

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_decode_only(self):
        """Test positions for a pure decode batch."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_compute_positions,
        )

        # Batch of 3 decode sequences
        batch_id_per_token = paddle.to_tensor([0, 1, 2], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, 1, 2, 3], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([10, 20, 15], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([5, 3, 8], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([1, 1, 1], dtype="int32")

        result = v100_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        expected = ref_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        np.testing.assert_array_equal(result.numpy(), expected.numpy())

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_mixed_batch(self):
        """Test positions for a mixed prefill + decode batch."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_compute_positions,
        )

        # Batch 0: prefill (len=3), Batch 1: decode (len=1)
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 1], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, 3, 4], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([3, 10], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 5], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([3, 1], dtype="int32")

        result = v100_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        expected = ref_compute_positions(
            batch_id_per_token,
            cu_seqlens_q,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
        )
        np.testing.assert_array_equal(result.numpy(), expected.numpy())


class TestFusedRoPE(unittest.TestCase):
    """Test v100_fused_rope kernel."""

    def _make_rope_inputs(self, num_tokens, num_heads, kv_num_heads, head_dim, max_seq_len=128, neox_style=False):
        rotary_dim = head_dim if neox_style else head_dim // 2
        q = paddle.randn([num_tokens, num_heads, head_dim], dtype="float16")
        k = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        cos = paddle.randn([max_seq_len, rotary_dim], dtype="float16")
        sin = paddle.randn([max_seq_len, rotary_dim], dtype="float16")
        rotary_embs = paddle.stack(
            [
                cos.unsqueeze(0).unsqueeze(-2),  # [1, max_seq_len, 1, rotary_dim]
                sin.unsqueeze(0).unsqueeze(-2),
            ],
            axis=0,
        )  # [2, 1, max_seq_len, 1, rotary_dim]
        positions = paddle.to_tensor(np.random.randint(0, max_seq_len, size=num_tokens), dtype="int64")
        return q, k, cos, sin, rotary_embs, positions

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_interleaved_style(self):
        """Test interleaved RoPE style."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_fused_rope,
        )

        q, k, cos, sin, rotary_embs, positions = self._make_rope_inputs(
            num_tokens=8,
            num_heads=32,
            kv_num_heads=8,
            head_dim=128,
            neox_style=False,
        )
        q_ref, k_ref = ref_apply_rope_interleaved(
            q.cast("float32"),
            k.cast("float32"),
            cos.cast("float32"),
            sin.cast("float32"),
            positions,
        )

        q_triton = q.clone()
        k_triton = k.clone()
        v100_fused_rope(q_triton, k_triton, rotary_embs, positions, use_neox_style=False)

        np.testing.assert_allclose(
            q_triton.cast("float32").numpy(),
            q_ref.numpy(),
            atol=1e-2,
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            k_triton.cast("float32").numpy(),
            k_ref.numpy(),
            atol=1e-2,
            rtol=1e-2,
        )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_neox_style(self):
        """Test neox RoPE style."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_fused_rope,
        )

        q, k, cos, sin, rotary_embs, positions = self._make_rope_inputs(
            num_tokens=8,
            num_heads=32,
            kv_num_heads=8,
            head_dim=128,
            neox_style=True,
        )
        q_ref, k_ref = ref_apply_rope_neox(
            q.cast("float32"),
            k.cast("float32"),
            cos.cast("float32"),
            sin.cast("float32"),
            positions,
        )

        q_triton = q.clone()
        k_triton = k.clone()
        v100_fused_rope(q_triton, k_triton, rotary_embs, positions, use_neox_style=True)

        np.testing.assert_allclose(
            q_triton.cast("float32").numpy(),
            q_ref.numpy(),
            atol=1e-2,
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            k_triton.cast("float32").numpy(),
            k_ref.numpy(),
            atol=1e-2,
            rtol=1e-2,
        )


class TestWriteKVCache(unittest.TestCase):
    """Test v100_write_kv_cache kernel."""

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_basic_write(self):
        """Test basic KV cache write."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_write_kv_cache,
        )

        num_tokens = 7
        kv_num_heads = 8
        head_dim = 128
        block_size = 64
        max_num_blocks = 16

        k = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        v = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")

        # Two sequences: prefill len=4, decode at pos=15
        positions = paddle.to_tensor([0, 1, 2, 3, 15, 16, 17], dtype="int64")
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 1, 1, 1], dtype="int32")
        block_tables = paddle.to_tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="int32")

        # Triton write
        key_cache_triton = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        val_cache_triton = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        v100_write_kv_cache(k, v, key_cache_triton, val_cache_triton, block_tables, positions, batch_id_per_token)

        # Reference write
        key_cache_ref = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        val_cache_ref = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        ref_write_kv_cache(k, v, key_cache_ref, val_cache_ref, block_tables, positions, batch_id_per_token, block_size)

        np.testing.assert_array_equal(key_cache_triton.numpy(), key_cache_ref.numpy())
        np.testing.assert_array_equal(val_cache_triton.numpy(), val_cache_ref.numpy())


class TestDecodeAttention(unittest.TestCase):
    """Test v100_decode_attention (2-stage flash-decoding)."""

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_single_sequence(self):
        """Test decode attention for a single sequence."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_decode_attention,
        )

        num_heads = 32
        kv_num_heads = 8
        head_dim = 128
        group_size = num_heads // kv_num_heads
        block_size = 64
        kv_len = 100  # total KV length
        max_num_blocks = 16

        # Create KV cache with known data
        key_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        value_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")

        # Block table: 2 blocks for 100 tokens (block_size=64)
        block_tables = paddle.to_tensor([[0, 1, 2, 3]], dtype="int32")

        # Query: single decode token
        q = paddle.randn([1, num_heads, head_dim], dtype="float16")

        # Gather KV from cache for reference
        k_seq = paddle.concat(
            [
                key_cache[0, :, :, :].transpose([1, 0, 2]),  # block 0: 64 tokens
                key_cache[1, :, :36, :].transpose([1, 0, 2]),  # block 1: 36 tokens
            ],
            axis=0,
        )  # [100, kv_num_heads, head_dim]
        v_seq = paddle.concat(
            [
                value_cache[0, :, :, :].transpose([1, 0, 2]),
                value_cache[1, :, :36, :].transpose([1, 0, 2]),
            ],
            axis=0,
        )

        # Expand for GQA
        k_expanded = k_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])
        v_expanded = v_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])

        # Reference attention
        ref_out = ref_attention(q, k_expanded, v_expanded, is_causal=True)

        # Triton attention
        output = paddle.empty([1, num_heads, head_dim], dtype="float16")
        seq_lens = paddle.to_tensor([kv_len], dtype="int32")
        q_start_locs = paddle.to_tensor([0], dtype="int32")

        v100_decode_attention(
            q,
            key_cache,
            value_cache,
            output,
            block_tables,
            seq_lens,
            q_start_locs,
            num_heads,
            kv_num_heads,
            head_dim,
            head_dim**-0.5,
        )

        np.testing.assert_allclose(
            output.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            atol=5e-2,
            rtol=5e-2,
        )


class TestExtendAttention(unittest.TestCase):
    """Test v100_extend_attention (tiled flash attention for prefill)."""

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_single_sequence_prefill(self):
        """Test prefill attention for a single sequence."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_extend_attention,
        )

        num_heads = 32
        kv_num_heads = 8
        head_dim = 128
        group_size = num_heads // kv_num_heads
        block_size = 64
        q_len = 20
        kv_len = 20  # prefill: q_len == kv_len
        max_num_blocks = 16

        key_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        value_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        block_tables = paddle.to_tensor([[0, 1, 2, 3]], dtype="int32")

        q = paddle.randn([q_len, num_heads, head_dim], dtype="float16")

        # Gather KV from cache
        k_seq = key_cache[0, :, :kv_len, :].transpose([1, 0, 2])  # [kv_len, kv_num_heads, head_dim]
        v_seq = value_cache[0, :, :kv_len, :].transpose([1, 0, 2])

        k_expanded = k_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])
        v_expanded = v_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])

        ref_out = ref_attention(q, k_expanded, v_expanded, is_causal=True)

        output = paddle.empty([q_len, num_heads, head_dim], dtype="float16")
        q_start_locs = paddle.to_tensor([0], dtype="int32")
        q_seq_lens = paddle.to_tensor([q_len], dtype="int32")
        kv_seq_lens = paddle.to_tensor([kv_len], dtype="int32")

        v100_extend_attention(
            q,
            key_cache,
            value_cache,
            output,
            block_tables,
            q_start_locs,
            q_seq_lens,
            kv_seq_lens,
            num_heads,
            kv_num_heads,
            head_dim,
            head_dim**-0.5,
            is_causal=True,
        )

        np.testing.assert_allclose(
            output.cast("float32").numpy(),
            ref_out.cast("float32").numpy(),
            atol=5e-2,
            rtol=5e-2,
        )


# ---------------------------------------------------------------------------
# Performance Benchmark
# ---------------------------------------------------------------------------


def run_benchmark():
    """Run performance benchmark comparing Triton vs Python fallback."""
    try:
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_compute_positions,
            v100_fused_rope,
            v100_write_kv_cache,
        )
    except ImportError:
        print("ERROR: Triton kernels not available, cannot benchmark.")
        return

    print("=" * 70)
    print("V100 Triton Attention Kernels - Performance Benchmark")
    print("=" * 70)

    warmup = 10
    repeat = 100

    # --- Benchmark: Compute Positions ---
    for num_tokens in [32, 128, 512, 2048]:
        batch_size = min(num_tokens, 32)
        tokens_per_batch = num_tokens // batch_size
        batch_ids = []
        for b in range(batch_size):
            batch_ids.extend([b] * tokens_per_batch)
        batch_id_per_token = paddle.to_tensor(batch_ids[:num_tokens], dtype="int32")

        cu_seqs = [0]
        for b in range(batch_size):
            cu_seqs.append(cu_seqs[-1] + tokens_per_batch)
        cu_seqlens_q = paddle.to_tensor(cu_seqs, dtype="int32")
        seq_lens_encoder = paddle.full([batch_size], tokens_per_batch, dtype="int32")
        seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_this_time = paddle.full([batch_size], tokens_per_batch, dtype="int32")

        # Warmup
        for _ in range(warmup):
            v100_compute_positions(
                batch_id_per_token, cu_seqlens_q, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time
            )
        paddle.device.cuda.synchronize()

        # Triton
        start = time.perf_counter()
        for _ in range(repeat):
            v100_compute_positions(
                batch_id_per_token, cu_seqlens_q, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time
            )
        paddle.device.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / repeat * 1000

        # Python reference
        start = time.perf_counter()
        for _ in range(repeat):
            ref_compute_positions(
                batch_id_per_token, cu_seqlens_q, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time
            )
        paddle.device.cuda.synchronize()
        python_time = (time.perf_counter() - start) / repeat * 1000

        speedup = python_time / triton_time if triton_time > 0 else float("inf")
        print(
            f"[compute_positions] tokens={num_tokens:>5d}  "
            f"Triton={triton_time:.3f}ms  Python={python_time:.3f}ms  "
            f"Speedup={speedup:.1f}x"
        )

    print()

    # --- Benchmark: Fused RoPE ---
    for num_tokens in [32, 128, 512]:
        num_heads = 32
        kv_num_heads = 8
        head_dim = 128
        max_seq_len = 2048
        rotary_dim = head_dim // 2

        q = paddle.randn([num_tokens, num_heads, head_dim], dtype="float16")
        k = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        rotary_embs = paddle.randn([2, 1, max_seq_len, 1, rotary_dim], dtype="float16")
        positions = paddle.randint(0, max_seq_len, [num_tokens], dtype="int64")

        for _ in range(warmup):
            q_c, k_c = q.clone(), k.clone()
            v100_fused_rope(q_c, k_c, rotary_embs, positions, use_neox_style=False)
        paddle.device.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeat):
            q_c, k_c = q.clone(), k.clone()
            v100_fused_rope(q_c, k_c, rotary_embs, positions, use_neox_style=False)
        paddle.device.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / repeat * 1000

        cos = rotary_embs[0, 0, :, 0, :]
        sin = rotary_embs[1, 0, :, 0, :]
        cos_f32 = cos.cast("float32")
        sin_f32 = sin.cast("float32")
        for _ in range(warmup):
            q_c, k_c = q.clone(), k.clone()
            ref_apply_rope_interleaved(q_c.cast("float32"), k_c.cast("float32"), cos_f32, sin_f32, positions)
        paddle.device.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeat):
            q_c, k_c = q.clone(), k.clone()
            ref_apply_rope_interleaved(q_c.cast("float32"), k_c.cast("float32"), cos_f32, sin_f32, positions)
        paddle.device.cuda.synchronize()
        python_time = (time.perf_counter() - start) / repeat * 1000

        speedup = python_time / triton_time if triton_time > 0 else float("inf")
        print(
            f"[fused_rope]        tokens={num_tokens:>5d}  "
            f"Triton={triton_time:.3f}ms  Python={python_time:.3f}ms  "
            f"Speedup={speedup:.1f}x"
        )

    print()

    # --- Benchmark: Write KV Cache ---
    for num_tokens in [32, 128, 512]:
        kv_num_heads = 8
        head_dim = 128
        block_size = 64
        max_num_blocks = 256

        k = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        v = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        positions = paddle.arange(0, num_tokens, dtype="int64")
        batch_id_per_token = paddle.zeros([num_tokens], dtype="int32")
        block_tables = paddle.arange(0, 16, dtype="int32").unsqueeze(0)

        key_cache = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        val_cache = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")

        for _ in range(warmup):
            v100_write_kv_cache(k, v, key_cache, val_cache, block_tables, positions, batch_id_per_token)
        paddle.device.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeat):
            v100_write_kv_cache(k, v, key_cache, val_cache, block_tables, positions, batch_id_per_token)
        paddle.device.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / repeat * 1000

        key_cache2 = paddle.zeros_like(key_cache)
        val_cache2 = paddle.zeros_like(val_cache)
        start = time.perf_counter()
        for _ in range(repeat):
            ref_write_kv_cache(k, v, key_cache2, val_cache2, block_tables, positions, batch_id_per_token, block_size)
        paddle.device.cuda.synchronize()
        python_time = (time.perf_counter() - start) / repeat * 1000

        speedup = python_time / triton_time if triton_time > 0 else float("inf")
        print(
            f"[write_kv_cache]    tokens={num_tokens:>5d}  "
            f"Triton={triton_time:.3f}ms  Python={python_time:.3f}ms  "
            f"Speedup={speedup:.1f}x"
        )

    print()
    print("=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        sys.argv.remove("--benchmark")
        run_benchmark()
    else:
        unittest.main()
