"""
Test script for V100 Triton attention kernels.

Tests numerical correctness by comparing Triton kernel outputs against
Python reference implementations. Also provides basic performance benchmarks.

Usage:
    # Run on a V100 GPU:
    python tests/model_executor/ops/triton_ops/test_v100_attn_kernels.py

    # Run specific test:
    python -m pytest tests/model_executor/ops/triton_ops/test_v100_attn_kernels.py::TestWriteKVCache -v

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

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_write_kv_heads_2(self):
        """Test KV cache write with kv_num_heads=2 (ERNIE 4.5 0.3B config)."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_write_kv_cache,
        )

        num_tokens = 6
        kv_num_heads = 2
        head_dim = 128
        block_size = 64
        max_num_blocks = 16

        k = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")
        v = paddle.randn([num_tokens, kv_num_heads, head_dim], dtype="float16")

        positions = paddle.to_tensor([0, 1, 2, 3, 4, 5], dtype="int64")
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 0, 0, 0], dtype="int32")
        block_tables = paddle.to_tensor([[0, 1, 2, 3]], dtype="int32")

        key_cache_triton = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        val_cache_triton = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        v100_write_kv_cache(k, v, key_cache_triton, val_cache_triton, block_tables, positions, batch_id_per_token)

        key_cache_ref = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        val_cache_ref = paddle.zeros([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        ref_write_kv_cache(k, v, key_cache_ref, val_cache_ref, block_tables, positions, batch_id_per_token, block_size)

        np.testing.assert_array_equal(key_cache_triton.numpy(), key_cache_ref.numpy())
        np.testing.assert_array_equal(val_cache_triton.numpy(), val_cache_ref.numpy())


class TestDecodeFusedAttention(unittest.TestCase):
    """Test v100_decode_fused (fused KV write + flash-decoding)."""

    def _run_decode_fused(self, num_heads, kv_num_heads, head_dim, block_size, kv_lens, max_num_blocks=16):
        """Helper: run v100_decode_fused and compare against reference."""
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
            v100_decode_fused,
        )

        batch_size = len(kv_lens)
        group_size = num_heads // kv_num_heads

        key_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")
        value_cache = paddle.randn([max_num_blocks, kv_num_heads, block_size, head_dim], dtype="float16")

        # Each sequence gets its own blocks
        blocks_per_seq = max_num_blocks // batch_size
        block_table_list = []
        for i in range(batch_size):
            block_table_list.append(list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq)))
        block_tables = paddle.to_tensor(block_table_list, dtype="int32")

        q = paddle.randn([batch_size, num_heads, head_dim], dtype="float16")

        # New K/V for KV write (1 new token per seq at the end)
        k_new = paddle.randn([batch_size, kv_num_heads, head_dim], dtype="float16")
        v_new = paddle.randn([batch_size, kv_num_heads, head_dim], dtype="float16")

        # Positions: each token is at the end of its sequence (kv_len - 1)
        positions = paddle.to_tensor([kv - 1 for kv in kv_lens], dtype="int64")
        batch_id_per_token = paddle.to_tensor(list(range(batch_size)), dtype="int32")
        seq_lens = paddle.to_tensor(kv_lens, dtype="int32")
        q_start_locs = paddle.to_tensor(list(range(batch_size)), dtype="int32")
        max_kv_len = max(kv_lens)

        output = paddle.empty([batch_size, num_heads, head_dim], dtype="float16")

        v100_decode_fused(
            q,
            k_new,
            v_new,
            key_cache,
            value_cache,
            output,
            block_tables,
            seq_lens,
            positions,
            batch_id_per_token,
            q_start_locs,
            num_heads,
            kv_num_heads,
            head_dim,
            head_dim**-0.5,
            max_kv_len=max_kv_len,
        )

        # Verify no NaN
        self.assertFalse(
            np.any(np.isnan(output.cast("float32").numpy())),
            "Decode fused attention output contains NaN",
        )

        # Verify each sequence against reference
        for i, kv_len in enumerate(kv_lens):
            # Write the new K/V to the reference cache at the correct position
            key_cache_ref = key_cache.clone()
            value_cache_ref = value_cache.clone()
            pos = kv_len - 1
            blk_idx = pos // block_size
            blk_off = pos % block_size
            phys_blk = int(block_tables[i, blk_idx].item())
            key_cache_ref[phys_blk, :, blk_off, :] = k_new[i]
            value_cache_ref[phys_blk, :, blk_off, :] = v_new[i]

            # Gather full KV from cache
            num_blocks = (kv_len + block_size - 1) // block_size
            k_blocks = []
            v_blocks = []
            remaining = kv_len
            for b in range(num_blocks):
                phys_block = int(block_tables[i, b].item())
                tokens_in_block = min(block_size, remaining)
                k_blocks.append(key_cache_ref[phys_block, :, :tokens_in_block, :].transpose([1, 0, 2]))
                v_blocks.append(value_cache_ref[phys_block, :, :tokens_in_block, :].transpose([1, 0, 2]))
                remaining -= tokens_in_block
            k_seq = paddle.concat(k_blocks, axis=0)
            v_seq = paddle.concat(v_blocks, axis=0)

            k_expanded = k_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])
            v_expanded = v_seq.unsqueeze(2).tile([1, 1, group_size, 1]).reshape([kv_len, num_heads, head_dim])

            q_i = q[i : i + 1]
            ref_out = ref_attention(q_i, k_expanded, v_expanded, is_causal=True)

            np.testing.assert_allclose(
                output[i : i + 1].cast("float32").numpy(),
                ref_out.cast("float32").numpy(),
                atol=5e-2,
                rtol=5e-2,
                err_msg=f"Sequence {i} (kv_len={kv_len}) mismatch",
            )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_single_sequence(self):
        """Test decode fused attention for a single sequence."""
        self._run_decode_fused(
            num_heads=32,
            kv_num_heads=8,
            head_dim=128,
            block_size=64,
            kv_lens=[100],
        )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_small_kv_len(self):
        """Test decode fused with small kv_len (e.g. 7), simulating early decode steps."""
        self._run_decode_fused(
            num_heads=16,
            kv_num_heads=4,
            head_dim=128,
            block_size=64,
            kv_lens=[7],
            max_num_blocks=8,
        )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_multi_sequence_decode(self):
        """Test decode fused with multiple sequences of varying kv_len."""
        self._run_decode_fused(
            num_heads=16,
            kv_num_heads=4,
            head_dim=128,
            block_size=64,
            kv_lens=[7, 65, 3],
        )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_decode_kv_heads_2(self):
        """Test decode fused with num_heads=8, kv_num_heads=2 (ERNIE 4.5 0.3B config)."""
        self._run_decode_fused(
            num_heads=8,
            kv_num_heads=2,
            head_dim=128,
            block_size=64,
            kv_lens=[7],
            max_num_blocks=8,
        )

    @skip_if_no_gpu
    @skip_if_no_triton
    def test_decode_multi_split(self):
        """Test decode fused with kv_len large enough to trigger num_kv_splits > 1."""
        self._run_decode_fused(
            num_heads=8,
            kv_num_heads=2,
            head_dim=128,
            block_size=64,
            kv_lens=[600],
        )


# ---------------------------------------------------------------------------
# Performance Benchmark
# ---------------------------------------------------------------------------


def run_benchmark():
    """Run performance benchmark for Triton write_kv_cache kernel."""
    try:
        from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
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
