"""
Triton MLA cache write kernel tests — correctness, determinism, and edge cases.

Tests for mla_write_cache_triton from
fastdeploy/model_executor/layers/attention/triton_ops/mla_cache_kernel.py

Correctness verification strategy:
    A naive Python reference implementation writes [compressed_kv || k_pe] into
    the paged cache at slot_mapping positions. The Triton kernel output is compared
    against the reference using exact bitwise equality (pure copy, no arithmetic).

Test scenarios:
1. Basic write: single batch, sequential slots
2. Multi-batch: multiple batches with different slot positions
3. Non-contiguous slots: random slot assignments across blocks
4. k_pe with extra dim: [num_tokens, 1, qk_rope_head_dim] shape
5. Determinism: multiple runs produce identical cache
6. Edge cases: single token, large batch, unaligned dims

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/deterministic/test_triton_mla_cache_kernel.py -v
"""

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers.attention.triton_ops.mla_cache_kernel import (
    mla_write_cache_triton,
)

# ---------------------------------------------------------------------------
# Skip if no CUDA
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0,
    reason="Triton MLA cache kernel requires CUDA",
)

# ---------------------------------------------------------------------------
# Typical MLA dimensions
# ---------------------------------------------------------------------------
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def mla_write_cache_ref(compressed_kv, k_pe, latent_cache, slot_mapping):
    """
    Naive Python reference for MLA cache write.

    Writes [compressed_kv || k_pe] into paged latent_cache at slot_mapping positions.

    Args:
        compressed_kv: [num_tokens, kv_lora_rank]
        k_pe: [num_tokens, 1, qk_rope_head_dim] or [num_tokens, qk_rope_head_dim]
        latent_cache: [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
        slot_mapping: [num_tokens] int64
    """
    num_tokens = compressed_kv.shape[0]
    kv_lora_rank = compressed_kv.shape[-1]

    ckv_np = compressed_kv.astype("float32").numpy()
    kpe_np = k_pe.reshape([num_tokens, -1]).astype("float32").numpy()
    cache_np = latent_cache.astype("float32").numpy()
    slots_np = slot_mapping.numpy()

    kv_block_size = latent_cache.shape[2]

    for i in range(num_tokens):
        slot = int(slots_np[i])
        block_id = slot // kv_block_size
        offset = slot % kv_block_size
        # Write [compressed_kv || k_pe] into cache
        cache_np[block_id, 0, offset, :kv_lora_rank] = ckv_np[i]
        cache_np[block_id, 0, offset, kv_lora_rank:] = kpe_np[i]

    return paddle.to_tensor(cache_np)


# ---------------------------------------------------------------------------
# Helper: build test data
# ---------------------------------------------------------------------------
def build_cache_test_data(
    num_tokens,
    kv_lora_rank=KV_LORA_RANK,
    qk_rope_head_dim=QK_ROPE_HEAD_DIM,
    block_size=BLOCK_SIZE,
    num_blocks=None,
    dtype="bfloat16",
    kpe_3d=False,
    seed=42,
    slot_mapping=None,
):
    """Build test data for mla_write_cache_triton."""
    np.random.seed(seed)
    paddle.seed(seed)

    if num_blocks is None:
        num_blocks = max((num_tokens + block_size - 1) // block_size + 4, 8)

    latent_dim = kv_lora_rank + qk_rope_head_dim
    compressed_kv = paddle.randn([num_tokens, kv_lora_rank]).astype(dtype)

    if kpe_3d:
        k_pe = paddle.randn([num_tokens, 1, qk_rope_head_dim]).astype(dtype)
    else:
        k_pe = paddle.randn([num_tokens, qk_rope_head_dim]).astype(dtype)

    latent_cache = paddle.zeros([num_blocks, 1, block_size, latent_dim]).astype(dtype)

    if slot_mapping is None:
        slot_mapping = paddle.arange(num_tokens, dtype="int64")
    else:
        slot_mapping = paddle.to_tensor(slot_mapping, dtype="int64")

    return {
        "compressed_kv": compressed_kv,
        "k_pe": k_pe,
        "latent_cache": latent_cache,
        "slot_mapping": slot_mapping,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "block_size": block_size,
    }


# ===========================================================================
# Basic correctness tests
# ===========================================================================
class TestMLAWriteCacheBasic:
    """Basic correctness tests for mla_write_cache_triton."""

    def test_sequential_slots(self):
        """Sequential slot mapping: tokens 0..N-1 map to slots 0..N-1."""
        data = build_cache_test_data(num_tokens=16)

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
            err_msg="Sequential slots: triton vs ref mismatch",
        )

    def test_single_token(self):
        """Single token write."""
        data = build_cache_test_data(num_tokens=1)

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
        )

    def test_large_batch(self):
        """Large batch of tokens."""
        data = build_cache_test_data(num_tokens=512, num_blocks=64)

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
        )


# ===========================================================================
# Non-contiguous slot tests
# ===========================================================================
class TestMLAWriteCacheNonContiguous:
    """Tests with non-sequential slot mappings."""

    def test_scattered_slots(self):
        """Slots scattered across multiple blocks."""
        num_tokens = 8
        block_size = 4
        num_blocks = 16
        # Scatter tokens across different blocks
        slots = [0, 5, 10, 15, 20, 25, 30, 35]

        data = build_cache_test_data(
            num_tokens=num_tokens,
            block_size=block_size,
            num_blocks=num_blocks,
            slot_mapping=slots,
        )

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
            err_msg="Scattered slots: triton vs ref mismatch",
        )

    def test_random_slots(self):
        """Random slot assignments."""
        num_tokens = 32
        block_size = 8
        num_blocks = 32
        np.random.seed(123)
        # Generate unique random slots within valid range
        max_slot = num_blocks * block_size
        slots = np.random.choice(max_slot, size=num_tokens, replace=False).tolist()

        data = build_cache_test_data(
            num_tokens=num_tokens,
            block_size=block_size,
            num_blocks=num_blocks,
            slot_mapping=slots,
        )

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
            err_msg="Random slots: triton vs ref mismatch",
        )


# ===========================================================================
# k_pe shape variants
# ===========================================================================
class TestMLAWriteCacheKpeShape:
    """Tests for different k_pe tensor shapes."""

    def test_kpe_2d(self):
        """k_pe shape: [num_tokens, qk_rope_head_dim]."""
        data = build_cache_test_data(num_tokens=16, kpe_3d=False)

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
        )

    def test_kpe_3d(self):
        """k_pe shape: [num_tokens, 1, qk_rope_head_dim]."""
        data = build_cache_test_data(num_tokens=16, kpe_3d=True)

        cache_ref = mla_write_cache_ref(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"].clone(),
            data["slot_mapping"],
        )

        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            data["latent_cache"],
            data["slot_mapping"],
        )

        np.testing.assert_allclose(
            data["latent_cache"].astype("float32").numpy(),
            cache_ref.numpy(),
            atol=1e-6,
        )


# ===========================================================================
# Parametrized dtype/dimension tests
# ===========================================================================
_DIMENSION_CASES = [
    # (name, kv_lora_rank, qk_rope_head_dim) — typical MLA configs
    ("deepseek_v3", 512, 64),
    ("deepseek_v2_lite", 256, 32),
    ("small_dims", 128, 32),
]


@pytest.mark.parametrize(
    "name,kv_lora_rank,qk_rope_head_dim",
    _DIMENSION_CASES,
    ids=[c[0] for c in _DIMENSION_CASES],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_write_cache_dimensions(name, kv_lora_rank, qk_rope_head_dim, dtype):
    """Test cache write across different MLA dimension configurations and dtypes."""
    data = build_cache_test_data(
        num_tokens=32,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
    )

    cache_ref = mla_write_cache_ref(
        data["compressed_kv"],
        data["k_pe"],
        data["latent_cache"].clone(),
        data["slot_mapping"],
    )

    mla_write_cache_triton(
        data["compressed_kv"],
        data["k_pe"],
        data["latent_cache"],
        data["slot_mapping"],
    )

    np.testing.assert_allclose(
        data["latent_cache"].astype("float32").numpy(),
        cache_ref.numpy(),
        atol=1e-6,
        err_msg=f"[{name}/{dtype}] triton vs ref mismatch",
    )


# ===========================================================================
# Determinism test
# ===========================================================================
def test_write_cache_determinism():
    """Multiple runs should produce bitwise identical cache contents."""
    data = build_cache_test_data(num_tokens=64)

    results = []
    for _ in range(5):
        cache = data["latent_cache"].clone()
        mla_write_cache_triton(
            data["compressed_kv"],
            data["k_pe"],
            cache,
            data["slot_mapping"],
        )
        results.append(cache.astype("float32").numpy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0], results[i],
            err_msg=f"Run 0 vs run {i} differ — non-deterministic!"
        )


# ===========================================================================
# Manual baseline test (hand-crafted small tensors)
# ===========================================================================
def test_manual_baseline():
    """Hand-crafted small tensors to verify exact values end up in the right cache slots."""
    kv_lora_rank = 4
    qk_rope_head_dim = 2
    block_size = 2
    num_blocks = 4
    latent_dim = kv_lora_rank + qk_rope_head_dim  # 6

    # 3 tokens, deterministic values
    compressed_kv = paddle.to_tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ], dtype="float32")

    k_pe = paddle.to_tensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ], dtype="float32")

    latent_cache = paddle.zeros([num_blocks, 1, block_size, latent_dim], dtype="float32")

    # slot_mapping: token 0 -> slot 0 (block 0, offset 0)
    #               token 1 -> slot 3 (block 1, offset 1)
    #               token 2 -> slot 5 (block 2, offset 1)
    slot_mapping = paddle.to_tensor([0, 3, 5], dtype="int64")

    mla_write_cache_triton(compressed_kv, k_pe, latent_cache, slot_mapping)

    cache_np = latent_cache.numpy()

    # Token 0 -> block 0, offset 0: [1, 2, 3, 4, 0.1, 0.2]
    expected_0 = np.array([1.0, 2.0, 3.0, 4.0, 0.1, 0.2], dtype=np.float32)
    np.testing.assert_allclose(cache_np[0, 0, 0, :], expected_0, atol=1e-6)

    # Token 1 -> block 1, offset 1: [5, 6, 7, 8, 0.3, 0.4]
    expected_1 = np.array([5.0, 6.0, 7.0, 8.0, 0.3, 0.4], dtype=np.float32)
    np.testing.assert_allclose(cache_np[1, 0, 1, :], expected_1, atol=1e-6)

    # Token 2 -> block 2, offset 1: [9, 10, 11, 12, 0.5, 0.6]
    expected_2 = np.array([9.0, 10.0, 11.0, 12.0, 0.5, 0.6], dtype=np.float32)
    np.testing.assert_allclose(cache_np[2, 0, 1, :], expected_2, atol=1e-6)

    # All other slots should remain zero
    zero_slots = [
        (0, 0, 1),  # block 0, offset 1
        (1, 0, 0),  # block 1, offset 0
        (2, 0, 0),  # block 2, offset 0
        (3, 0, 0),  # block 3, offset 0
        (3, 0, 1),  # block 3, offset 1
    ]
    for blk, head, off in zero_slots:
        np.testing.assert_array_equal(
            cache_np[blk, head, off, :],
            np.zeros(latent_dim, dtype=np.float32),
            err_msg=f"Slot ({blk},{head},{off}) should be zero",
        )


# ===========================================================================
# Empty token test
# ===========================================================================
def test_empty_tokens():
    """Zero tokens should be a no-op (no crash)."""
    latent_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    compressed_kv = paddle.empty([0, KV_LORA_RANK], dtype="bfloat16")
    k_pe = paddle.empty([0, QK_ROPE_HEAD_DIM], dtype="bfloat16")
    latent_cache = paddle.zeros([4, 1, BLOCK_SIZE, latent_dim], dtype="bfloat16")
    slot_mapping = paddle.empty([0], dtype="int64")

    # Should not crash
    mla_write_cache_triton(compressed_kv, k_pe, latent_cache, slot_mapping)

    # Cache should remain all zeros
    np.testing.assert_array_equal(
        latent_cache.astype("float32").numpy(),
        np.zeros_like(latent_cache.astype("float32").numpy()),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
