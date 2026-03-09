"""
Test: Is matmul_persistent (Triton) M-invariant for QKV projection GEMM?

M-invariance means: for the same token data at tail positions,
  matmul(full_batch, W)[-tail:] == matmul(tail_only, W)
regardless of how many other rows precede them.

This is the root cause of prefix caching non-determinism when using cuBLAS.
"""

import paddle
import pytest

from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    matmul_persistent,
)

# (M_full, M_tail, K, N)
SHAPES = [
    (825, 57, 3584, 4608),  # Qwen2-7B real case
    (1024, 128, 4096, 4096),  # power-of-2
    (512, 1, 3584, 4608),  # single-token tail
]
N_SEEDS = 5


def _check_m_invariance(fn, M_full, M_tail, K, N, seed):
    """Return max abs diff between fn(full)[-tail:] and fn(tail)."""
    paddle.seed(seed)
    W = paddle.randn([K, N], dtype="bfloat16")
    full_input = paddle.randn([M_full, K], dtype="bfloat16")
    tail_input = full_input[-M_tail:].clone()

    full_out = fn(full_input, W)
    tail_out = fn(tail_input, W)
    diff = (full_out[-M_tail:].astype("float32") - tail_out.astype("float32")).abs()
    return float(diff.max().item())


def test_cublas_is_m_non_invariant():
    """Confirm cuBLAS matmul is NOT M-invariant (baseline, documents the problem)."""
    non_invariant_count = 0
    for M_full, M_tail, K, N in SHAPES:
        for seed in range(N_SEEDS):
            diff = _check_m_invariance(paddle.matmul, M_full, M_tail, K, N, seed)
            if diff > 0:
                non_invariant_count += 1
    assert non_invariant_count > 0, (
        "Expected cuBLAS bf16 matmul to be M-non-invariant in at least one case. "
        "If this fails, cuBLAS behavior may have changed."
    )


@pytest.mark.parametrize("M_full,M_tail,K,N", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_triton_persistent_is_m_invariant(M_full, M_tail, K, N):
    """Triton persistent matmul must be bit-identical regardless of M."""
    for seed in range(N_SEEDS):
        diff = _check_m_invariance(matmul_persistent, M_full, M_tail, K, N, seed)
        assert diff == 0.0, (
            f"matmul_persistent NOT M-invariant: " f"shape=({M_full},{M_tail},{K},{N}) seed={seed} diff={diff}"
        )


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
