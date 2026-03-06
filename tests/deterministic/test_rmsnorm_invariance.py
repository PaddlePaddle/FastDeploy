"""
Test: Is fused_rms_norm M-invariant?

The real inference path: bf16 input, bf16 weight, called via
  paddle.incubate.nn.functional.fused_rms_norm(x, w, None, eps, 1)
which dispatches to _C_ops.fused_rms_norm_quant internally.

M-invariance means: for the same token data at position i,
  fused_rms_norm(full_batch)[i] == fused_rms_norm(sub_batch)[mapped_i]
regardless of how many other rows are in the batch.
"""

import hashlib

import paddle


def md5(t):
    return hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()[:16]


def check_m_invariance(M_full, M_tail, hidden_size, dtype, seed, norm_fn=None):
    """Return max abs diff between full[-M_tail:] and part for a norm function."""
    paddle.seed(seed)
    w = paddle.randn([hidden_size], dtype=dtype)
    full = paddle.randn([M_full, hidden_size], dtype=dtype)
    part = full[-M_tail:].clone()

    if norm_fn is None:
        norm_fn = lambda x, w, eps: paddle.incubate.nn.functional.fused_rms_norm(x, w, None, eps, 1)[0]

    a = norm_fn(full, w, 1e-6)[-M_tail:]
    b = norm_fn(part, w, 1e-6)

    if dtype == "bfloat16":
        diff = float((a.astype("float32") - b.astype("float32")).abs().max())
    else:
        diff = float((a - b).abs().max())
    return diff


def test_m_invariance_bf16_multi_seed():
    """bf16 path (real inference): test across 10 seeds to avoid lucky zeros."""
    fails = []
    for seed in range(10):
        diff = check_m_invariance(825, 57, 3584, "bfloat16", seed)
        tag = "FAIL" if diff > 0 else "ok"
        print(f"  seed={seed} diff={diff} {tag}")
        if diff > 0:
            fails.append((seed, diff))
    print(f"\n  bf16: {len(fails)}/10 seeds show M-non-invariance")
    assert len(fails) > 0, (
        "Expected fused_rms_norm bf16 to be M-non-invariant, "
        "but all 10 seeds passed. Check if Paddle kernel changed."
    )


def test_m_invariance_f32_multi_seed():
    """f32 path: test across 10 seeds."""
    fails = []
    for seed in range(10):
        diff = check_m_invariance(825, 57, 3584, "float32", seed)
        tag = "FAIL" if diff > 0 else "ok"
        print(f"  seed={seed} diff={diff} {tag}")
        if diff > 0:
            fails.append((seed, diff))
    print(f"\n  f32: {len(fails)}/10 seeds show M-non-invariance")
    assert len(fails) > 0, "Expected fused_rms_norm f32 to be M-non-invariant, " "but all 10 seeds passed."


def test_m_invariance_various_shapes():
    """Test M-invariance across different (M_full, M_tail) combos in bf16.
    Use multiple seeds per shape to avoid false negatives from bf16 truncation."""
    shapes = [
        (825, 57),  # real case: 825 total, 57 new tokens
        (1024, 128),  # power-of-2
        (2048, 1),  # single token tail
        (512, 256),  # half split
        (100, 99),  # almost equal
    ]
    n_seeds = 10
    fails = []
    for M_full, M_tail in shapes:
        max_diff = 0.0
        fail_count = 0
        for seed in range(n_seeds):
            diff = check_m_invariance(M_full, M_tail, 3584, "bfloat16", seed)
            max_diff = max(max_diff, diff)
            if diff > 0:
                fail_count += 1
        tag = "FAIL" if fail_count > 0 else "ok"
        print(f"  M_full={M_full} M_tail={M_tail} fail={fail_count}/{n_seeds} max_diff={max_diff} {tag}")
        if fail_count > 0:
            fails.append((M_full, M_tail, max_diff))
    print(f"\n  {len(fails)}/{len(shapes)} shapes show M-non-invariance")


def test_triton_rms_norm_m_invariant():
    """Verify our Triton batch-invariant RMSNorm IS M-invariant across all seeds and shapes."""
    from fastdeploy.model_executor.layers.batch_invariant_ops import (
        rms_norm_batch_invariant,
    )

    triton_fn = lambda x, w, eps: rms_norm_batch_invariant(x, w, eps)

    shapes = [(825, 57), (1024, 128), (2048, 1), (512, 256)]
    n_seeds = 10
    for M_full, M_tail in shapes:
        for seed in range(n_seeds):
            diff = check_m_invariance(M_full, M_tail, 3584, "bfloat16", seed, norm_fn=triton_fn)
            assert diff == 0.0, f"Triton RMSNorm NOT M-invariant: shape=({M_full},{M_tail}) seed={seed} diff={diff}"
        diff_f32 = check_m_invariance(M_full, M_tail, 3584, "float32", 0, norm_fn=triton_fn)
        assert diff_f32 == 0.0, f"Triton RMSNorm NOT M-invariant (f32): shape=({M_full},{M_tail}) diff={diff_f32}"
        print(f"  ({M_full}, {M_tail}): bf16 10 seeds + f32 all PASS")
    print("\n  Triton RMSNorm is M-invariant across all tested shapes ✅")


if __name__ == "__main__":
    import pytest

    pytest.main(["-sv", __file__])
