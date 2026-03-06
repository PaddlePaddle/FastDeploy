"""
Direct test: does QKV projection GEMM produce bit-identical results
for the same tokens when M (total token count) differs?

Cache miss: GEMM(825, K) @ W(K, N) → take last 57 rows
Cache hit:  GEMM(57, K)  @ W(K, N) → all 57 rows

If the last 57 rows differ, that's the root cause of prefix caching non-determinism.
"""

import os
import sys

import paddle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def test_gemm_m_invariance():
    """Test whether matmul results are invariant to M dimension for overlapping rows."""
    paddle.seed(42)

    # Typical Qwen2-7B dimensions: hidden=3584, qkv_out=4608
    K = 3584
    N = 4608
    M_full = 825  # cache miss
    M_partial = 57  # cache hit (only new tokens)

    W = paddle.randn([K, N], dtype="bfloat16")
    full_input = paddle.randn([M_full, K], dtype="bfloat16")
    partial_input = full_input[-M_partial:].clone()  # same data

    print("=" * 70)
    print("Test: GEMM M-dimension invariance (QKV projection)")
    print(f"  W: ({K}, {N}), full: ({M_full}, {K}), partial: ({M_partial}, {K})")
    print("=" * 70)

    # --- Test 1: Default paddle matmul (cuBLAS) ---
    print("\n[1] Default paddle.matmul (cuBLAS):")
    full_out = paddle.matmul(full_input, W)
    partial_out = paddle.matmul(partial_input, W)
    overlap = full_out[-M_partial:]

    diff = (overlap.float() - partial_out.float()).abs()
    max_diff = diff.max().item()
    n_differ = (diff > 0).sum().item()
    total = diff.numel()
    print(f"  max_diff = {max_diff}")
    print(f"  differing elements: {n_differ}/{total} ({100*n_differ/total:.2f}%)")
    cublas_identical = max_diff == 0.0
    print(f"  bit-identical: {'YES' if cublas_identical else 'NO'}")

    # --- Test 2: Batch-invariant matmul (Triton persistent) ---
    print("\n[2] Batch-invariant matmul (Triton persistent kernel):")
    from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
        matmul_persistent,
    )

    full_out_bi = matmul_persistent(full_input, W)
    partial_out_bi = matmul_persistent(partial_input, W)
    overlap_bi = full_out_bi[-M_partial:]

    diff_bi = (overlap_bi.float() - partial_out_bi.float()).abs()
    max_diff_bi = diff_bi.max().item()
    n_differ_bi = (diff_bi > 0).sum().item()
    print(f"  max_diff = {max_diff_bi}")
    print(f"  differing elements: {n_differ_bi}/{total} ({100*n_differ_bi/total:.2f}%)")
    bi_identical = max_diff_bi == 0.0
    print(f"  bit-identical: {'YES' if bi_identical else 'NO'}")

    # --- Test 3: Check if F.linear / paddle.nn.Linear uses matmul or linear op ---
    print("\n[3] paddle.nn.Linear path:")
    linear = paddle.nn.Linear(K, N, bias_attr=False)
    linear.weight.set_value(W.cast("float32"))  # Linear default float32, cast for set_value
    linear = linear.to(dtype="bfloat16")

    full_out_nn = linear(full_input)
    partial_out_nn = linear(partial_input)
    overlap_nn = full_out_nn[-M_partial:]

    diff_nn = (overlap_nn.float() - partial_out_nn.float()).abs()
    max_diff_nn = diff_nn.max().item()
    n_differ_nn = (diff_nn > 0).sum().item()
    print(f"  max_diff = {max_diff_nn}")
    print(f"  differing elements: {n_differ_nn}/{total} ({100*n_differ_nn/total:.2f}%)")
    nn_identical = max_diff_nn == 0.0
    print(f"  bit-identical: {'YES' if nn_identical else 'NO'}")

    # --- Test 4: paddle.nn.Linear WITH batch-invariant mode enabled ---
    print("\n[4] paddle.nn.Linear + batch-invariant mode enabled:")
    from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()

    full_out_patched = linear(full_input)
    partial_out_patched = linear(partial_input)
    overlap_patched = full_out_patched[-M_partial:]

    diff_patched = (overlap_patched.float() - partial_out_patched.float()).abs()
    max_diff_patched = diff_patched.max().item()
    n_differ_patched = (diff_patched > 0).sum().item()
    print(f"  max_diff = {max_diff_patched}")
    print(f"  differing elements: {n_differ_patched}/{total} ({100*n_differ_patched/total:.2f}%)")
    patched_identical = max_diff_patched == 0.0
    print(f"  bit-identical: {'YES' if patched_identical else 'NO'}")

    disable_batch_invariant_mode()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  [1] cuBLAS direct:               {'PASS' if cublas_identical else 'FAIL'}")
    print(f"  [2] Triton persistent (direct):   {'PASS' if bi_identical else 'FAIL'}")
    print(f"  [3] nn.Linear (no patch):         {'PASS' if nn_identical else 'FAIL'}")
    print(f"  [4] nn.Linear (batch-invariant):  {'PASS' if patched_identical else 'FAIL'}")
    print("=" * 70)

    if not cublas_identical and bi_identical and not patched_identical:
        print("\nDIAGNOSIS: Triton persistent matmul IS batch-invariant,")
        print("  but nn.Linear bypasses it even with batch-invariant mode on!")
        print("  → nn.Linear calls _C_ops.linear, not _C_ops.matmul")
        print("  → FIX: patch _C_ops.linear to also use Triton persistent matmul")


if __name__ == "__main__":
    test_gemm_m_invariance()
