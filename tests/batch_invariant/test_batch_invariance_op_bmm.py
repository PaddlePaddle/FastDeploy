# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py
#
# Test suite for batch-invariant bmm (batched matrix multiply).
#
# Purpose:
#   Verify that the batch-invariant bmm implementation (Triton-based) produces
#   deterministic and numerically acceptable results, ensuring inference output
#   does not change when requests are batched together.
#
# Test items:
#   1. test_batch_invariance
#      - Core property: bmm(A, B)[i] must be BIT-EXACT regardless of batch size.
#      - Compares batch=1 result vs slicing from a larger-batch bmm.
#      - Covers float32, float16, bfloat16 with various shapes (power-of-2 and
#        non-power-of-2 dimensions), repeated across multiple iterations.
#
#   2. test_numerical_correctness
#      - Ensures the Triton kernel output is numerically close to a numpy float64
#        reference, using np.allclose-style tolerance (atol + rtol * |ref|).
#      - Accounts for TF32 tensor-core rounding in float32 and reduced precision
#        in float16/bfloat16.
#
#   3. test_special_inputs
#      - Zero matrix: A @ 0 must produce exact zeros.
#      - Identity matrix: A @ I must approximate A within TF32 tolerance.
#      - Per-element batch consistency: each batch element computed individually
#        must match the corresponding slice from the batched computation (bit-exact).

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)

# Tolerance for numerical correctness (bmm result vs numpy reference).
# Triton tl.dot uses tensor cores which operate in TF32 for float32 inputs
# (10-bit mantissa, eps ≈ 1e-3). Combined with K-dim accumulation, absolute
# error can be significant for outputs near zero, so we need both rtol and atol.
_RTOL = {
    paddle.float32: 2e-3,
    paddle.float16: 2e-2,
    paddle.bfloat16: 2e-2,
}
_ATOL = {
    paddle.float32: 5e-2,  # TF32 eps * K, covers K up to ~256 with |input|≤1
    paddle.float16: 1e-1,
    paddle.bfloat16: 1e-1,
}

# Input value range per dtype to avoid overflow in bmm dot products.
# float16 max=65504, so with K=256: need |val| < sqrt(65504/256) ~ 16
_INPUT_RANGE = {
    paddle.float32: 100,
    paddle.float16: 10,
    paddle.bfloat16: 100,
}


class TestBatchInvariantForBMM(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def _check_batch_invariance(self, Batch, M, K, N, dtype):
        """
        Check that bmm produces identical results regardless of batch size.
        Compare: bmm with batch=1 vs slicing from a larger batch bmm.
        """
        r = _INPUT_RANGE[dtype]
        a = paddle.linspace(-r, r, Batch * M * K, dtype=dtype).reshape(Batch, M, K)
        b = paddle.linspace(-r, r, Batch * K * N, dtype=dtype).reshape(Batch, K, N)

        out1 = paddle.bmm(a[:1], b[:1])
        out2 = paddle.bmm(a, b)[:1]

        diff = (out1 - out2).abs().max()
        return diff.item() == 0, diff

    def _run_iters(self, iters=10, assert_equal=False, shapes=None):
        if shapes is None:
            shapes = [
                (32, 64, 128, 64),  # default
                (16, 33, 97, 51),  # non-power-of-2
                (2, 128, 256, 128),  # small batch, large dims
            ]
        for dtype in [paddle.float32, paddle.float16, paddle.bfloat16]:
            for Batch, M, K, N in shapes:
                is_invariant = True
                difflist = []
                for i in range(iters):
                    isd, df = self._check_batch_invariance(Batch, M, K, N, dtype)
                    is_invariant = is_invariant and isd
                    difflist.append(df)
                print(
                    f"Batch invariant: {is_invariant} max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} "
                    f"for shape=({Batch},{M},{K},{N}) {dtype} in {iters} iters"
                )
                if assert_equal:
                    assert (
                        max(difflist) == 0
                    ), f"Batch invariance failed for shape=({Batch},{M},{K},{N}) {dtype}: max diff={max(difflist)}"

    def _check_correctness(self, Batch, M, K, N, dtype):
        """
        Verify that the batch-invariant bmm produces numerically correct results.
        Reference: numpy float64 matmul on the SAME truncated inputs the GPU sees.
        This isolates computation error from input quantization error.
        """
        rng = np.random.RandomState(42)
        a_fp64 = rng.uniform(-1, 1, (Batch, M, K))
        b_fp64 = rng.uniform(-1, 1, (Batch, K, N))

        # Simulate the same input truncation path the GPU takes: fp64 -> fp32 -> dtype
        a_fp32 = a_fp64.astype(np.float32)
        b_fp32 = b_fp64.astype(np.float32)
        if dtype == paddle.float16:
            a_trunc = a_fp32.astype(np.float16).astype(np.float64)
            b_trunc = b_fp32.astype(np.float16).astype(np.float64)
        else:
            # float32 and bfloat16 (numpy has no bf16; kernel accumulates in fp32)
            a_trunc = a_fp32.astype(np.float64)
            b_trunc = b_fp32.astype(np.float64)

        # Ground truth: same truncated inputs, computed in float64
        ref = np.matmul(a_trunc, b_trunc)

        # GPU result
        a_pd = paddle.to_tensor(a_fp32).cast(dtype)
        b_pd = paddle.to_tensor(b_fp32).cast(dtype)
        out = paddle.bmm(a_pd, b_pd)
        out_np = out.cast(paddle.float32).numpy().astype(np.float64)

        rtol = _RTOL[dtype]
        atol = _ATOL[dtype]
        # np.allclose style: |out - ref| <= atol + rtol * |ref|
        passed = bool(np.all(np.abs(out_np - ref) <= atol + rtol * np.abs(ref)))
        max_abs = float(np.abs(out_np - ref).max())
        return passed, max_abs, rtol, atol

    def test_batch_invariance(self):
        """Batch-invariant mode must produce bit-exact results across batch sizes."""
        print("Standard Paddle:")
        with set_batch_invariant_mode(False):
            self._run_iters(assert_equal=False)
        print("\nBatch-Invariant Mode:")
        with set_batch_invariant_mode(True):
            self._run_iters(assert_equal=True)

    def test_numerical_correctness(self):
        """Batch-invariant bmm must produce numerically correct results vs numpy reference."""
        shapes = [
            (4, 64, 128, 64),
            (2, 33, 97, 51),
            (1, 128, 256, 128),
        ]
        for dtype in [paddle.float32, paddle.float16, paddle.bfloat16]:
            for Batch, M, K, N in shapes:
                with set_batch_invariant_mode(True):
                    passed, max_abs, rtol, atol = self._check_correctness(Batch, M, K, N, dtype)
                print(
                    f"Correctness: passed={passed} max_abs_err={max_abs:.6e} rtol={rtol:.0e} atol={atol:.0e} "
                    f"shape=({Batch},{M},{K},{N}) {dtype}"
                )
                self.assertTrue(
                    passed,
                    f"Numerical correctness failed: max_abs_err={max_abs:.6e} "
                    f"for shape=({Batch},{M},{K},{N}) {dtype} (rtol={rtol}, atol={atol})",
                )

    def test_unsupported_dtype_raises(self):
        """bmm_persistent must raise ValueError for unsupported dtypes (e.g., int32)."""
        with set_batch_invariant_mode(True):
            a = paddle.randint(0, 10, [2, 16, 32], dtype=paddle.int32)
            b = paddle.randint(0, 10, [2, 32, 16], dtype=paddle.int32)
            with self.assertRaises(ValueError) as ctx:
                paddle.bmm(a, b)
            self.assertIn("Unsupported dtype", str(ctx.exception))

    def test_special_inputs(self):
        """Batch-invariant bmm must handle special input patterns correctly."""
        with set_batch_invariant_mode(True):
            # Zero matrix: A @ 0 = 0 (must be exact, no accumulation error possible)
            a = paddle.randn([2, 16, 32], dtype=paddle.float32)
            b = paddle.zeros([2, 32, 16], dtype=paddle.float32)
            out = paddle.bmm(a, b)
            self.assertTrue((out == 0).all().item(), "bmm with zero matrix B should produce all zeros")

            # Identity: A @ I ≈ A (tensor core TF32 rounding, K-dim accumulation)
            K = 64
            a = paddle.randn([2, 16, K], dtype=paddle.float32)
            b = paddle.eye(K, dtype=paddle.float32).unsqueeze(0).expand([2, K, K])
            out = paddle.bmm(a, b)
            diff = (out - a).abs().max().item()
            self.assertLessEqual(
                diff,
                _ATOL[paddle.float32],
                f"bmm with identity matrix: max diff={diff} exceeds tolerance",
            )

            # Per-element batch consistency: (A @ B)[i] == bmm(A[i:i+1], B[i:i+1])
            a = paddle.randn([4, 32, 64], dtype=paddle.bfloat16)
            b = paddle.randn([4, 64, 32], dtype=paddle.bfloat16)
            batched_out = paddle.bmm(a, b)
            for i in range(4):
                single_out = paddle.bmm(a[i : i + 1], b[i : i + 1])
                diff = (batched_out[i : i + 1] - single_out).abs().max().item()
                self.assertEqual(diff, 0.0, f"Batch element {i} mismatch: max diff={diff}")


if __name__ == "__main__":
    unittest.main()
