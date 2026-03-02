# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)
from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    addmm_batch_invariant,
)


class TestBatchInvariantForAddmm(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def test_batch_invariance(self, B: int = 2048, D: int = 4096, dtype=paddle.float32):
        a = paddle.linspace(-100, 100, B * D, dtype=dtype).reshape(B, D)
        b = paddle.linspace(-100, 100, D * D, dtype=dtype).reshape(D, D)

        # Method 1: Matrix-vector multiplication and add (batch size 1)
        out1 = paddle.addmm(a[:1].squeeze(0), a[:1], b)

        # Method 2: Matrix-matrix multiplication and add, then slice (full batch)
        out2 = paddle.addmm(a[:1].squeeze(0), a, b)[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item() == 0, diff

    def run_iters(self, iters=10, ass=False):
        for dtype in [paddle.float32, paddle.bfloat16]:
            is_deterministic = True
            difflist = []
            for i in range(iters):
                isd, df = self.test_batch_invariance(dtype=dtype)
                is_deterministic = is_deterministic and isd
                difflist.append(df)
            print(
                f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations"
            )
            if ass:
                assert max(difflist) == 0

    def test_alpha_zero(self):
        """alpha == 0: result should be beta * input broadcast to [M, N]"""
        M, N, K = 32, 64, 128
        for dtype in [paddle.float32, paddle.bfloat16]:
            x = paddle.randn([M, K], dtype=dtype)
            y = paddle.randn([K, N], dtype=dtype)
            bias = paddle.randn([N], dtype=dtype)

            for beta in [0.0, 1.0, 2.5]:
                out = addmm_batch_invariant(bias, x, y, beta=beta, alpha=0.0)
                expected = (beta * bias).expand([M, N])
                # shape must be [M, N]
                assert out.shape == [M, N], f"Expected shape [{M}, {N}], got {out.shape}"
                # cast to float32 for comparison (bfloat16 not supported by isclose)
                diff = (out.cast(paddle.float32) - expected.cast(paddle.float32)).abs().max()
                assert diff.item() == 0, f"dtype={dtype}, beta={beta}, max diff={diff.item()}"

    def test_case(self):
        # Test with standard Paddle (likely to show differences)
        print("Standard Paddle:")
        with set_batch_invariant_mode(False):
            self.run_iters(ass=False)
        # Test with batch-invariant operations
        print("\nBatch-Invariant Mode:")
        with set_batch_invariant_mode(True):
            self.run_iters(ass=True)


if __name__ == "__main__":
    unittest.main()
    """
    Standard Paddle:
    Batch Deterministic: False run-to-run max/min/diff 10.7294921875/10.7294921875/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations

    Batch-Invariant Mode:
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations
    """
