# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)


class TestBatchInvariantForMean(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def test_batch_invariance(self, B: int = 2048, D: int = 4096, dtype=paddle.float32):
        a = paddle.linspace(-100, 100, B * D, dtype=dtype).reshape(B, D)

        # Method 1: Mean reduction over last axis (batch size 1)
        out1 = paddle.mean(a[:1], axis=-1)

        # Method 2: Mean reduction over last axis (full batch)
        out2 = paddle.mean(a, axis=-1)[:1]

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
    Batch Deterministic: False run-to-run max/min/diff 7.62939453125e-06/7.62939453125e-06/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations

    Batch-Invariant Mode:
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations
    """
