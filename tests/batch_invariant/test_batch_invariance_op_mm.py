# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)
from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    matmul_persistent,
)

# Real-world shapes for M-invariance testing: (M_full, M_tail, K, N)
QKV_SHAPES = [
    (825, 57, 3584, 4608),  # Qwen2-7B real case
    (1024, 128, 4096, 4096),  # power-of-2
    (512, 1, 3584, 4608),  # single-token tail
]
N_SEEDS = 5


class TestBatchInvariantForMM(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def test_batch_invariance(self, B: int = 2048, D: int = 4096, dtype=paddle.float32):
        a = paddle.linspace(-100, 100, B * D, dtype=dtype).reshape(B, D)
        b = paddle.linspace(-100, 100, D * D, dtype=dtype).reshape(D, D)

        # Method 1: Matrix-vector multiplication (batch size 1)
        out1 = paddle.mm(a[:1], b)

        # Method 2: Matrix-matrix multiplication, then slice (full batch)
        out2 = paddle.mm(a, b)[:1]

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

    @staticmethod
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

    def test_triton_persistent_is_m_invariant(self):
        """Triton persistent matmul must be bit-identical regardless of M."""
        for M_full, M_tail, K, N in QKV_SHAPES:
            for seed in range(N_SEEDS):
                diff = self._check_m_invariance(matmul_persistent, M_full, M_tail, K, N, seed)
                self.assertEqual(
                    diff,
                    0.0,
                    f"matmul_persistent NOT M-invariant: shape=({M_full},{M_tail},{K},{N}) seed={seed} diff={diff}",
                )


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
