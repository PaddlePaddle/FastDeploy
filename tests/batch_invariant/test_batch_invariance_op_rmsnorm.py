# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
)


def fused_rms_norm(x, weight, eps=1e-6):
    """Standard Paddle fused_rms_norm (M-non-invariant)."""
    return paddle.incubate.nn.functional.fused_rms_norm(x, weight, None, eps, 1)[0]


class TestBatchInvariantForRMSNorm(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def test_batch_invariance(
        self, B: int = 825, M_tail: int = 57, D: int = 3584, dtype=paddle.bfloat16, norm_fn=None
    ):
        """Test M-invariance: norm(full)[-M_tail:] == norm(tail).

        Uses random data with different seeds to avoid bf16 false negatives.
        """
        if norm_fn is None:
            norm_fn = fused_rms_norm

        a = paddle.randn([B, D], dtype=dtype)
        w = paddle.randn([D], dtype=dtype)

        # Method 1: Normalize sub-batch only (batch size M_tail)
        part = a[-M_tail:].clone()
        out1 = norm_fn(part, w)

        # Method 2: Normalize full batch, then slice (batch size B)
        out2 = norm_fn(a, w)[-M_tail:]

        # Check if results are identical
        if dtype == paddle.bfloat16:
            diff = (out1.astype("float32") - out2.astype("float32")).abs().max()
        else:
            diff = (out1 - out2).abs().max()
        return diff.item() == 0, diff

    def run_iters(self, iters=10, ass=False, norm_fn=None):
        for dtype in [paddle.float32, paddle.bfloat16]:
            is_deterministic = True
            difflist = []
            for i in range(iters):
                paddle.seed(i)
                isd, df = self.test_batch_invariance(dtype=dtype, norm_fn=norm_fn)
                is_deterministic = is_deterministic and isd
                difflist.append(df)
            print(
                f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations"
            )
            if ass:
                assert max(difflist) == 0, f"RMSNorm not M-invariant for {dtype}: max_diff={max(difflist)}"

    def test_case(self):
        # Test with standard Paddle fused_rms_norm (expected to be M-non-invariant)
        print("Standard Paddle fused_rms_norm:")
        with set_batch_invariant_mode(False):
            self.run_iters(ass=False, norm_fn=fused_rms_norm)
        # Test with batch-invariant Triton RMSNorm (must be M-invariant)
        print("\nBatch-Invariant Mode (Triton RMSNorm):")
        with set_batch_invariant_mode(True):
            self.run_iters(ass=True, norm_fn=rms_norm_batch_invariant)

    def test_various_shapes(self):
        """Test M-invariance across different (B, M_tail) combos with Triton kernel."""
        shapes = [
            (825, 57),  # real case: Qwen2-7B prefix caching
            (1024, 128),  # power-of-2
            (2048, 1),  # single token tail
            (512, 256),  # half split
            (100, 99),  # almost equal
        ]
        print("Triton RMSNorm across various shapes:")
        for B, M_tail in shapes:
            for dtype in [paddle.float32, paddle.bfloat16]:
                for seed in range(10):
                    paddle.seed(seed)
                    isd, df = self.test_batch_invariance(
                        B=B,
                        M_tail=M_tail,
                        dtype=dtype,
                        norm_fn=rms_norm_batch_invariant,
                    )
                    assert isd, (
                        f"Triton RMSNorm NOT M-invariant: " f"shape=({B},{M_tail}) dtype={dtype} seed={seed} diff={df}"
                    )
            print(f"  ({B}, {M_tail}): f32+bf16 x 10 seeds PASS")


if __name__ == "__main__":
    unittest.main()
    """
    Standard Paddle fused_rms_norm:
    Batch Deterministic: False run-to-run max/min/diff ... for paddle.float32 in 10 iterations
    Batch Deterministic: False run-to-run max/min/diff ... for paddle.bfloat16 in 10 iterations

    Batch-Invariant Mode (Triton RMSNorm):
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.float32 in 10 iterations
    Batch Deterministic: True run-to-run max/min/diff 0.0/0.0/0.0 for paddle.bfloat16 in 10 iterations
    """
