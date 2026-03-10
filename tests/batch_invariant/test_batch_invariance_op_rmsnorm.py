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


def _reference_rms_norm(x, w, eps=1e-6):
    """Pure Paddle reference implementation for numerical correctness check."""
    x_f32 = x.astype("float32")
    w_f32 = w.astype("float32")
    rms = paddle.sqrt(paddle.mean(x_f32 * x_f32, axis=-1, keepdim=True) + eps)
    return (x_f32 / rms * w_f32).astype(x.dtype)


class TestBatchInvariantForRMSNorm(unittest.TestCase):
    def setUp(self):
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def _check_batch_invariance(
        self, B: int = 825, M_tail: int = 57, D: int = 3584, dtype=paddle.bfloat16, norm_fn=None, eps=1e-6
    ):
        """Check M-invariance: norm(full)[-M_tail:] == norm(tail).

        Returns (is_invariant, max_diff).
        """
        if norm_fn is None:
            norm_fn = fused_rms_norm

        a = paddle.randn([B, D], dtype=dtype)
        w = paddle.randn([D], dtype=dtype)

        # Method 1: Normalize sub-batch only (batch size M_tail)
        part = a[-M_tail:].clone()
        out1 = norm_fn(part, w, eps)

        # Method 2: Normalize full batch, then slice (batch size B)
        out2 = norm_fn(a, w, eps)[-M_tail:]

        # Check if results are identical
        if dtype == paddle.bfloat16:
            diff = (out1.astype("float32") - out2.astype("float32")).abs().max()
        else:
            diff = (out1 - out2).abs().max()
        return diff.item() == 0, diff

    def _run_iters(self, iters=10, assert_invariant=False, norm_fn=None, eps=1e-6):
        for dtype in [paddle.float32, paddle.bfloat16]:
            max_diff = 0.0
            for i in range(iters):
                paddle.seed(i)
                isd, df = self._check_batch_invariance(dtype=dtype, norm_fn=norm_fn, eps=eps)
                if df.item() > max_diff:
                    max_diff = df.item()
            if assert_invariant:
                self.assertEqual(max_diff, 0.0, f"RMSNorm not M-invariant for {dtype}: max_diff={max_diff}")

    def test_case(self):
        """Basic M-invariance: standard Paddle vs batch-invariant Triton."""
        with set_batch_invariant_mode(False):
            self._run_iters(assert_invariant=False, norm_fn=fused_rms_norm)
        with set_batch_invariant_mode(True):
            self._run_iters(assert_invariant=True, norm_fn=rms_norm_batch_invariant)

    def test_various_shapes(self):
        """Test M-invariance across different (B, M_tail) combos with Triton kernel."""
        shapes = [
            (825, 57),  # real case: Qwen2-7B prefix caching
            (1024, 128),  # power-of-2
            (2048, 1),  # single token tail
            (512, 256),  # half split
            (100, 99),  # almost equal
        ]
        for B, M_tail in shapes:
            for dtype in [paddle.float32, paddle.bfloat16]:
                for seed in range(10):
                    paddle.seed(seed)
                    isd, df = self._check_batch_invariance(
                        B=B,
                        M_tail=M_tail,
                        dtype=dtype,
                        norm_fn=rms_norm_batch_invariant,
                    )
                    self.assertTrue(
                        isd,
                        f"NOT M-invariant: shape=({B},{M_tail}) dtype={dtype} seed={seed} diff={df}",
                    )

    def test_various_hidden_dims(self):
        """Test M-invariance with D values triggering different BLOCK_SIZE=1024 paths."""
        dims = [
            1,  # degenerate: mean(x^2) == x^2
            128,  # < BLOCK_SIZE, single block with heavy masking
            1024,  # == BLOCK_SIZE, exact fit, no mask remainder
            2048,  # BLOCK_SIZE multiple, multi-block no remainder
            3584,  # non-divisible (3.5 blocks), current default
        ]
        for D in dims:
            for dtype in [paddle.float32, paddle.bfloat16]:
                for seed in range(5):
                    paddle.seed(seed)
                    isd, df = self._check_batch_invariance(
                        B=256,
                        M_tail=32,
                        D=D,
                        dtype=dtype,
                        norm_fn=rms_norm_batch_invariant,
                    )
                    self.assertTrue(
                        isd,
                        f"NOT M-invariant: D={D} dtype={dtype} seed={seed} diff={df}",
                    )

    def test_numerical_correctness(self):
        """Verify Triton kernel output matches pure-Paddle reference implementation."""
        test_configs = [
            (64, 1024, paddle.float32, 1e-5),
            (64, 1024, paddle.bfloat16, 1e-3),
            (64, 3584, paddle.float32, 1e-5),
            (64, 3584, paddle.bfloat16, 1e-2),
            (64, 128, paddle.float32, 1e-5),
        ]
        for B, D, dtype, atol in test_configs:
            paddle.seed(42)
            x = paddle.randn([B, D], dtype=dtype)
            w = paddle.randn([D], dtype=dtype)

            out_triton = rms_norm_batch_invariant(x, w)
            out_ref = _reference_rms_norm(x, w)

            diff = (out_triton.astype("float32") - out_ref.astype("float32")).abs().max().item()
            self.assertLessEqual(
                diff,
                atol,
                f"Triton vs reference mismatch: B={B} D={D} dtype={dtype} diff={diff} atol={atol}",
            )

    def test_various_eps(self):
        """Test M-invariance with different eps values."""
        eps_values = [
            1e-5,  # RMSNorm layer actual value (normalization.py)
            1e-6,  # function default
            1e-8,  # extreme small
        ]
        for eps in eps_values:
            for dtype in [paddle.float32, paddle.bfloat16]:
                paddle.seed(0)
                isd, df = self._check_batch_invariance(
                    B=256,
                    M_tail=32,
                    D=3584,
                    dtype=dtype,
                    norm_fn=rms_norm_batch_invariant,
                    eps=eps,
                )
                self.assertTrue(
                    isd,
                    f"NOT M-invariant: eps={eps} dtype={dtype} diff={df}",
                )

    def test_higher_rank_input(self):
        """Test M-invariance with 3D input [batch, seq_len, hidden_dim]."""
        B, S, D = 8, 32, 1024
        M_tail = 4  # tail batches

        for dtype in [paddle.float32, paddle.bfloat16]:
            paddle.seed(0)
            a = paddle.randn([B, S, D], dtype=dtype)
            w = paddle.randn([D], dtype=dtype)

            # Method 1: Normalize tail sub-batch only
            part = a[-M_tail:].clone()
            out1 = rms_norm_batch_invariant(part, w)

            # Method 2: Normalize full batch, then slice
            out2 = rms_norm_batch_invariant(a, w)[-M_tail:]

            if dtype == paddle.bfloat16:
                diff = (out1.astype("float32") - out2.astype("float32")).abs().max()
            else:
                diff = (out1 - out2).abs().max()
            self.assertEqual(
                diff.item(),
                0.0,
                f"3D input NOT M-invariant: dtype={dtype} diff={diff.item()}",
            )
            self.assertEqual(list(out1.shape), [M_tail, S, D], "Output shape mismatch for 3D input")

    def test_special_input_values(self):
        """Test with special input values: zeros, weight=1, negative weight."""
        D = 1024

        for dtype in [paddle.float32, paddle.bfloat16]:
            # All-zero input: rms -> sqrt(eps), should not produce NaN/Inf
            paddle.seed(0)
            x_zero = paddle.zeros([64, D], dtype=dtype)
            w = paddle.randn([D], dtype=dtype)
            out = rms_norm_batch_invariant(x_zero, w)
            self.assertFalse(paddle.isnan(out).any().item(), f"NaN in zero-input output ({dtype})")
            self.assertFalse(paddle.isinf(out).any().item(), f"Inf in zero-input output ({dtype})")

            # Weight = all ones: isolate norm logic
            paddle.seed(0)
            x = paddle.randn([64, D], dtype=dtype)
            w_ones = paddle.ones([D], dtype=dtype)
            out_ones = rms_norm_batch_invariant(x, w_ones)
            ref_ones = _reference_rms_norm(x, w_ones)
            diff = (out_ones.astype("float32") - ref_ones.astype("float32")).abs().max().item()
            atol = 1e-3 if dtype == paddle.bfloat16 else 1e-6
            self.assertLessEqual(diff, atol, f"weight=1 mismatch ({dtype}): diff={diff}")

            # Negative weight: verify sign correctness
            paddle.seed(0)
            x = paddle.randn([64, D], dtype=dtype)
            w_neg = -paddle.ones([D], dtype=dtype)
            out_neg = rms_norm_batch_invariant(x, w_neg)
            out_pos = rms_norm_batch_invariant(x, -w_neg)
            diff_sign = (out_neg.astype("float32") + out_pos.astype("float32")).abs().max().item()
            self.assertLessEqual(diff_sign, 1e-6, f"Negative weight sign error ({dtype}): diff={diff_sign}")

    def test_boundary_batch_sizes(self):
        """Test M-invariance at boundary batch sizes."""
        boundary_cases = [
            (128, 128),  # M_tail == B: tail is entire batch
            (1, 1),  # minimal batch
        ]
        for B, M_tail in boundary_cases:
            for dtype in [paddle.float32, paddle.bfloat16]:
                paddle.seed(0)
                isd, df = self._check_batch_invariance(
                    B=B,
                    M_tail=M_tail,
                    D=3584,
                    dtype=dtype,
                    norm_fn=rms_norm_batch_invariant,
                )
                self.assertTrue(
                    isd,
                    f"NOT M-invariant: B={B} M_tail={M_tail} dtype={dtype} diff={df}",
                )

    def test_run_to_run_determinism(self):
        """Same input executed twice must produce bitwise identical output."""
        for dtype in [paddle.float32, paddle.bfloat16]:
            paddle.seed(42)
            x = paddle.randn([256, 3584], dtype=dtype)
            w = paddle.randn([3584], dtype=dtype)

            out1 = rms_norm_batch_invariant(x, w)
            out2 = rms_norm_batch_invariant(x, w)

            diff = (out1.astype("float32") - out2.astype("float32")).abs().max().item()
            self.assertEqual(diff, 0.0, f"Run-to-run non-determinism for {dtype}: diff={diff}")


if __name__ == "__main__":
    unittest.main()
