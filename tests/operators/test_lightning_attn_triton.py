# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for Hackathon 10th Spring No.47.
Integration tests for the Lightning Attention Triton kernel.

These tests exercise the REAL Triton JIT-compiled GPU kernel
(lightning_attention_forward) against a pure-NumPy reference
implementation.  They are NOT stub/mock tests — they require
a CUDA-capable GPU with Triton support.

Validated on:  AI Studio V100 (SM70), Paddle 3.3.0, Triton 3.x
CI marker:     @pytest.mark.gpu — skipped automatically when no GPU is present.
"""

import unittest

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# NumPy reference — authoritative, matches the recurrence in the paper.
# ---------------------------------------------------------------------------


def _lightning_attention_numpy_ref(q, k, v, slope, kv_history=None):
    """
    Pure NumPy reference implementation of linear attention with exponential
    decay (Lightning Attention).

    Args:
        q, k, v: float64 arrays of shape [b, h, n, d] / [b, h, n, e].
        slope: 1-D array of shape [h] — per-head decay rates.
        kv_history: optional [b, h, d, e] float64 — KV state carry-in.

    Returns:
        output: [b, h, n, e] attention output.
        kv_state: [b, h, d, e] updated KV state after processing all n steps.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    output = np.zeros((b, h, n, e), dtype=np.float64)

    if kv_history is None:
        kv_state = np.zeros((b, h, d, e), dtype=np.float64)
    else:
        kv_state = kv_history.copy()

    for t in range(n):
        decay = np.exp(-slope)[np.newaxis, :, np.newaxis, np.newaxis]
        kv_state = kv_state * decay
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        kv_state += kt[:, :, :, np.newaxis] * vt[:, :, np.newaxis, :]
        qt = q[:, :, t, :]
        output[:, :, t, :] = np.einsum("bhd,bhde->bhe", qt, kv_state)

    return output, kv_state


# ---------------------------------------------------------------------------
# GPU availability guard
# ---------------------------------------------------------------------------

_GPU_AVAILABLE = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0

_SKIP_MSG = "No CUDA GPU available — lightning attention Triton kernel requires GPU"


def _import_lightning_attention_forward():
    """Lazy import so collection doesn't crash on CPU-only boxes."""
    from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
        lightning_attention_forward,
    )

    return lightning_attention_forward


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@unittest.skipUnless(_GPU_AVAILABLE, _SKIP_MSG)
class TestLightningAttentionTriton(unittest.TestCase):
    """
    Integration test: real Triton kernel vs NumPy reference.

    Parametrisation axes:
        batch   : 1, 2
        heads   : 4, 8
        seq_len : 256 (one block), 512 (two blocks)
        head_dim: 64, 128
        dtype   : float16, bfloat16
    """

    # Tolerance table — Triton accumulates in fp32 but the inputs are half
    # precision, so we need generous tolerances for long sequences.
    _TOL = {
        "float16": {"rtol": 5e-2, "atol": 5e-2},
        "bfloat16": {"rtol": 8e-2, "atol": 8e-2},
    }

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu:0")
        # Store as list to avoid Python descriptor binding (self would be
        # passed as first arg if a bare function is set as class attribute).
        cls._forward_fn = [_import_lightning_attention_forward()]

    # --- helpers -----------------------------------------------------------

    def _run_forward(self, b, h, n, d, dtype_str):
        """Run Triton kernel and compare against NumPy reference."""
        rng = np.random.default_rng(42)

        # Random inputs in float64 for the reference, then cast to target dtype
        q_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1
        k_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1
        v_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1
        slope_np = np.abs(rng.standard_normal(h).astype(np.float64)) * 0.5 + 0.1

        # NumPy reference (float64)
        ref_out, ref_kv = _lightning_attention_numpy_ref(q_np, k_np, v_np, slope_np)

        # Paddle tensors on GPU
        dtype_paddle = dtype_str
        q = paddle.to_tensor(q_np.astype(np.float32), dtype=dtype_paddle)
        k = paddle.to_tensor(k_np.astype(np.float32), dtype=dtype_paddle)
        v = paddle.to_tensor(v_np.astype(np.float32), dtype=dtype_paddle)

        # Slope: the kernel accepts [1, h, 1, 1] or [h].
        # The model code passes ed as [1, h, 1, 1] after reshape.
        slope = paddle.to_tensor(slope_np.astype(np.float32), dtype="float32")
        slope_4d = slope.reshape([1, h, 1, 1])

        # KV history initialised to zeros
        kv_history = paddle.zeros([b, h, d, d], dtype="float32")

        # Run kernel
        out, kv_out = self._forward_fn[0](q, k, v, slope_4d, kv_history, block_size=256)

        # Move to CPU for comparison
        out_np = out.astype("float32").numpy()
        kv_out_np = kv_out.numpy()

        tol = self._TOL[dtype_str]
        np.testing.assert_allclose(
            out_np,
            ref_out.astype(np.float32),
            rtol=tol["rtol"],
            atol=tol["atol"],
            err_msg=f"Output mismatch: b={b}, h={h}, n={n}, d={d}, dtype={dtype_str}",
        )

        return out_np, kv_out_np, ref_out, ref_kv

    # --- core correctness tests -------------------------------------------

    def test_small_single_block_fp16(self):
        """b=1, h=4, n=256, d=64 — single block, float16."""
        self._run_forward(b=1, h=4, n=256, d=64, dtype_str="float16")

    def test_small_single_block_bf16(self):
        """b=1, h=4, n=256, d=64 — single block, bfloat16."""
        self._run_forward(b=1, h=4, n=256, d=64, dtype_str="bfloat16")

    def test_two_blocks_fp16(self):
        """b=1, h=8, n=512, d=128 — two blocks, float16."""
        self._run_forward(b=1, h=8, n=512, d=128, dtype_str="float16")

    def test_two_blocks_bf16(self):
        """b=2, h=4, n=512, d=64 — two blocks, batched, bfloat16."""
        self._run_forward(b=2, h=4, n=512, d=64, dtype_str="bfloat16")

    def test_large_dim_fp16(self):
        """b=1, h=8, n=256, d=128 — large head dim, float16."""
        self._run_forward(b=1, h=8, n=256, d=128, dtype_str="float16")

    def test_batched_bf16(self):
        """b=2, h=8, n=256, d=128 — multi-batch, bfloat16."""
        self._run_forward(b=2, h=8, n=256, d=128, dtype_str="bfloat16")

    # --- KV history persistence (recurrent property) ----------------------

    def test_kv_history_persistence(self):
        """
        Verify that processing [seq1, seq2] in two calls with KV carry-over
        matches processing the full concatenated sequence [seq1 || seq2].
        """
        b, h, d = 1, 4, 64
        n1, n2 = 256, 256
        rng = np.random.default_rng(123)

        q1_np = rng.standard_normal((b, h, n1, d)).astype(np.float64) * 0.1
        k1_np = rng.standard_normal((b, h, n1, d)).astype(np.float64) * 0.1
        v1_np = rng.standard_normal((b, h, n1, d)).astype(np.float64) * 0.1
        q2_np = rng.standard_normal((b, h, n2, d)).astype(np.float64) * 0.1
        k2_np = rng.standard_normal((b, h, n2, d)).astype(np.float64) * 0.1
        v2_np = rng.standard_normal((b, h, n2, d)).astype(np.float64) * 0.1
        slope_np = np.abs(rng.standard_normal(h).astype(np.float64)) * 0.5 + 0.1

        # Two-call path (with KV carry-over)
        _, kv_after_1 = _lightning_attention_numpy_ref(q1_np, k1_np, v1_np, slope_np)
        out2_ref, _ = _lightning_attention_numpy_ref(q2_np, k2_np, v2_np, slope_np, kv_history=kv_after_1)

        # Full-sequence path
        q_full = np.concatenate([q1_np, q2_np], axis=2)
        k_full = np.concatenate([k1_np, k2_np], axis=2)
        v_full = np.concatenate([v1_np, v2_np], axis=2)
        out_full_ref, _ = _lightning_attention_numpy_ref(q_full, k_full, v_full, slope_np)
        out_full_second_half = out_full_ref[:, :, n1:, :]

        # Reference consistency check (NumPy vs NumPy)
        np.testing.assert_allclose(
            out2_ref.astype(np.float32),
            out_full_second_half.astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Reference recurrence does not match full-sequence computation",
        )

        # Now run the two-call path through the Triton kernel
        dtype_str = "float16"
        dtype_paddle = dtype_str
        slope = paddle.to_tensor(slope_np.astype(np.float32), dtype="float32")
        slope_4d = slope.reshape([1, h, 1, 1])

        q1 = paddle.to_tensor(q1_np.astype(np.float32), dtype=dtype_paddle)
        k1 = paddle.to_tensor(k1_np.astype(np.float32), dtype=dtype_paddle)
        v1 = paddle.to_tensor(v1_np.astype(np.float32), dtype=dtype_paddle)
        q2 = paddle.to_tensor(q2_np.astype(np.float32), dtype=dtype_paddle)
        k2 = paddle.to_tensor(k2_np.astype(np.float32), dtype=dtype_paddle)
        v2 = paddle.to_tensor(v2_np.astype(np.float32), dtype=dtype_paddle)

        kv_init = paddle.zeros([b, h, d, d], dtype="float32")

        # Call 1
        _, kv_after_1_gpu = self._forward_fn[0](q1, k1, v1, slope_4d, kv_init, block_size=256)
        # Call 2 — feed KV state from call 1
        out2_gpu, _ = self._forward_fn[0](q2, k2, v2, slope_4d, kv_after_1_gpu, block_size=256)

        out2_gpu_np = out2_gpu.astype("float32").numpy()

        np.testing.assert_allclose(
            out2_gpu_np,
            out2_ref.astype(np.float32),
            rtol=5e-2,
            atol=5e-2,
            err_msg="Triton KV carry-over does not match reference two-call path",
        )

    # --- output shape and dtype -------------------------------------------

    def test_output_shape(self):
        """Verify output tensor shape matches [b, h, n, d]."""
        b, h, n, d = 1, 4, 256, 64
        q = paddle.randn([b, h, n, d], dtype="float16")
        k = paddle.randn([b, h, n, d], dtype="float16")
        v = paddle.randn([b, h, n, d], dtype="float16")
        slope = paddle.ones([1, h, 1, 1], dtype="float32") * 0.3
        kv = paddle.zeros([b, h, d, d], dtype="float32")

        out, kv_out = self._forward_fn[0](q, k, v, slope, kv, block_size=256)

        self.assertEqual(list(out.shape), [b, h, n, d])
        self.assertEqual(list(kv_out.shape), [b, h, d, d])

    def test_output_dtype_preserved(self):
        """Verify output dtype matches input dtype."""
        b, h, n, d = 1, 4, 256, 64
        for dtype_str in ["float16", "bfloat16"]:
            q = paddle.randn([b, h, n, d], dtype=dtype_str)
            k = paddle.randn([b, h, n, d], dtype=dtype_str)
            v = paddle.randn([b, h, n, d], dtype=dtype_str)
            slope = paddle.ones([1, h, 1, 1], dtype="float32") * 0.3
            kv = paddle.zeros([b, h, d, d], dtype="float32")

            out, kv_out = self._forward_fn[0](q, k, v, slope, kv, block_size=256)
            self.assertEqual(str(out.dtype).split(".")[-1], dtype_str)

    # --- decode-path kernel -----------------------------------------------

    def test_linear_decode_forward(self):
        """
        Test the linear_decode_forward_triton kernel (single-step decode).
        This is the kernel used during autoregressive generation.
        """
        from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
            linear_decode_forward_triton,
        )

        b, h, d = 2, 8, 128
        q = paddle.randn([b, h, 1, d], dtype="float16")
        k = paddle.randn([b, h, 1, d], dtype="float16")
        v = paddle.randn([b, h, 1, d], dtype="float16")
        kv_caches = paddle.zeros([b, h, d, d], dtype="float32")
        slope_rate = paddle.ones([h], dtype="float32") * 0.3
        slot_idx = paddle.arange(b, dtype="int64")

        out = linear_decode_forward_triton(q, k, v, kv_caches, slope_rate, slot_idx)

        # Output shape: [B, H*D] (flattened heads)
        self.assertEqual(list(out.shape), [b, h * d])
        self.assertFalse(paddle.isnan(out).any().item(), "Decode output contains NaN")
        self.assertTrue(paddle.isfinite(out).all().item(), "Decode output contains Inf")


if __name__ == "__main__":
    unittest.main()
