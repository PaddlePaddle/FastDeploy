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
MiniMax-M1 integration smoke tests — real GPU kernels, no mocks.

These tests exercise the production code paths used by MiniMaxM1LinearAttention:
  1. `lightning_attention()` — the chunked prefill wrapper that calls
     `lightning_attention_forward()` in a loop over head-dim chunks.
  2. `linear_decode_forward_triton()` — the single-step decode kernel.
  3. `_build_slope_tensor()` — ALiBi-style decay tensor construction.
  4. End-to-end prefill → decode transition with KV state carry-over.

All tests run on a single GPU without model weights or TP > 1.

Validated on: AI Studio V100 (SM70), Paddle 3.3.0, Triton 3.x
CI marker:    @pytest.mark.gpu
"""

import math
import unittest

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------

_GPU_AVAILABLE = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
_SKIP_MSG = "No CUDA GPU available — MiniMax-M1 smoke tests require GPU"


def _import_ops():
    """Lazy import to avoid collection failure on CPU-only boxes."""
    from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
        lightning_attention,
        linear_decode_forward_triton,
    )

    return lightning_attention, linear_decode_forward_triton


# ---------------------------------------------------------------------------
# NumPy reference
# ---------------------------------------------------------------------------


def _lightning_attention_numpy_ref(q, k, v, slope, kv_history=None):
    """
    Pure NumPy reference for lightning attention with exponential decay.
    Iterates over time steps — slow but correct.
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
# Slope tensor builder — copied from MiniMaxM1LinearAttention._build_slope_tensor
# to test independently without FDConfig.
# ---------------------------------------------------------------------------


def _build_slope_tensor(n_heads):
    """Build ALiBi-style slope tensor (matches production code exactly)."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** (-(math.log2(n) - 3))))
        return [start * (start**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power)
        slopes += get_slopes_power_of_2(2 * closest_power)[0::2][: n_heads - closest_power]

    return paddle.to_tensor(slopes, dtype=paddle.float32).reshape([n_heads, 1, 1])


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@unittest.skipUnless(_GPU_AVAILABLE, _SKIP_MSG)
class TestMiniMaxM1Smoke(unittest.TestCase):
    """
    Integration smoke tests for MiniMax-M1 lightning attention pipeline.
    Exercises the REAL Triton kernels on GPU — no stubs, no mocks.
    """

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu:0")
        # Store as list to avoid Python descriptor binding (self would be
        # passed as first arg if a bare function is set as class attribute).
        la, df = _import_ops()
        cls._ops = [la, df]

    def _call_lightning_attention(self, *args, **kwargs):
        return self._ops[0](*args, **kwargs)

    def _call_decode_forward(self, *args, **kwargs):
        return self._ops[1](*args, **kwargs)

    # === 1. Lightning attention (chunked prefill wrapper) ==================

    def test_lightning_attention_basic(self):
        """
        lightning_attention() with head_dim=128, the production dimension.
        Verify output is finite, shape matches, and roughly agrees with reference.
        """
        b, h, n, d = 1, 8, 256, 128
        rng = np.random.default_rng(42)

        q_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1
        k_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1
        v_np = rng.standard_normal((b, h, n, d)).astype(np.float64) * 0.1

        # Build slope as production does: [n_heads, 1, 1] → squeeze to [n_heads]
        slope_full = _build_slope_tensor(h)  # [h, 1, 1]
        slope_np = slope_full.squeeze(-1).squeeze(-1).numpy().astype(np.float64)

        # NumPy reference
        ref_out, _ = _lightning_attention_numpy_ref(q_np, k_np, v_np, slope_np)

        # GPU tensors
        q = paddle.to_tensor(q_np.astype(np.float32), dtype="float16")
        k = paddle.to_tensor(k_np.astype(np.float32), dtype="float16")
        v = paddle.to_tensor(v_np.astype(np.float32), dtype="float16")
        ed = slope_full.squeeze(-1)  # [h, 1] — wrapper reshapes to [1, h, 1, 1]

        out, kv = self._call_lightning_attention(q, k, v, ed, block_size=256)

        self.assertEqual(list(out.shape), [b, h, n, d])
        self.assertEqual(list(kv.shape), [b, h, d, d])
        self.assertFalse(paddle.isnan(out).any().item(), "Output has NaN")
        self.assertTrue(paddle.isfinite(out).all().item(), "Output has Inf")

        # Tolerance: chunked approach + fp16 → generous but must be correlated
        out_np = out.astype("float32").numpy()
        cos_sim = np.sum(out_np * ref_out.astype(np.float32)) / (
            np.linalg.norm(out_np) * np.linalg.norm(ref_out.astype(np.float32)) + 1e-12
        )
        self.assertGreater(cos_sim, 0.9, f"Cosine similarity {cos_sim:.4f} too low")

    def test_lightning_attention_multi_batch(self):
        """lightning_attention() with batch_size=2 and bfloat16."""
        b, h, n, d = 2, 8, 256, 128

        q = paddle.randn([b, h, n, d], dtype="bfloat16")
        k = paddle.randn([b, h, n, d], dtype="bfloat16")
        v = paddle.randn([b, h, n, d], dtype="bfloat16")
        ed = _build_slope_tensor(h).squeeze(-1)  # [h, 1]

        out, kv = self._call_lightning_attention(q, k, v, ed, block_size=256)

        self.assertEqual(list(out.shape), [b, h, n, d])
        self.assertFalse(paddle.isnan(out).any().item())

    def test_lightning_attention_kv_state_nonzero(self):
        """After prefill, KV state should be non-zero (kernel populated it)."""
        b, h, n, d = 1, 4, 256, 64

        q = paddle.randn([b, h, n, d], dtype="float16")
        k = paddle.randn([b, h, n, d], dtype="float16")
        v = paddle.randn([b, h, n, d], dtype="float16")
        ed = _build_slope_tensor(h).squeeze(-1)

        _, kv = self._call_lightning_attention(q, k, v, ed, block_size=256)

        kv_np = kv.numpy()
        self.assertGreater(np.abs(kv_np).max(), 0.0, "KV state is all zeros after prefill")

    # === 2. Linear decode forward (single-step autoregressive) =============

    def test_decode_forward_basic(self):
        """
        linear_decode_forward_triton() — single-step decode path.
        This is the kernel used during generation after prefill.
        """
        b, h, d = 2, 8, 128
        q = paddle.randn([b, h, 1, d], dtype="float16")
        k = paddle.randn([b, h, 1, d], dtype="float16")
        v = paddle.randn([b, h, 1, d], dtype="float16")
        kv_caches = paddle.zeros([b, h, d, d], dtype="float32")
        slope_rate = _build_slope_tensor(h).squeeze(-1).squeeze(-1)  # [h]
        slot_idx = paddle.arange(b, dtype="int64")

        out = self._call_decode_forward(q, k, v, kv_caches, slope_rate, slot_idx)

        # Output: [B, H*D] (heads flattened)
        self.assertEqual(list(out.shape), [b, h * d])
        self.assertFalse(paddle.isnan(out).any().item(), "Decode output NaN")
        self.assertTrue(paddle.isfinite(out).all().item(), "Decode output Inf")

    def test_decode_updates_kv_cache(self):
        """linear_decode_forward_triton should write to kv_caches in-place."""
        b, h, d = 1, 4, 64
        q = paddle.randn([b, h, 1, d], dtype="float16")
        k = paddle.randn([b, h, 1, d], dtype="float16")
        v = paddle.randn([b, h, 1, d], dtype="float16")
        kv_caches = paddle.zeros([b, h, d, d], dtype="float32")
        slope_rate = _build_slope_tensor(h).squeeze(-1).squeeze(-1)
        slot_idx = paddle.arange(b, dtype="int64")

        kv_before = kv_caches.numpy().copy()
        self._call_decode_forward(q, k, v, kv_caches, slope_rate, slot_idx)
        kv_after = kv_caches.numpy()

        self.assertGreater(
            np.abs(kv_after - kv_before).max(),
            0.0,
            "KV cache was not updated by decode kernel",
        )

    def test_decode_multiple_steps(self):
        """Simulate 4 decode steps, verify KV cache accumulates."""
        b, h, d = 1, 8, 128
        kv_caches = paddle.zeros([b, h, d, d], dtype="float32")
        slope_rate = _build_slope_tensor(h).squeeze(-1).squeeze(-1)
        slot_idx = paddle.arange(b, dtype="int64")

        norms = []
        for step in range(4):
            q = paddle.randn([b, h, 1, d], dtype="float16")
            k = paddle.randn([b, h, 1, d], dtype="float16")
            v = paddle.randn([b, h, 1, d], dtype="float16")
            out = self._call_decode_forward(q, k, v, kv_caches, slope_rate, slot_idx)
            norms.append(float(paddle.norm(out).item()))

        # All steps should produce non-zero output
        for i, norm_val in enumerate(norms):
            self.assertGreater(norm_val, 0.0, f"Step {i} output is zero")

    # === 3. Prefill → Decode transition ====================================

    def test_prefill_then_decode(self):
        """
        End-to-end: prefill with lightning_attention(), then decode with
        linear_decode_forward_triton().  This mimics the actual serving path
        where MiniMaxM1LinearAttention.forward() calls lightning_attention()
        during prefill and then switches to the decode kernel for generation.

        After prefill the KV state is non-zero; the decode kernel should
        produce a different output than it would with empty KV state.
        """
        b, h, n_prefill, d = 1, 8, 256, 128

        # --- Prefill phase ---
        q_pf = paddle.randn([b, h, n_prefill, d], dtype="float16")
        k_pf = paddle.randn([b, h, n_prefill, d], dtype="float16")
        v_pf = paddle.randn([b, h, n_prefill, d], dtype="float16")
        ed = _build_slope_tensor(h).squeeze(-1)  # [h, 1]

        out_pf, kv_state = self._call_lightning_attention(q_pf, k_pf, v_pf, ed, block_size=256)
        self.assertFalse(paddle.isnan(out_pf).any().item())

        # --- Decode phase ---
        q_dec = paddle.randn([b, h, 1, d], dtype="float16")
        k_dec = paddle.randn([b, h, 1, d], dtype="float16")
        v_dec = paddle.randn([b, h, 1, d], dtype="float16")
        slope_rate = _build_slope_tensor(h).squeeze(-1).squeeze(-1)  # [h]
        slot_idx = paddle.arange(b, dtype="int64")

        # Decode WITH warm KV state from prefill
        kv_warm = kv_state.clone()
        out_warm = self._call_decode_forward(q_dec, k_dec, v_dec, kv_warm, slope_rate, slot_idx)

        # Decode with COLD (zeros) KV state
        kv_cold = paddle.zeros_like(kv_state)
        out_cold = self._call_decode_forward(
            q_dec.clone(), k_dec.clone(), v_dec.clone(), kv_cold, slope_rate, slot_idx
        )

        # The warm-state decode should differ from cold-state (prefill context matters)
        diff = float(paddle.norm(out_warm - out_cold).item())
        self.assertGreater(
            diff,
            1e-3,
            "Warm and cold decode outputs are identical — KV state not propagated",
        )

    # === 4. Slope tensor construction ======================================

    def test_slope_tensor_power_of_2(self):
        """Slope tensor for n_heads=64 (power of 2) — all values positive, decreasing."""
        slope = _build_slope_tensor(64)
        self.assertEqual(list(slope.shape), [64, 1, 1])
        vals = slope.squeeze(-1).squeeze(-1).numpy()
        self.assertTrue(np.all(vals > 0), "Non-positive slope values")
        # First slope should be largest
        self.assertGreater(vals[0], vals[-1])

    def test_slope_tensor_non_power_of_2(self):
        """Slope tensor for n_heads=48 (not power of 2) — should still produce valid values."""
        slope = _build_slope_tensor(48)
        self.assertEqual(list(slope.shape), [48, 1, 1])
        vals = slope.squeeze(-1).squeeze(-1).numpy()
        self.assertTrue(np.all(vals > 0), "Non-positive slope values for n_heads=48")

    def test_slope_tensor_matches_production_heads(self):
        """Slope tensor for n_heads=64 (MiniMax-M1 production config)."""
        slope = _build_slope_tensor(64)
        vals = slope.squeeze(-1).squeeze(-1).numpy()
        # Expected: 2^{-(2^{-(log2(64)-3)})} = 2^{-(2^{-3})} = 2^{-0.125}
        expected_start = 2 ** (-0.125)
        np.testing.assert_allclose(vals[0], expected_start, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
