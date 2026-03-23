"""Test RMSNorm layer's batch_invariant_mode forward path (normalization.py:244-248).

This covers the integration between the RMSNorm *layer* and the Triton
rms_norm_batch_invariant kernel when batch_invariant_mode is enabled.
We bypass RMSNorm.__init__ (heavy FDConfig dependency) and set only
the attributes needed by forward().
"""

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    rms_norm_batch_invariant,
    set_batch_invariant_mode,
)
from fastdeploy.model_executor.layers.normalization import RMSNorm


def _make_minimal_rmsnorm(hidden_size, eps=1e-5, dtype="float32"):
    """Create a minimal RMSNorm without FDConfig by bypassing __init__."""
    layer = object.__new__(RMSNorm)
    paddle.nn.Layer.__init__(layer)
    # Attributes used by forward()
    layer.weight = paddle.create_parameter(
        shape=[hidden_size],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(value=1.0),
    )
    layer.eps = eps
    layer.bias = None
    layer.split_x = False
    layer.allgather_out = False
    return layer


class TestRMSNormBatchInvariantPath(unittest.TestCase):
    """Test RMSNorm.forward with batch_invariant_mode enabled."""

    def setUp(self):
        paddle.set_device("gpu")

    def test_no_residual(self):
        """batch_invariant path without residual_input."""
        D = 1024
        layer = _make_minimal_rmsnorm(D, dtype="float32")
        paddle.seed(42)
        x = paddle.randn([16, D], dtype="float32")

        with set_batch_invariant_mode(True):
            out, residual_out = layer.forward(x, residual_input=None)

        # residual_out should be x itself (line 236: residual_out = x)
        expected_norm = rms_norm_batch_invariant(x, layer.weight, layer.eps)
        paddle.device.synchronize()
        self.assertEqual(out.shape, [16, D])
        diff = (out.astype("float32") - expected_norm.astype("float32")).abs().max().item()
        self.assertEqual(diff, 0.0, f"Output mismatch: diff={diff}")

    def test_with_residual(self):
        """batch_invariant path with residual_input (covers lines 246-248)."""
        D = 1024
        layer = _make_minimal_rmsnorm(D, dtype="float32")
        paddle.seed(42)
        x = paddle.randn([16, D], dtype="float32")
        residual = paddle.randn([16, D], dtype="float32")

        with set_batch_invariant_mode(True):
            out, residual_out = layer.forward(x, residual_input=residual)

        # Expected: x + residual -> rms_norm_batch_invariant, residual_out = x + residual
        fused_x = x + residual
        expected_norm = rms_norm_batch_invariant(fused_x, layer.weight, layer.eps)
        paddle.device.synchronize()

        norm_diff = (out.astype("float32") - expected_norm.astype("float32")).abs().max().item()
        res_diff = (residual_out.astype("float32") - fused_x.astype("float32")).abs().max().item()
        self.assertEqual(norm_diff, 0.0, f"Norm output mismatch: diff={norm_diff}")
        self.assertEqual(res_diff, 0.0, f"Residual output mismatch: diff={res_diff}")

    def test_bfloat16(self):
        """batch_invariant path with bfloat16 input."""
        D = 3584
        layer = _make_minimal_rmsnorm(D, dtype="bfloat16")
        paddle.seed(0)
        x = paddle.randn([32, D], dtype="bfloat16")
        residual = paddle.randn([32, D], dtype="bfloat16")

        with set_batch_invariant_mode(True):
            out, residual_out = layer.forward(x, residual_input=residual)

        fused_x = x + residual
        expected_norm = rms_norm_batch_invariant(fused_x, layer.weight, layer.eps)
        paddle.device.synchronize()

        norm_diff = (out.astype("float32") - expected_norm.astype("float32")).abs().max().item()
        self.assertEqual(norm_diff, 0.0, f"bf16 norm output mismatch: diff={norm_diff}")


if __name__ == "__main__":
    unittest.main()
