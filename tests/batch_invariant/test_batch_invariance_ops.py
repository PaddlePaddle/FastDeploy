# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/test_batch_invariance.py

import random
import unittest

import paddle
import pytest

from fastdeploy.model_executor.layers.batch_invariant_ops import (
    disable_batch_invariant_mode,
    init_deterministic_mode,
    is_batch_invariant_mode_enabled,
    set_batch_invariant_mode,
)

pytestmark = pytest.mark.gpu

B, D = 2048, 4096
NUM_ITERS = 10


def _create_softmax_trap_tensor(B, D, dtype):
    """
    Constructs a "trap" tensor designed to trigger batch-invariance issues in Softmax/LogSoftmax.
    Values spanning vastly different magnitudes after exp trigger rounding order sensitivity.
    """
    max_val = 20.0
    trap_values = [max_val, max_val - 4.6, max_val - 11.5, max_val - 23.0]
    a = paddle.full((B, D), -1000.0, dtype=dtype)
    for i in range(B):
        indices = random.sample(range(D), k=len(trap_values))
        for j, val in enumerate(trap_values):
            a[i, indices[j]] = val
    return a


def _compute_single_vs_batch(op_name, a, b=None):
    """Compute op on single row vs full batch, return (is_equal, max_diff)."""
    if op_name == "mm":
        out1 = paddle.mm(a[:1], b)
        out2 = paddle.mm(a, b)[:1]
    elif op_name == "mean":
        out1 = paddle.mean(a[:1], axis=-1)
        out2 = paddle.mean(a, axis=-1)[:1]
    elif op_name == "log_softmax":
        out1 = paddle.nn.functional.log_softmax(a[:1])
        out2 = paddle.nn.functional.log_softmax(a)[:1]
    elif op_name == "addmm":
        out1 = paddle.addmm(a[:1].squeeze(0), a[:1], b)
        out2 = paddle.addmm(a[:1].squeeze(0), a, b)[:1]
    else:
        raise ValueError(f"Unknown op: {op_name}")

    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff


def _make_inputs(op_name, dtype):
    """Create input tensors for the given op."""
    if op_name == "log_softmax":
        a = _create_softmax_trap_tensor(B, D, dtype)
        return a, None
    else:
        a = paddle.linspace(-100, 100, B * D, dtype=dtype).reshape(B, D)
        b = paddle.linspace(-100, 100, D * D, dtype=dtype).reshape(D, D)
        return a, b


# Define test parameters: (op_name, dtypes)
OP_CONFIGS = [
    ("mm", [paddle.float32, paddle.bfloat16]),
    ("mean", [paddle.float32, paddle.bfloat16]),
    ("log_softmax", [paddle.float32, paddle.bfloat16, paddle.float16]),
    ("addmm", [paddle.float32, paddle.bfloat16]),
]


class TestBatchInvariance(unittest.TestCase):
    def setUp(self):
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)

    def test_init_deterministic_mode_enables_batch_invariant(self):
        """init_deterministic_mode() must enable batch-invariant mode."""
        # Ensure mode is disabled first
        disable_batch_invariant_mode()
        self.assertFalse(is_batch_invariant_mode_enabled())

        # Call init_deterministic_mode
        init_deterministic_mode()

        # Verify mode is now enabled
        self.assertTrue(is_batch_invariant_mode_enabled())

    def test_init_deterministic_mode_idempotent(self):
        """init_deterministic_mode() should be idempotent (no error when already enabled)."""
        # Ensure mode is enabled
        init_deterministic_mode()
        self.assertTrue(is_batch_invariant_mode_enabled())

        # Call again - should not raise
        init_deterministic_mode()

        # Still enabled
        self.assertTrue(is_batch_invariant_mode_enabled())

    def _run_invariance_check(self, op_name, dtypes, should_assert):
        """Run batch invariance check for an op across dtypes."""
        for dtype in dtypes:
            a, b = _make_inputs(op_name, dtype)
            max_diffs = []
            for _ in range(NUM_ITERS):
                _, diff = _compute_single_vs_batch(op_name, a, b)
                max_diffs.append(diff)

            max_diff = max(max_diffs)
            print(f"  {op_name} {dtype}: batch_invariant={max_diff == 0}, " f"max_diff={max_diff}")
            if should_assert:
                self.assertEqual(max_diff, 0, f"{op_name} is not batch-invariant for {dtype}, max_diff={max_diff}")


def _make_test(op_name, dtypes):
    """Factory function to generate a test method for each op."""

    def test_method(self):
        print(f"\nStandard Paddle ({op_name}):")
        with set_batch_invariant_mode(False):
            self._run_invariance_check(op_name, dtypes, should_assert=False)

        print(f"\nBatch-Invariant Mode ({op_name}):")
        with set_batch_invariant_mode(True):
            self._run_invariance_check(op_name, dtypes, should_assert=True)

    test_method.__doc__ = f"Test batch invariance for {op_name}"
    return test_method


# Dynamically generate test methods for each op
for _op_name, _dtypes in OP_CONFIGS:
    setattr(
        TestBatchInvariance,
        f"test_{_op_name}_batch_invariance",
        _make_test(_op_name, _dtypes),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
