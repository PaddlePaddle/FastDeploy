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

"""
Tests for communication.py error handling improvements (aff1eae8 + 029e4cf8).

Covers:
1. tensor_byte_size() — pure computation, no mocking needed.
2. The _reg_err closure pattern — 029e4cf8 fixed a Python 3 bug where the
   except-block variable `e` was garbage-collected, breaking closures that
   reference it.  Pure Python tests, no mocking needed.
3. Fallback function behavior — when op registration fails, the fallback
   functions must raise RuntimeError with the original error message.
   In GPU environments where registration succeeds, these tests are skipped.
"""

import unittest

import paddle

from fastdeploy.distributed.communication import tensor_byte_size


# ---------------------------------------------------------------------------
# 1. tensor_byte_size() — behaviour tests
# ---------------------------------------------------------------------------
class TestTensorByteSize(unittest.TestCase):
    """tensor_byte_size must return shape-product * element_size."""

    def test_1d_float32(self):
        t = paddle.zeros([10], dtype=paddle.float32)
        self.assertEqual(tensor_byte_size(t), 10 * 4)

    def test_2d_float16(self):
        t = paddle.zeros([4, 8], dtype=paddle.float16)
        self.assertEqual(tensor_byte_size(t), 4 * 8 * 2)

    def test_3d_bfloat16(self):
        t = paddle.zeros([2, 3, 4], dtype=paddle.bfloat16)
        self.assertEqual(tensor_byte_size(t), 2 * 3 * 4 * 2)

    def test_single_element(self):
        t = paddle.zeros([1], dtype=paddle.float32)
        self.assertEqual(tensor_byte_size(t), 4)

    def test_matches_numel_times_element_size(self):
        """Result must be identical to numel * element_size for arbitrary shapes."""
        cases = [
            ([16], paddle.float32),
            ([4, 8], paddle.float16),
            ([2, 3, 5], paddle.bfloat16),
            ([1, 1, 1, 1], paddle.float32),
        ]
        for shape, dtype in cases:
            t = paddle.zeros(shape, dtype=dtype)
            expected = t.numel().item() * t.element_size()
            self.assertEqual(tensor_byte_size(t), expected, f"shape={shape}, dtype={dtype}")


# ---------------------------------------------------------------------------
# 2. _reg_err closure pattern — pure Python behaviour tests
# ---------------------------------------------------------------------------
class TestRegErrClosurePattern(unittest.TestCase):
    """029e4cf8 fixed a closure bug in communication.py.

    In Python 3, the `as` target of an except clause is deleted after
    the block exits.  Using `_reg_err = e` inside the block preserves
    the exception for closures defined alongside it.
    """

    def test_fixed_pattern_preserves_exception(self):
        """_reg_err = e keeps the exception accessible after except exits."""
        try:
            raise ImportError("simulated op registration failure")
        except Exception as e:
            _reg_err = e

            def fallback():
                raise RuntimeError(f"Not available. Failed with: {_reg_err}")

        with self.assertRaises(RuntimeError) as ctx:
            fallback()
        self.assertIn("simulated op registration failure", str(ctx.exception))

    def test_buggy_pattern_loses_exception(self):
        """Direct reference to `e` in closure raises NameError after except block."""
        try:
            raise ImportError("original error")
        except Exception as e:  # noqa: F841 — intentionally "unused"; Python 3 deletes it

            def buggy():
                return str(e)  # noqa: F821 — `e` is undefined here, that's the point

        # Python 3 deletes `e` after the except block; closure sees unbound var
        with self.assertRaises(NameError):
            buggy()

    def test_two_independent_except_blocks(self):
        """Each except block must use a separate variable (_reg_err / _reg_err2)."""
        try:
            raise ValueError("first failure")
        except Exception as e:
            _reg_err = e

            def fallback1():
                raise RuntimeError(f"first: {_reg_err}")

        try:
            raise TypeError("second failure")
        except Exception as e:
            _reg_err2 = e

            def fallback2():
                raise RuntimeError(f"second: {_reg_err2}")

        with self.assertRaises(RuntimeError) as ctx1:
            fallback1()
        self.assertIn("first failure", str(ctx1.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            fallback2()
        self.assertIn("second failure", str(ctx2.exception))


# ---------------------------------------------------------------------------
# 3. Fallback functions — only testable when op registration failed
# ---------------------------------------------------------------------------
class TestCommunicationFallbackFunctions(unittest.TestCase):
    """When op registration fails at import time, calling the functions
    must raise RuntimeError containing the original error message.

    In GPU environments where registration succeeds, these tests are skipped.
    """

    def test_fallback_tensor_model_parallel_all_reduce(self):
        from fastdeploy.distributed import communication

        if not hasattr(communication, "_reg_err"):
            self.skipTest("Op registration succeeded; no fallback to test")

        inp = paddle.zeros([2, 16], dtype=paddle.float16)
        with self.assertRaises(RuntimeError) as ctx:
            communication.tensor_model_parallel_all_reduce(inp)
        self.assertIn("not available", str(ctx.exception))
        self.assertIn("Registration failed with", str(ctx.exception))

    def test_fallback_decode_alltoall_transpose(self):
        from fastdeploy.distributed import communication

        if not hasattr(communication, "_reg_err"):
            self.skipTest("Op registration succeeded; no fallback to test")

        inp = paddle.zeros([2, 16], dtype=paddle.float16)
        with self.assertRaises(RuntimeError) as ctx:
            communication.decode_alltoall_transpose(inp)
        self.assertIn("not available", str(ctx.exception))

    def test_fallback_tensor_model_parallel_all_reduce_custom(self):
        from fastdeploy.distributed import communication

        if not hasattr(communication, "_reg_err2"):
            self.skipTest("Op registration succeeded; no fallback to test")

        inp = paddle.zeros([2, 16], dtype=paddle.float16)
        with self.assertRaises(RuntimeError) as ctx:
            communication.tensor_model_parallel_all_reduce_custom(inp)
        self.assertIn("not available", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
