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
Tests for communication.py fallback paths via module reload.

Covers:
- communication.py lines 186-195: first except block fallback functions
- communication.py lines 223-229: second except block fallback function

Extends the existing test_communication_fallback.py with reload-based tests
that force the except blocks to execute even in GPU environments.
"""

import importlib
import unittest

import paddle


class TestCommunicationReloadFallbacks(unittest.TestCase):
    """Force communication.py except blocks by breaking register_custom_python_op."""

    MODULE = "fastdeploy.distributed.communication"

    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = importlib.import_module(cls.MODULE)
            cls.can_test = True
        except Exception:
            cls.can_test = False

    def _reload_with_broken_registration(self):
        """Reload communication module with register_custom_python_op that raises."""
        from unittest.mock import patch

        # Make register_custom_python_op raise, triggering except blocks
        def failing_register(**kwargs):
            def decorator(fn):
                raise RuntimeError("mocked registration failure")

            return decorator

        with patch("fastdeploy.utils.register_custom_python_op", failing_register):
            importlib.reload(self.mod)

    def test_first_except_block_fallback_functions(self):
        """After registration failure, fallback functions raise RuntimeError (L186-195)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        try:
            self._reload_with_broken_registration()

            # Verify _reg_err is set (L189)
            self.assertTrue(hasattr(self.mod, "_reg_err"))

            # Verify fallback tensor_model_parallel_all_reduce raises (L191-192)
            inp = paddle.zeros([2, 16], dtype=paddle.float16)
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.tensor_model_parallel_all_reduce(inp)
            self.assertIn("not available", str(ctx.exception))
            self.assertIn("Registration failed with", str(ctx.exception))

            # Verify fallback decode_alltoall_transpose raises (L194-195)
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.decode_alltoall_transpose(inp)
            self.assertIn("not available", str(ctx.exception))

        finally:
            # Restore module to original state
            importlib.reload(self.mod)

    def test_second_except_block_fallback_function(self):
        """After second try block failure, fallback function raises RuntimeError (L223-229)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        try:
            # For the second try block (L201-222), we need paddle.jit.marker.unified to fail
            from unittest.mock import patch

            def failing_decorator(fn):
                raise RuntimeError("mocked jit marker failure")

            with patch("paddle.jit.marker.unified", failing_decorator):
                importlib.reload(self.mod)

            # Verify _reg_err2 is set (L226)
            self.assertTrue(hasattr(self.mod, "_reg_err2"))

            # Verify fallback tensor_model_parallel_all_reduce_custom raises (L228-229)
            inp = paddle.zeros([2, 16], dtype=paddle.float16)
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.tensor_model_parallel_all_reduce_custom(inp)
            self.assertIn("not available", str(ctx.exception))

        finally:
            # Restore module to original state
            importlib.reload(self.mod)


if __name__ == "__main__":
    unittest.main()
