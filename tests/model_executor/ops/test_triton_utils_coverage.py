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
Tests for triton_utils_v2.py assert error messages.

Covers:
- triton_utils_v2.py line 192: assert for unsupported constexpr type
- triton_utils_v2.py line 200: assert for unsupported arg type
"""

import unittest
from unittest.mock import MagicMock

try:
    from fastdeploy.model_executor.ops.triton_ops.triton_utils_v2 import KernelInterface

    _CAN_IMPORT = True
except ImportError:
    _CAN_IMPORT = False


@unittest.skipUnless(_CAN_IMPORT, "triton_utils_v2 not importable")
class TestKernelInterfaceAssertMessages(unittest.TestCase):
    """Test assert messages in KernelInterface.__call__ (L192, L200)."""

    def _make_kernel_interface(self, arg_names, constexprs):
        """Create a minimal KernelInterface mock for testing arg validation."""
        ki = object.__new__(KernelInterface)
        ki.arg_names = arg_names
        ki.constexprs = constexprs
        ki.arg_exclude_constexpr = [n for i, n in enumerate(arg_names) if i not in constexprs]
        ki.op_name = "test_kernel"
        ki.fn = MagicMock()
        ki.fn.__name__ = "test_kernel"
        ki.grid = (1,)
        ki.debug = False
        ki.nargs = {}
        return ki

    def test_unsupported_constexpr_type_assert(self):
        """Unsupported constexpr type triggers assert with message (L192)."""

        # constexpr arg at index 0, pass a list (unsupported type)
        ki = self._make_kernel_interface(arg_names=["x"], constexprs={0})
        with self.assertRaises(AssertionError) as ctx:
            ki([1.5])  # float is not bool or int for constexpr
        self.assertIn("Unsupported constexpr type", str(ctx.exception))

    def test_unsupported_arg_type_assert(self):
        """Unsupported non-constexpr arg type triggers assert with message (L200)."""

        # non-constexpr arg at index 0, pass a list (unsupported type)
        ki = self._make_kernel_interface(arg_names=["x"], constexprs=set())
        with self.assertRaises((AssertionError, TypeError)) as ctx:
            ki([1, 2, 3])  # list is not Tensor, int, or float


if __name__ == "__main__":
    unittest.main()
