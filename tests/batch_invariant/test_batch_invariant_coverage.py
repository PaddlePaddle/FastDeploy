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
Tests for batch_invariant_ops coverage gaps.

Covers:
- batch_invariant_ops.py lines 143-144: get_compute_units() CUDA fallback
- batch_invariant_ops.py lines 476-477: mm_batch_invariant() with out parameter
"""

import os
import unittest
from unittest.mock import patch


class TestGetComputeUnitsFallback(unittest.TestCase):
    """Test get_compute_units() exception fallback (L143-144)."""

    def test_cuda_device_properties_failure_falls_back_to_cpu(self):
        """When CUDA device properties fail, falls back to os.cpu_count() (L143-144)."""
        try:
            from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
                get_compute_units,
            )
        except ImportError:
            self.skipTest("batch_invariant_ops not importable (requires triton)")

        with (
            patch("paddle.is_compiled_with_cuda", return_value=True),
            patch("paddle.device.get_device", side_effect=RuntimeError("mocked CUDA failure")),
        ):
            result = get_compute_units()
            self.assertEqual(result, os.cpu_count())

    def test_no_cuda_uses_cpu(self):
        """When CUDA is not compiled, uses CPU core count."""
        try:
            from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
                get_compute_units,
            )
        except ImportError:
            self.skipTest("batch_invariant_ops not importable (requires triton)")

        with patch("paddle.is_compiled_with_cuda", return_value=False):
            result = get_compute_units()
            self.assertEqual(result, os.cpu_count())


class TestMmBatchInvariantOut(unittest.TestCase):
    """Test mm_batch_invariant() with out parameter (L476-477)."""

    def test_out_parameter(self):
        """When out is provided, result is copied into out and returned (L476-477)."""
        try:
            import paddle

            from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
                mm_batch_invariant,
            )
        except ImportError:
            self.skipTest("batch_invariant_ops not importable (requires triton)")

        a = paddle.randn([4, 8])
        b = paddle.randn([8, 4])
        out = paddle.zeros([4, 4])

        result = mm_batch_invariant(a, b, out=out)

        # Result should be the same object as out
        self.assertIs(result, out)
        # out should no longer be zeros
        self.assertFalse(paddle.allclose(out, paddle.zeros([4, 4])).item())

    def test_no_out_parameter(self):
        """When out is None, result is returned directly (L478)."""
        try:
            import paddle

            from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
                mm_batch_invariant,
            )
        except ImportError:
            self.skipTest("batch_invariant_ops not importable (requires triton)")

        a = paddle.randn([4, 8])
        b = paddle.randn([8, 4])

        result = mm_batch_invariant(a, b)
        # Result should have correct shape
        self.assertEqual(result.shape, [4, 4])


if __name__ == "__main__":
    unittest.main()
