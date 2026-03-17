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
Tests for CustomAllreduce._initialized guard (aff1eae8).

Behavior under test:
  - should_custom_ar() returns False when _initialized is False.
  - Construction with custom_ar=True but no distributed environment
    leaves _initialized=False (world_size=1 early return).

Why mock:
  - paddle.distributed.get_rank / get_world_size are distributed communication
    primitives that require a real multi-GPU NCCL group. We mock them at the
    external system boundary so the test runs on a single process.
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.distributed.custom_all_reduce.custom_all_reduce import (
    CustomAllreduce,
    custom_ar,
)


class TestCustomAllreduceInitializedGuard(unittest.TestCase):
    """Behavior: should_custom_ar returns False when not fully initialized."""

    @unittest.skipUnless(custom_ar, "custom allreduce library not available")
    @patch("paddle.distributed.get_world_size", return_value=1)
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_single_gpu_not_initialized(self, _mock_rank, _mock_ws):
        """world_size=1 → constructor returns early → _initialized stays False."""
        fake_group = MagicMock()
        ar = CustomAllreduce(group=fake_group, max_size=8192 * 1024)
        self.assertFalse(ar._initialized)

    @unittest.skipUnless(custom_ar, "custom allreduce library not available")
    @patch("paddle.distributed.get_world_size", return_value=1)
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_should_custom_ar_false_when_not_initialized(self, _mock_rank, _mock_ws):
        """should_custom_ar must return False when _initialized is False."""
        fake_group = MagicMock()
        ar = CustomAllreduce(group=fake_group, max_size=8192 * 1024)

        inp = paddle.zeros([4, 1024], dtype=paddle.float16)
        self.assertFalse(ar.should_custom_ar(inp))

    @unittest.skipUnless(custom_ar, "custom allreduce library not available")
    @patch("paddle.distributed.get_world_size", return_value=3)
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_unsupported_world_size_not_initialized(self, _mock_rank, _mock_ws):
        """world_size=3 (not in SUPPORTED_WORLD_SIZES) → _initialized stays False."""
        fake_group = MagicMock()
        ar = CustomAllreduce(group=fake_group, max_size=8192 * 1024)
        self.assertFalse(ar._initialized)

        inp = paddle.zeros([4, 1024], dtype=paddle.float16)
        self.assertFalse(ar.should_custom_ar(inp))


if __name__ == "__main__":
    unittest.main()
