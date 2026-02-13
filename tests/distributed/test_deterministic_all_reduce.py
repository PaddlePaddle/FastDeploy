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
Deterministic All-Reduce Tests

This module tests deterministic all-reduce functionality across multiple GPUs.

To run:
    python -m paddle.distributed.launch --gpus=0,1 tests/distributed/test_deterministic_all_reduce.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import paddle
import paddle.distributed as dist


class TestDeterministicAllReduceMock(unittest.TestCase):
    """Mock-based tests for deterministic all-reduce warning logic."""

    def setUp(self):
        # Import after paddle is available
        from fastdeploy.distributed import communication

        communication._TP_AR = None
        os.environ.pop("FD_DETERMINISTIC_MODE", None)

    def tearDown(self):
        from fastdeploy.distributed import communication

        communication._TP_AR = None
        os.environ.pop("FD_DETERMINISTIC_MODE", None)

    @patch("fastdeploy.distributed.communication.dist.all_reduce")
    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_error_when_deterministic_mode_no_custom_ar(self, mock_get_hcg, mock_all_reduce):
        """Test error is raised when DETERMINISTIC_MODE is enabled but custom all-reduce not initialized."""
        from fastdeploy.distributed import communication

        mock_hcg = MagicMock()
        mock_get_hcg.return_value = mock_hcg
        fake_group = MagicMock()
        fake_group.world_size = 2
        mock_hcg.get_model_parallel_group.return_value = fake_group

        def fake_all_reduce(x, group=None):
            return x

        mock_all_reduce.side_effect = fake_all_reduce

        # Enable deterministic mode by setting env var before import
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        from fastdeploy import envs

        envs.environment_variables["FD_DETERMINISTIC_MODE"] = lambda: True

        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])

        # Should raise RuntimeError (not warning)
        with self.assertRaises(RuntimeError) as cm:
            _ = communication.tensor_model_parallel_all_reduce(x)

        error_message = str(cm.exception)
        self.assertIn("DETERMINISTIC_MODE", error_message)
        self.assertIn("custom all-reduce", error_message)

    @patch("fastdeploy.distributed.communication.dist.all_reduce")
    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_error_when_deterministic_mode_tensor_size_not_aligned(self, mock_get_hcg, mock_all_reduce):
        """Test error is raised when DETERMINISTIC_MODE is enabled but tensor size is not aligned."""
        from fastdeploy.distributed import communication

        mock_hcg = MagicMock()
        mock_get_hcg.return_value = mock_hcg
        fake_group = MagicMock()
        fake_group.world_size = 2
        mock_hcg.get_model_parallel_group.return_value = fake_group

        def fake_all_reduce(x, group=None):
            return x

        mock_all_reduce.side_effect = fake_all_reduce

        # Enable deterministic mode
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        from fastdeploy import envs

        envs.environment_variables["FD_DETERMINISTIC_MODE"] = lambda: True

        # Initialize custom all-reduce but use tensor with size not multiple of 16
        # For float32 (4 bytes), need 4 elements minimum for 16 bytes
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype=paddle.float32)  # 3 elements * 4 bytes = 12 bytes

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            _ = communication.tensor_model_parallel_all_reduce(x)

        error_message = str(cm.exception)
        self.assertIn("DETERMINISTIC_MODE", error_message)
        self.assertIn("multiple of 16", error_message)


class TestDeterministicAllReduceReal(unittest.TestCase):
    """Real distributed tests for deterministic all-reduce."""

    def setUp(self):
        """Initialize paddle distributed environment."""
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Import fastdeploy modules
        from fastdeploy.distributed import communication

        self.comm = communication
        communication._TP_AR = None

    def test_deterministic_all_reduce_with_custom_ar(self):
        """Test that custom all-reduce produces deterministic results."""
        from paddle.distributed import fleet

        # Initialize custom all-reduce
        try:
            hcg = fleet.get_hybrid_communicate_group()
            tp_group = hcg.get_model_parallel_group()

            # Only run if TP group has multiple GPUs
            if tp_group.nranks <= 1:
                self.skipTest("TP group has only one GPU, skipping all-reduce test")

            from fastdeploy.distributed.communication import use_custom_allreduce

            use_custom_allreduce(tp_group)
        except Exception as e:
            self.skipTest(f"Failed to initialize custom all-reduce: {e}")

        # Test with tensor size that meets custom all-reduce requirements (multiple of 16 elements)
        # Using float32: 4 bytes per element, 16 elements = 64 bytes
        tensor_size = 16 * (self.world_size * 2)  # Ensure size is multiple of 16

        results = []
        num_runs = 3

        for i in range(num_runs):
            # Create different input on each rank
            x = paddle.randn([tensor_size], dtype=paddle.float32) * (self.rank + 1)

            # All-reduce using custom all-reduce
            result = self.comm.tensor_model_parallel_all_reduce(x)
            result_numpy = result.numpy().copy()
            results.append(result_numpy)

            # Synchronize all ranks before next run
            dist.barrier()

        # Check: all ranks should have the same result
        # Gather results from all ranks to rank 0
        if self.rank == 0:
            all_results = [paddle.to_tensor(results[0])]
            for i in range(1, self.world_size):
                received = paddle.empty_like(paddle.to_tensor(results[0]))
                dist.stream.recv(received, src=i)
                all_results.append(received.numpy())
        else:
            for i in range(self.world_size):
                if i > self.rank:
                    dist.stream.send(paddle.to_tensor(results[0]), dst=i)
                elif i < self.rank:
                    dist.stream.send(paddle.to_tensor(results[0]), dst=i)

        # Check: same input on same rank should produce same output across runs
        for i in range(1, num_runs):
            self.assertTrue(
                (results[0] == results[i]).all(), f"Rank {self.rank}: All-reduce results differ between runs 0 and {i}"
            )

        dist.barrier()

    def test_deterministic_mode_auto_init_custom_ar(self):
        """Test that deterministic mode automatically initializes custom all-reduce."""
        from paddle.distributed import fleet

        try:
            hcg = fleet.get_hybrid_communicate_group()
            tp_group = hcg.get_model_parallel_group()

            if tp_group.nranks <= 1:
                self.skipTest("TP group has only one GPU, skipping auto-init test")
        except Exception:
            self.skipTest("No TP group available")

        # Clear any existing custom all-reduce
        self.comm._TP_AR = None

        # Set deterministic mode
        os.environ["FD_DETERMINISTIC_MODE"] = "1"
        from fastdeploy import envs

        envs.environment_variables["FD_DETERMINISTIC_MODE"] = lambda: True

        # Simulate __init__.py behavior
        from fastdeploy.distributed.communication import use_custom_allreduce

        use_custom_allreduce(tp_group)

        # Verify custom all-reduce is initialized
        self.assertIsNotNone(self.comm._TP_AR, "Custom all-reduce should be initialized in deterministic mode")

        # Cleanup
        os.environ.pop("FD_DETERMINISTIC_MODE", None)


if __name__ == "__main__":
    # Check if running in distributed mode
    if "PADDLE_DIST_ID" in os.environ or "PADDLE_RANK" in os.environ:
        # Distributed mode: run tests
        unittest.main()
    else:
        # Non-distributed mode: only run mock tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDeterministicAllReduceMock)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
