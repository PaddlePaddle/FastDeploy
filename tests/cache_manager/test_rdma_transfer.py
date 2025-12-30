# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import unittest
from unittest.mock import Mock, patch

from fastdeploy.cache_manager.transfer_factory.rdma_cache_transfer import (
    RDMACommManager,
)


class TestRDMACommManager(unittest.TestCase):
    def setUp(self):
        self.args = {
            "splitwise_role": "prefill",
            "gpu_id": 0,
            "cache_k_ptr_list": [1, 2, 3],
            "cache_v_ptr_list": [4, 5, 6],
            "max_block_num": 10,
            "block_bytes": 1024,
            "rdma_port": 12345,
            "prefill_tp_size": 1,
            "prefill_tp_idx": 0,
        }

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "", "KVCACHE_RDMA_NICS": ""})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_init_rdma_comm_manager_on_gpu_init_all(self, mock_run, mock_rdma_comm, mock_platform):
        # Case: Automatically set all environment variables
        mock_platform.is_cuda.return_value = True
        mock_platform.device_name = "gpu"
        mock_run.side_effect = [
            Mock(returncode=0, stdout="8.0\n"),
            Mock(returncode=0, stdout="KVCACHE_RDMA_NICS=mlx5_2\n"),
        ]

        manager = RDMACommManager(**self.args)
        self.assertEqual(manager.splitwise_role, "prefill")
        self.assertEqual(mock_run.call_count, 2)
        mock_rdma_comm.assert_called_once()
        self.assertEqual(os.getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE"), "1")
        self.assertEqual(os.getenv("KVCACHE_RDMA_NICS"), "mlx5_2")

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "", "KVCACHE_RDMA_NICS": "mlx5_1"})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_init_rdma_comm_manager_on_gpu_init_gdrcopy(self, mock_run, mock_rdma_comm, mock_platform):
        # Case: Only set KVCACHE_GDRCOPY_FLUSH_ENABLE
        mock_platform.is_cuda.return_value = True
        mock_platform.device_name = "gpu"
        mock_run.side_effect = [Mock(returncode=0, stdout="8.0\n")]

        manager = RDMACommManager(**self.args)
        self.assertEqual(manager.splitwise_role, "prefill")
        self.assertEqual(mock_run.call_count, 1)
        mock_rdma_comm.assert_called_once()
        self.assertEqual(os.getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE"), "1")
        self.assertEqual(os.getenv("KVCACHE_RDMA_NICS"), "mlx5_1")

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "0", "KVCACHE_RDMA_NICS": ""})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_init_rdma_comm_manager_on_gpu_init_nics(self, mock_run, mock_rdma_comm, mock_platform):
        # Case: Only set KVCACHE_RDMA_NICS
        mock_platform.is_cuda.return_value = True
        mock_platform.device_name = "gpu"
        mock_run.side_effect = [Mock(returncode=0, stdout="KVCACHE_RDMA_NICS=mlx5_2\n")]

        manager = RDMACommManager(**self.args)
        self.assertEqual(manager.splitwise_role, "prefill")
        self.assertEqual(mock_run.call_count, 1)
        mock_rdma_comm.assert_called_once()
        self.assertEqual(os.getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE"), "0")
        self.assertEqual(os.getenv("KVCACHE_RDMA_NICS"), "mlx5_2")

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "0", "KVCACHE_RDMA_NICS": "mlx5_1"})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_init_rdma_comm_manager_on_gpu_init_nothing(self, mock_run, mock_rdma_comm, mock_platform):
        # Case: Do not set any environment variables
        mock_platform.is_cuda.return_value = True
        mock_platform.device_name = "gpu"

        manager = RDMACommManager(**self.args)
        self.assertEqual(manager.splitwise_role, "prefill")
        self.assertEqual(mock_run.call_count, 0)
        mock_rdma_comm.assert_called_once()
        self.assertEqual(os.getenv("KVCACHE_GDRCOPY_FLUSH_ENABLE"), "0")
        self.assertEqual(os.getenv("KVCACHE_RDMA_NICS"), "mlx5_1")

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "0", "KVCACHE_RDMA_NICS": "mlx5_1"})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_connect_success(self, mock_run, mock_rdma_comm, mock_platform):
        """Test successful connection"""
        manager = RDMACommManager(**self.args)
        manager.messager.is_connected.return_value = False
        manager.messager.connect.return_value = 0

        result = manager.connect("127.0.0.1", 12345)
        self.assertTrue(result)
        manager.messager.connect.assert_called_once_with("127.0.0.1", "12345", 0)

    @patch.dict("os.environ", {"KVCACHE_GDRCOPY_FLUSH_ENABLE": "0", "KVCACHE_RDMA_NICS": "mlx5_1"})
    @patch("fastdeploy.platforms.current_platform")
    @patch("rdma_comm.RDMACommunicator")
    @patch("subprocess.run")
    def test_write_cache(self, mock_run, mock_rdma_comm, mock_platform):
        """Test write_cache method"""
        manager = RDMACommManager(**self.args)
        manager.messager.write_cache.return_value = True

        result = manager.write_cache("127.0.0.1", 12345, [1, 2], [3, 4], 0)
        self.assertTrue(result)
        manager.messager.write_cache.assert_called_once_with("127.0.0.1", "12345", [1, 2], [3, 4], 0)


if __name__ == "__main__":
    unittest.main()
