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

import subprocess
import unittest
from unittest.mock import patch

from fastdeploy.cache_manager.transfer_factory.rdma_cache_transfer import (
    RDMACommManager,
)


class TestRDMACommManager(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.patcher1 = patch.dict("os.environ", {}, clear=True)
        self.mock_env = self.patcher1.start()

        # Mock subprocess run
        self.patcher2 = patch("subprocess.run", wraps=subprocess.run)
        self.mock_run = self.patcher2.start()

        # Mock current_platform
        self.patcher3 = patch("fastdeploy.platforms.current_platform")
        self.mock_platform = self.patcher3.start()
        self.mock_platform.is_cuda.return_value = True
        self.mock_platform.device_name = "gpu"

        # Mock RDMA library
        self.patcher4 = patch("rdma_comm.RDMACommunicator")
        self.mock_rdma_comm = self.patcher4.start()

        # Test parameters
        self.test_params = {
            "splitwise_role": "prefill",
            "rank": 0,
            "gpu_id": 0,
            "cache_k_ptr_list": [1, 2, 3],
            "cache_v_ptr_list": [4, 5, 6],
            "max_block_num": 10,
            "block_bytes": 1024,
            "rdma_port": 12345,
            "prefill_tp_size": 1,
            "prefill_tp_idx": 0,
        }

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()

    def test_init_rdma_comm_manager(self):
        """Test RDMACommManager initialization"""
        self.assertIsNone(self.mock_env.get("KVCACHE_RDMA_NICS"), None)
        manager = RDMACommManager(**self.test_params)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.splitwise_role, "prefill")
        self.mock_rdma_comm.assert_called_once()
        self.mock_run.assert_called()
        self.assertIsNotNone(self.mock_env.get("KVCACHE_RDMA_NICS"))

    def test_connect_success(self):
        """Test successful connection"""
        manager = RDMACommManager(**self.test_params)
        manager.messager.is_connected.return_value = False
        manager.messager.connect.return_value = 0

        result = manager.connect("127.0.0.1", 12345)
        self.assertTrue(result)
        manager.messager.connect.assert_called_once_with("127.0.0.1", "12345", 0)

    def test_write_cache(self):
        """Test write_cache method"""
        manager = RDMACommManager(**self.test_params)
        manager.messager.write_cache.return_value = True

        result = manager.write_cache("127.0.0.1", 12345, [1, 2], [3, 4], 0)
        self.assertTrue(result)
        manager.messager.write_cache.assert_called_once_with("127.0.0.1", "12345", [1, 2], [3, 4], 0)


if __name__ == "__main__":
    unittest.main()
