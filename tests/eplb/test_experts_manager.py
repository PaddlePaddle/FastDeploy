"""
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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.config import EPLBConfig, FDConfig, ModelConfig
from fastdeploy.eplb.experts_manager import RedundantExpertManager


class TestRedundantExpertManager(unittest.TestCase):
    """Test cases for experts_manager.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config objects
        self.model_config = ModelConfig()
        self.model_config.num_hidden_layers = 3
        self.model_config.moe_num_experts = 64
        self.model_config.moe_layer_start_index = 1
        self.model_config.model = "/test/model"

        self.eplb_config = EPLBConfig()
        self.eplb_config.redundant_experts_num = 0
        self.eplb_config.redundant_expert_api_user = "test_user"
        self.eplb_config.redundant_expert_api_password = "test_pass"
        self.eplb_config.redundant_expert_eplb_strategy = ""
        self.eplb_config.redundant_expert_ip_shm_size = 1024
        self.eplb_config.moe_quant_type = ""
        self.eplb_config.redundant_expert_enable_schedule_cordon = False

        self.fd_config = FDConfig()
        self.fd_config.model_config = self.model_config
        self.fd_config.eplb_config = self.eplb_config
        self.fd_config.parallel_config.local_data_parallel_id = 0
        self.fd_config.splitwise_role = "decode"

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_init(self, mock_thread, mock_process, mock_get_logger):
        """Test RedundantExpertManager initialization"""
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock process and thread
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Test initialization
        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Verify initialization
        self.assertEqual(manager.rank, 0)
        self.assertEqual(manager.ep_size, 32)
        self.assertEqual(manager.fd_config, self.fd_config)
        self.assertEqual(manager.num_logical_experts, 64)
        self.assertEqual(manager.num_replicas, 64)  # 64 + 0 redundant

        # Verify arrays are created
        self.assertEqual(manager.model_ep_rank_to_expert_id_list.shape, (3, 64))
        self.assertEqual(manager.model_expert_id_to_ep_rank_array.shape, (3, 64, 1))
        self.assertEqual(manager.model_expert_in_rank_num_list.shape, (3, 64))

        # Verify process and thread are started
        mock_process.assert_called_once()
        mock_thread.assert_called_once()

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_init_with_redundant_experts(self, mock_thread, mock_process, mock_get_logger):
        """Test initialization with redundant experts"""
        # Set up redundant experts
        self.eplb_config.redundant_experts_num = 16

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Verify with redundant experts
        self.assertEqual(manager.num_replicas, 80)  # 64 + 16 redundant
        self.assertEqual(manager.model_ep_rank_to_expert_id_list.shape, (3, 80))
        self.assertEqual(manager.model_expert_id_to_ep_rank_array.shape, (3, 64, 17))  # 16 redundant + 1

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_get_ep_rank_to_expert_id_list(self, mock_thread, mock_process, mock_get_logger):
        """Test get_ep_rank_to_expert_id_list method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set some test data
        manager.model_ep_rank_to_expert_id_list = np.array([[0, 1, 2, 3]])
        manager.model_expert_id_to_ep_rank_array = np.array([[[0], [1], [2], [3]]])
        manager.model_expert_in_rank_num_list = np.array([[1, 1, 1, 1]])

        result = manager.get_ep_rank_to_expert_id_list()

        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], np.array([[0, 1, 2, 3]]))
        np.testing.assert_array_equal(result[1], np.array([[[0], [1], [2], [3]]]))
        np.testing.assert_array_equal(result[2], np.array([[1, 1, 1, 1]]))

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_caculate_expert_rank_table(self, mock_thread, mock_process, mock_get_logger):
        """Test caculate_expert_rank_table method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set up test data
        manager.model_tokens_per_expert_stats_list = np.array([[10, 20, 30, 40], [5, 15, 25, 35]])

        # Mock the rebalance_experts function
        with patch("fastdeploy.eplb.experts_manager.rebalance_experts") as mock_rebalance:
            mock_rebalance.return_value = (
                np.array([[0, 1, 2, 3]]),  # phy2log
                np.array([[[0], [1], [2], [3]]]),  # log2phy
                np.array([[1, 1, 1, 1]]),  # logcnt
            )

            manager.caculate_expert_rank_table(is_init=True)

            # Verify that rebalance_experts was called with correct parameters
            mock_rebalance.assert_called_once()

            # Verify that arrays are updated
            np.testing.assert_array_equal(manager.model_ep_rank_to_expert_id_list[0], np.array([0, 1, 2, 3]))

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_update_weight_from_disk(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test update_weight_from_disk method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Mock IPCSignal
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance

        # Mock parent connections
        manager.parent_mg_conn = MagicMock()
        manager.parent_data_conn = MagicMock()
        manager.parent_data_conn.recv.return_value = {"result": True, "weights": ["weight1", "weight2"]}

        # Set up test data
        manager.last_model_ep_rank_to_expert_id_list = np.array([[0, 1, 2, 3]])
        manager.model_ep_rank_to_expert_id_list = np.array([[1, 2, 3, 4]])

        with patch("time.time", return_value=1000):
            manager.update_weight_from_disk()

            # Verify that data was sent and received
            manager.parent_mg_conn.send.assert_called_once()
            manager.parent_data_conn.recv.assert_called_once()

            # Verify that tensor_infos was set
            self.assertEqual(manager.tensor_infos, ["weight1", "weight2"])

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.requests.post")
    def test_allgather_expert_token_stats(self, mock_requests, mock_thread, mock_process, mock_get_logger):
        """Test allgather_expert_token_stats method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set up test addresses
        manager.dp_rank_address = ["127.0.0.1:8000", "127.0.0.1:8001"]

        # Mock successful responses
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {"data": [[[1, 2], [3, 4]]]}  # 2 layers, 2 experts

        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {"data": [[[5, 6], [7, 8]]]}  # 2 layers, 2 experts

        mock_requests.side_effect = [mock_response1, mock_response2]

        # Update model config for this test
        manager.num_hidden_layers = 2
        manager.num_logical_experts = 2

        result = manager.allgather_expert_token_stats()

        self.assertTrue(result)
        # Verify that stats were accumulated
        expected_stats = np.array([[6, 8], [10, 12]])  # [1+5, 2+6], [3+7, 4+8]
        np.testing.assert_array_equal(manager.model_tokens_per_expert_stats_list, expected_stats)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.requests.post")
    def test_broadcast_expert_token_stats(self, mock_requests, mock_thread, mock_process, mock_get_logger):
        """Test broadcast_expert_token_stats method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set up test addresses
        manager.dp_rank_address = ["127.0.0.1:8000", "127.0.0.1:8001"]

        # Mock successful responses
        mock_response1 = MagicMock()
        mock_response1.status_code = 200

        mock_response2 = MagicMock()
        mock_response2.status_code = 200

        mock_requests.side_effect = [mock_response1, mock_response2]

        result = manager.broadcast_expert_token_stats()

        self.assertTrue(result)
        self.assertEqual(mock_requests.call_count, 2)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.requests.post")
    def test_allgather_load_weight_result(self, mock_requests, mock_thread, mock_process, mock_get_logger):
        """Test allgather_load_weight_result method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set up test addresses
        manager.dp_rank_address = ["127.0.0.1:8000", "127.0.0.1:8001"]

        # Mock successful responses with mixed results
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {"data": [1, 1]}  # Two successful loads

        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {"data": [-1, 1]}  # One failed, one successful

        mock_requests.side_effect = [mock_response1, mock_response2]

        all_success, exist_fail = manager.allgather_load_weight_result()

        self.assertFalse(all_success)  # Not all successful due to failure
        self.assertTrue(exist_fail)  # There is a failure

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with empty addresses
        with (
            patch("fastdeploy.eplb.experts_manager.get_logger"),
            patch("fastdeploy.eplb.experts_manager.Process"),
            patch("fastdeploy.eplb.experts_manager.threading.Thread"),
        ):

            manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

            manager.dp_rank_address = []

            # Test allgather with empty addresses
            result = manager.allgather_expert_token_stats()
            self.assertFalse(result)

            # Test broadcast with empty addresses
            result = manager.broadcast_expert_token_stats()
            self.assertTrue(result)  # Should return True for empty list


if __name__ == "__main__":
    unittest.main()
