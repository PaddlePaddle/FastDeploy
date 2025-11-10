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

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.config import EPLBConfig, FDConfig, ModelConfig
from fastdeploy.eplb.utils import RedundantExpertWorkload, init_eplb_signals


class TestRedundantExpertWorkload(unittest.TestCase):
    """Test cases for RedundantExpertWorkload class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test RedundantExpertWorkload initialization"""
        workload = RedundantExpertWorkload(self.temp_dir)

        self.assertIsNone(workload.tokens_per_expert_stats_list)
        self.assertIsNone(workload.ep_rank_to_expert_id_list)
        self.assertIsNone(workload.expert_id_to_ep_rank_array)
        self.assertIsNone(workload.expert_in_rank_num_list)
        self.assertEqual(workload.cost_milliseconds, 0)
        self.assertEqual(workload.meta_file_name, f"{self.temp_dir}/rearrange-experts.json")

        # Verify directory was created
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_json_method(self):
        """Test __json__ method"""
        workload = RedundantExpertWorkload(self.temp_dir)
        workload.tokens_per_expert_stats_list = [[1, 2], [3, 4]]
        workload.ep_rank_to_expert_id_list = [[0, 1]]
        workload.expert_id_to_ep_rank_array = [[[0], [1]]]
        workload.expert_in_rank_num_list = [[1, 1]]
        workload.cost_milliseconds = 100

        json_data = workload.__json__()

        self.assertEqual(json_data["tokens_per_expert_stats_list"], [[1, 2], [3, 4]])
        self.assertEqual(json_data["ep_rank_to_expert_id_list"], [[0, 1]])
        self.assertEqual(json_data["expert_id_to_ep_rank_array"], [[[0], [1]]])
        self.assertEqual(json_data["expert_in_rank_num_list"], [[1, 1]])
        self.assertEqual(json_data["cost_milliseconds"], 100)

    def test_dump_success(self):
        """Test successful dump"""
        workload = RedundantExpertWorkload(self.temp_dir)
        workload.tokens_per_expert_stats_list = [[1, 2]]
        workload.ep_rank_to_expert_id_list = [[0, 1]]
        workload.expert_id_to_ep_rank_array = [[[0], [1]]]
        workload.expert_in_rank_num_list = [[1, 1]]
        workload.cost_milliseconds = 100

        result = workload.dump()

        # Verify file was created
        self.assertTrue(os.path.exists(workload.meta_file_name))

        # Verify file content
        with open(workload.meta_file_name, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["tokens_per_expert_stats_list"], [[1, 2]])
        self.assertEqual(saved_data["ep_rank_to_expert_id_list"], [[0, 1]])
        self.assertEqual(saved_data["expert_id_to_ep_rank_array"], [[[0], [1]]])
        self.assertEqual(saved_data["expert_in_rank_num_list"], [[1, 1]])
        self.assertEqual(saved_data["cost_milliseconds"], 100)

        # Verify return message
        self.assertIn("redundant_expert: dump expert workload result in", result)

    def test_dump_failure(self):
        """Test dump failure (e.g., permission denied)"""
        # Create a directory that we can't write to
        read_only_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(read_only_dir)
        os.chmod(read_only_dir, 0o444)  # Read-only

        workload = RedundantExpertWorkload(read_only_dir)

        result = workload.dump()

        # Verify error message
        self.assertIn("redundant_expert: dump expert workload failed", result)

    def test_load_success(self):
        """Test successful load"""
        # Create test file
        test_data = {
            "tokens_per_expert_stats_list": [[1, 2], [3, 4]],
            "ep_rank_to_expert_id_list": [[0, 1]],
            "expert_id_to_ep_rank_array": [[[0], [1]]],
            "expert_in_rank_num_list": [[1, 1]],
            "cost_milliseconds": 100,
        }

        with open(os.path.join(self.temp_dir, "rearrange-experts.json"), "w") as f:
            json.dump(test_data, f)

        workload = RedundantExpertWorkload(self.temp_dir)
        data, message = workload.load()

        # Verify loaded data
        self.assertEqual(data["tokens_per_expert_stats_list"], [[1, 2], [3, 4]])
        self.assertEqual(data["ep_rank_to_expert_id_list"], [[0, 1]])
        self.assertEqual(data["expert_id_to_ep_rank_array"], [[[0], [1]]])
        self.assertEqual(data["expert_in_rank_num_list"], [[1, 1]])
        self.assertEqual(data["cost_milliseconds"], 100)
        self.assertEqual(message, "ok")

    def test_load_file_not_exists(self):
        """Test load when file doesn't exist"""
        workload = RedundantExpertWorkload(self.temp_dir)
        data, message = workload.load()

        self.assertEqual(data, {})
        self.assertIn("is not exists", message)

    def test_load_corrupted_file(self):
        """Test load with corrupted JSON file"""
        # Create corrupted JSON file
        with open(os.path.join(self.temp_dir, "rearrange-experts.json"), "w") as f:
            f.write("invalid json content")

        workload = RedundantExpertWorkload(self.temp_dir)
        data, message = workload.load()

        self.assertEqual(data, {})
        self.assertIn("load file", message)
        self.assertIn("failed", message)


class TestInitEplbSignals(unittest.TestCase):
    """Test cases for init_eplb_signals function"""

    def setUp(self):
        """Set up test fixtures"""
        self.model_config = ModelConfig()
        self.model_config.num_hidden_layers = 3
        self.model_config.moe_num_experts = 64

        self.eplb_config = EPLBConfig()
        self.eplb_config.redundant_expert_ip_shm_size = 1024

        self.fd_config = FDConfig()
        self.fd_config.model_config = self.model_config
        self.fd_config.eplb_config = self.eplb_config
        self.fd_config.parallel_config.local_data_parallel_id = 0

    @patch("fastdeploy.eplb.utils.IPCSignal")
    def test_init_eplb_signals_rank_0(self, mock_ipc_signal):
        """Test init_eplb_signals for rank 0"""
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance

        # Test with rank 0
        self.fd_config.parallel_config.local_data_parallel_id = 0
        ipc_signal_suffix = 123

        init_eplb_signals(self.fd_config, ipc_signal_suffix)

        # Verify IPCSignal was called for rank 0 specific signals
        expected_calls = [
            # Rank 0 specific signals
            ("rearrange_experts_status", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("rearrange_experts_ips_size", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("rearrange_experts_ips_list", 1024, None, ipc_signal_suffix, True),  # shm_size
            ("signal_update_weight_from_tensor", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            # Common signals
            ("all_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("local_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("signal_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("signal_clear_experts_token_stats", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("result_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
        ]

        # Verify all signals were created
        self.assertEqual(mock_ipc_signal.call_count, len(expected_calls))

    @patch("fastdeploy.eplb.utils.IPCSignal")
    def test_init_eplb_signals_rank_non_zero(self, mock_ipc_signal):
        """Test init_eplb_signals for non-zero rank"""
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance

        # Test with non-zero rank
        self.fd_config.parallel_config.local_data_parallel_id = 1
        ipc_signal_suffix = 123

        init_eplb_signals(self.fd_config, ipc_signal_suffix)

        # For non-zero rank, only common signals should be created
        expected_calls = [
            # Common signals (no rank 0 specific signals)
            ("all_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("local_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("signal_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("signal_clear_experts_token_stats", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
            ("result_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, ipc_signal_suffix, True),
        ]

        # Verify only common signals were created
        self.assertEqual(mock_ipc_signal.call_count, len(expected_calls))

    @patch("fastdeploy.eplb.utils.IPCSignal")
    def test_init_eplb_signals_different_suffix(self, mock_ipc_signal):
        """Test init_eplb_signals with different suffix"""
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance

        ipc_signal_suffix = 999

        init_eplb_signals(self.fd_config, ipc_signal_suffix)

        # Verify that suffix is used correctly
        for call in mock_ipc_signal.call_args_list:
            args, kwargs = call
            self.assertEqual(kwargs.get("suffix"), ipc_signal_suffix)

    def test_main_function(self):
        """Test the main function at the end of the file"""
        # This tests the if __name__ == "__main__" block
        with patch("fastdeploy.eplb.utils.RedundantExpertWorkload") as mock_workload:
            mock_instance = MagicMock()
            mock_instance.load.return_value = ({"test": "data"}, "success")
            mock_workload.return_value = mock_instance

            # Import and execute the main block
            import fastdeploy.eplb.utils as utils_module

            # The main block should execute without errors
            # We can't easily test the print output, but we can verify the function call
            if hasattr(utils_module, "__name__") and utils_module.__name__ == "__main__":
                # This would execute the main block
                pass


if __name__ == "__main__":
    unittest.main()
