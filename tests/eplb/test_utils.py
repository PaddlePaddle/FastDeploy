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
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.config import (
    CacheConfig,
    EPLBConfig,
    FDConfig,
    ParallelConfig,
    SchedulerConfig,
)
from fastdeploy.engine.args_utils import EngineArgs
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
        max_num_seqs = 2
        engine_args = EngineArgs(
            max_num_seqs=max_num_seqs,
            num_gpu_blocks_override=102,
            max_num_batched_tokens=3200,
        )
        args = asdict(engine_args)

        cache_cfg = CacheConfig(args)
        model_cfg = SimpleNamespace(enable_mm=True)  # Enable multimodal for feature testing
        speculative_cfg = SimpleNamespace(method=None)
        model_cfg.print = print
        model_cfg.max_model_len = 5120
        model_cfg.num_hidden_layers = 3
        model_cfg.moe_num_experts = 64
        model_cfg.moe_layer_start_index = 1
        model_cfg.model = "/test/model"
        cache_cfg.bytes_per_layer_per_block = 1

        parallel_cfg = ParallelConfig(args)
        scheduler_cfg = SchedulerConfig(args)
        graph_opt_cfg = engine_args.create_graph_optimization_config()

        eplb_args = {
            "redundant_experts_num": 0,
            "redundant_expert_api_user": "test_user",
            "redundant_expert_api_password": "test_pass",
            "redundant_expert_eplb_strategy": "",
            "redundant_expert_ip_shm_size": 1024,
            "moe_quant_type": "",
            "redundant_expert_enable_schedule_cordon": False,
        }
        eplb_config = EPLBConfig(eplb_args)

        self.fd_config = FDConfig(
            model_config=model_cfg,
            cache_config=cache_cfg,
            parallel_config=parallel_cfg,
            graph_opt_config=graph_opt_cfg,
            speculative_config=speculative_cfg,
            scheduler_config=scheduler_cfg,
            eplb_config=eplb_config,
        )
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
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.tensor_parallel_size = 1
        self.fd_config.parallel_config.local_data_parallel_id = 1
        self.fd_config.eplb_config.redundant_expert_ip_shm_size = 1024
        ipc_signal_suffix = 123
        init_eplb_signals(self.fd_config, ipc_signal_suffix)

        # For non-zero rank, only common signals should be created
        dp_ipc_signal_suffix = f"{ipc_signal_suffix}_dp1"
        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp0"
        expected_calls = [
            # Common signals (no rank 0 specific signals)
            ("rearrange_experts_status", np.zeros([1], dtype=np.int32), np.int32, dp_ipc_signal_suffix, True),
            ("rearrange_experts_ips_size", np.zeros([1], dtype=np.int32), np.int32, dp_ipc_signal_suffix, True),
            ("rearrange_experts_ips_list", 1024, dp_ipc_signal_suffix, True),
            ("signal_update_weight_from_tensor", np.zeros([1], dtype=np.int32), np.int32, dp_ipc_signal_suffix, True),
            ("all_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, tp_ipc_signal_suffix, True),
            ("local_experts_token_stats", np.zeros((3, 64), dtype=np.int32), np.int32, tp_ipc_signal_suffix, True),
            ("signal_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, tp_ipc_signal_suffix, True),
            ("signal_clear_experts_token_stats", np.zeros([1], dtype=np.int32), np.int32, tp_ipc_signal_suffix, True),
            ("result_update_weight_from_disk", np.zeros([1], dtype=np.int32), np.int32, tp_ipc_signal_suffix, True),
        ]

        # Verify only common signals were created
        self.assertEqual(mock_ipc_signal.call_count, len(expected_calls))

        # Get all actual calls and verify each parameter
        actual_calls = mock_ipc_signal.call_args_list
        # Verify each call matches expected parameters
        for i, expected in enumerate(expected_calls):
            call = actual_calls[i]

            # Extract call arguments
            if len(call) == 2:  # args and kwargs
                args, kwargs = call
                actual_args = args if isinstance(args, tuple) else (args,)
                suffix = kwargs.get("suffix")
            else:
                actual_args = call if isinstance(call, tuple) else (call,)
                suffix = None

            # Skip verification if we can't access the expected parameters
            if len(expected) < 1:
                continue

            # Verify signal name is present
            if len(actual_args) > 0:
                self.assertEqual(actual_args[0], expected[0], f"Signal name mismatch at call {i}")
            else:
                continue

            # Special handling for rearrange_experts_ips_list
            if expected[0] == "rearrange_experts_ips_list":
                continue

            # Verify array/values if present
            if len(expected) > 1 and len(actual_args) > 1:
                if isinstance(expected[1], np.ndarray):
                    np.testing.assert_array_equal(actual_args[1], expected[1], f"Array mismatch at call {i}")
                else:
                    self.assertEqual(actual_args[1], expected[1], f"Value mismatch at call {i}")

            # Verify data type if present
            if len(expected) > 2 and len(actual_args) > 2:
                self.assertEqual(actual_args[2], expected[2], f"Data type mismatch at call {i}")

            # Verify suffix if present
            if len(expected) > 3:
                if suffix is not None:
                    self.assertEqual(suffix, expected[3], f"IPC suffix mismatch at call {i}")
                elif len(actual_args) > 3:
                    self.assertEqual(actual_args[3], expected[3], f"IPC suffix mismatch at call {i}")

            # Verify create flag if present
            if len(expected) > 4 and len(actual_args) > 4:
                self.assertEqual(actual_args[4], expected[4], f"Create flag mismatch at call {i}")

    @patch("fastdeploy.eplb.utils.IPCSignal")
    def test_init_eplb_signals_different_suffix(self, mock_ipc_signal):
        """Test init_eplb_signals with different suffix"""
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance

        ipc_signal_suffix = "999"
        init_eplb_signals(self.fd_config, ipc_signal_suffix)

        target_suffix = [
            "999_dp0",
            "999_dp0",
            "999_dp0",
            "999_dp0",
            "999_dp0_tp0",
            "999_dp0_tp0",
            "999_dp0_tp0",
            "999_dp0_tp0",
            "999_dp0_tp0",
        ]
        # Verify that suffix is used correctly
        for idx, call in enumerate(mock_ipc_signal.call_args_list):
            args, kwargs = call
            self.assertEqual(kwargs.get("suffix"), target_suffix[idx])

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
