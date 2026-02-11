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
Tests for FDConfig and scheduler configuration, specifically for
max_num_batched_tokens assignment when ENABLE_V1_KVCACHE_SCHEDULER is enabled.
"""

import unittest
from unittest.mock import Mock, patch


class TestSchedulerConfigMaxNumBatchedTokens(unittest.TestCase):
    """Test cases for scheduler_config.max_num_batched_tokens assignment logic."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to ensure we can patch envs before other imports
        from fastdeploy import envs

        self.envs = envs

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_max_num_batched_tokens_set_to_8192_when_v1_scheduler_enabled(self):
        """
        Test that max_num_batched_tokens is set to 8192 when:
        1. scheduler_config.max_num_batched_tokens is None
        2. ENABLE_V1_KVCACHE_SCHEDULER is enabled (value is truthy)

        This test covers the line:
        self.scheduler_config.max_num_batched_tokens = 8192
        """
        from fastdeploy.config import FDConfig
        from fastdeploy.scheduler.config import SchedulerConfig

        # Create a mock scheduler_config with max_num_batched_tokens = None
        mock_scheduler_config = Mock(spec=SchedulerConfig)
        mock_scheduler_config.max_num_batched_tokens = None
        mock_scheduler_config.max_num_seqs = 34
        mock_scheduler_config.splitwise_role = "mixed"
        mock_scheduler_config.name = "local"
        mock_scheduler_config.max_extra_num_batched_tokens = 16384
        mock_scheduler_config.enable_overlap_schedule = False

        # Create necessary mock configs
        mock_model_config = Mock()
        mock_model_config.max_model_len = 8192
        mock_model_config.architectures = ["TestModel"]
        mock_model_config.enable_mm = False
        mock_model_config.is_reasoning_model = False
        mock_model_config.mm_max_tokens_per_item = None
        mock_model_config.moe_phase = None

        mock_cache_config = Mock()
        mock_cache_config.enable_prefix_caching = False
        mock_cache_config.block_size = 64
        mock_cache_config.enable_chunked_prefill = False
        mock_cache_config.max_block_num_per_seq = 128
        mock_cache_config.cache_queue_port = None
        mock_cache_config.pd_comm_port = None
        mock_cache_config.rdma_comm_ports = None
        mock_cache_config.max_encoder_cache = 0
        mock_cache_config.postprocess = Mock()

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_size = 1
        mock_parallel_config.data_parallel_size = 1
        mock_parallel_config.expert_parallel_size = 1
        mock_parallel_config.local_data_parallel_id = 0
        mock_parallel_config.engine_worker_queue_port = [8080]
        mock_parallel_config.local_engine_worker_queue_port = 8080
        mock_parallel_config.device_ids = "0"
        mock_parallel_config.use_sequence_parallel_moe = False

        mock_load_config = Mock()
        mock_load_config.load_strategy = "normal"
        mock_load_config.dynamic_load_weight = False

        mock_graph_opt_config = Mock()
        mock_graph_opt_config.use_cudagraph = False
        mock_graph_opt_config.cudagraph_capture_sizes = None
        mock_graph_opt_config.max_capture_shape_prefill = 512
        mock_graph_opt_config.graph_opt_level = 0
        mock_graph_opt_config.cudagraph_only_prefill = False
        mock_graph_opt_config.filter_capture_size = Mock()

        # Patch ENABLE_V1_KVCACHE_SCHEDULER to be enabled (truthy value)
        with patch.object(self.envs, "ENABLE_V1_KVCACHE_SCHEDULER", 1):
            with patch.object(self.envs, "FD_ENABLE_MAX_PREFILL", False):
                with patch.object(self.envs, "FD_FOR_TORCH_MODEL_FORMAT", False):
                    with patch.object(self.envs, "FD_MAX_STOP_SEQS_NUM", 10):
                        with patch.object(self.envs, "FD_STOP_SEQS_MAX_LEN", 100):
                            # Also patch at the location where it's used in config.py
                            with patch("fastdeploy.config.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
                                # Patch the disable_chunked_mm_input attribute check
                                with patch.object(FDConfig, "_disable_sequence_parallel_moe_if_needed"):
                                    # Create FDConfig with test_mode=True to skip full initialization
                                    fd_config = FDConfig.__new__(FDConfig)
                                    fd_config.model_config = mock_model_config
                                    fd_config.cache_config = mock_cache_config
                                    fd_config.scheduler_config = mock_scheduler_config
                                    fd_config.parallel_config = mock_parallel_config
                                    fd_config.load_config = mock_load_config
                                    fd_config.graph_opt_config = mock_graph_opt_config
                                    fd_config.speculative_config = None
                                    fd_config.eplb_config = None
                                    fd_config.structured_outputs_config = None
                                    fd_config.router_config = None
                                    fd_config.nnode = 1
                                    fd_config.node_rank = 0
                                    fd_config.worker_num_per_node = 1
                                    fd_config.master_ip = "127.0.0.1"
                                    fd_config.is_master = True
                                    fd_config.max_num_partial_prefills = 1
                                    fd_config.max_long_partial_prefills = 1
                                    fd_config.long_prefill_token_threshold = 0
                                    fd_config.paddle_commit_id = "test"
                                    fd_config.routing_replay_config = None

                                    # Call postprocess to trigger the assignment
                                    fd_config.postprocess()

                                    # Verify that max_num_batched_tokens was set to 8192
                                    self.assertEqual(
                                        fd_config.scheduler_config.max_num_batched_tokens,
                                        8192,
                                        "max_num_batched_tokens should be set to 8192 when "
                                        "ENABLE_V1_KVCACHE_SCHEDULER is enabled and value is None",
                                    )

    def test_max_num_batched_tokens_not_overwritten_when_already_set(self):
        """
        Test that max_num_batched_tokens is NOT overwritten when it already has a value.

        This test ensures that if max_num_batched_tokens is explicitly set to a non-None value,
        it should not be changed by the postprocess method.
        """
        from fastdeploy.config import FDConfig
        from fastdeploy.scheduler.config import SchedulerConfig

        # Create a mock scheduler_config with max_num_batched_tokens already set
        original_value = 4096
        mock_scheduler_config = Mock(spec=SchedulerConfig)
        mock_scheduler_config.max_num_batched_tokens = original_value
        mock_scheduler_config.max_num_seqs = 34
        mock_scheduler_config.splitwise_role = "mixed"
        mock_scheduler_config.name = "local"
        mock_scheduler_config.max_extra_num_batched_tokens = 16384
        mock_scheduler_config.enable_overlap_schedule = False

        # Create necessary mock configs
        mock_model_config = Mock()
        mock_model_config.max_model_len = 8192
        mock_model_config.architectures = ["TestModel"]
        mock_model_config.enable_mm = False
        mock_model_config.is_reasoning_model = False
        mock_model_config.mm_max_tokens_per_item = None
        mock_model_config.moe_phase = None

        mock_cache_config = Mock()
        mock_cache_config.enable_prefix_caching = False
        mock_cache_config.block_size = 64
        mock_cache_config.enable_chunked_prefill = False
        mock_cache_config.max_block_num_per_seq = 128
        mock_cache_config.cache_queue_port = None
        mock_cache_config.pd_comm_port = None
        mock_cache_config.rdma_comm_ports = None
        mock_cache_config.max_encoder_cache = 0
        mock_cache_config.postprocess = Mock()

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_size = 1
        mock_parallel_config.data_parallel_size = 1
        mock_parallel_config.expert_parallel_size = 1
        mock_parallel_config.local_data_parallel_id = 0
        mock_parallel_config.engine_worker_queue_port = [8080]
        mock_parallel_config.local_engine_worker_queue_port = 8080
        mock_parallel_config.device_ids = "0"
        mock_parallel_config.use_sequence_parallel_moe = False

        mock_load_config = Mock()
        mock_load_config.load_strategy = "normal"
        mock_load_config.dynamic_load_weight = False

        mock_graph_opt_config = Mock()
        mock_graph_opt_config.use_cudagraph = False
        mock_graph_opt_config.cudagraph_capture_sizes = None
        mock_graph_opt_config.max_capture_shape_prefill = 512
        mock_graph_opt_config.graph_opt_level = 0
        mock_graph_opt_config.cudagraph_only_prefill = False
        mock_graph_opt_config.filter_capture_size = Mock()

        # Patch ENABLE_V1_KVCACHE_SCHEDULER to be enabled
        with patch.object(self.envs, "ENABLE_V1_KVCACHE_SCHEDULER", 1):
            with patch.object(self.envs, "FD_ENABLE_MAX_PREFILL", False):
                with patch.object(self.envs, "FD_FOR_TORCH_MODEL_FORMAT", False):
                    with patch.object(self.envs, "FD_MAX_STOP_SEQS_NUM", 10):
                        with patch.object(self.envs, "FD_STOP_SEQS_MAX_LEN", 100):
                            with patch("fastdeploy.config.envs.ENABLE_V1_KVCACHE_SCHEDULER", 1):
                                with patch.object(FDConfig, "_disable_sequence_parallel_moe_if_needed"):
                                    fd_config = FDConfig.__new__(FDConfig)
                                    fd_config.model_config = mock_model_config
                                    fd_config.cache_config = mock_cache_config
                                    fd_config.scheduler_config = mock_scheduler_config
                                    fd_config.parallel_config = mock_parallel_config
                                    fd_config.load_config = mock_load_config
                                    fd_config.graph_opt_config = mock_graph_opt_config
                                    fd_config.speculative_config = None
                                    fd_config.eplb_config = None
                                    fd_config.structured_outputs_config = None
                                    fd_config.router_config = None
                                    fd_config.nnode = 1
                                    fd_config.node_rank = 0
                                    fd_config.worker_num_per_node = 1
                                    fd_config.master_ip = "127.0.0.1"
                                    fd_config.is_master = True
                                    fd_config.max_num_partial_prefills = 1
                                    fd_config.max_long_partial_prefills = 1
                                    fd_config.long_prefill_token_threshold = 0
                                    fd_config.paddle_commit_id = "test"
                                    fd_config.routing_replay_config = None

                                    fd_config.postprocess()

                                    # Verify that max_num_batched_tokens was NOT changed
                                    self.assertEqual(
                                        fd_config.scheduler_config.max_num_batched_tokens,
                                        original_value,
                                        "max_num_batched_tokens should not be overwritten when already set",
                                    )


if __name__ == "__main__":
    unittest.main()
