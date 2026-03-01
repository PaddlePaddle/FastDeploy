# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
from unittest.mock import Mock, patch

from fastdeploy.config import FDConfig
from fastdeploy.scheduler.config import SchedulerConfig


def _create_mock_configs():
    """Create all mock config objects needed for FDConfig initialization."""
    # Mock scheduler_config
    mock_scheduler = Mock(spec=SchedulerConfig)
    mock_scheduler.max_num_batched_tokens = None
    mock_scheduler.max_num_seqs = 34
    mock_scheduler.splitwise_role = "mixed"
    mock_scheduler.name = "local"
    mock_scheduler.max_extra_num_batched_tokens = 16384
    mock_scheduler.enable_overlap_schedule = False

    # Mock model_config
    mock_model = Mock()
    mock_model.max_model_len = 8192
    mock_model.architectures = ["TestModel"]
    mock_model.enable_mm = False
    mock_model.is_reasoning_model = False
    mock_model.mm_max_tokens_per_item = None
    mock_model.moe_phase = None

    # Mock cache_config
    mock_cache = Mock()
    mock_cache.enable_prefix_caching = False
    mock_cache.block_size = 64
    mock_cache.enable_chunked_prefill = False
    mock_cache.max_block_num_per_seq = 128
    mock_cache.cache_queue_port = None
    mock_cache.pd_comm_port = None
    mock_cache.rdma_comm_ports = None
    mock_cache.max_encoder_cache = 0
    mock_cache.postprocess = Mock()

    # Mock parallel_config
    mock_parallel = Mock()
    mock_parallel.tensor_parallel_size = 1
    mock_parallel.data_parallel_size = 1
    mock_parallel.expert_parallel_size = 1
    mock_parallel.local_data_parallel_id = 0
    mock_parallel.engine_worker_queue_port = [8080]
    mock_parallel.local_engine_worker_queue_port = 8080
    mock_parallel.device_ids = "0"
    mock_parallel.use_sequence_parallel_moe = False

    # Mock load_config
    mock_load = Mock()
    mock_load.load_strategy = "normal"
    mock_load.dynamic_load_weight = False

    # Mock graph_opt_config
    mock_graph = Mock()
    mock_graph.use_cudagraph = False
    mock_graph.cudagraph_capture_sizes = None
    mock_graph.max_capture_shape_prefill = 512
    mock_graph.graph_opt_level = 0
    mock_graph.cudagraph_only_prefill = False
    mock_graph.filter_capture_size = Mock()

    return mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph


def _create_fd_config_instance(mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph):
    """Create an FDConfig instance with the given mock configs."""
    fd_config = FDConfig.__new__(FDConfig)
    fd_config.model_config = mock_model
    fd_config.cache_config = mock_cache
    fd_config.scheduler_config = mock_scheduler
    fd_config.parallel_config = mock_parallel
    fd_config.load_config = mock_load
    fd_config.graph_opt_config = mock_graph
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
    return fd_config


@contextlib.contextmanager
def _patch_env_and_config(enable_v1_scheduler):
    """Context manager to patch all environment variables and config methods."""
    from fastdeploy import envs as fastdeploy_envs

    env_patches = [
        patch.object(fastdeploy_envs, "ENABLE_V1_KVCACHE_SCHEDULER", enable_v1_scheduler),
        patch.object(fastdeploy_envs, "FD_ENABLE_MAX_PREFILL", False),
        patch.object(fastdeploy_envs, "FD_FOR_TORCH_MODEL_FORMAT", False),
        patch.object(fastdeploy_envs, "FD_MAX_STOP_SEQS_NUM", 10),
        patch.object(fastdeploy_envs, "FD_STOP_SEQS_MAX_LEN", 100),
        patch("fastdeploy.config.envs.ENABLE_V1_KVCACHE_SCHEDULER", enable_v1_scheduler),
    ]

    with contextlib.ExitStack() as stack:
        for p in env_patches:
            stack.enter_context(p)
        stack.enter_context(patch.object(FDConfig, "_disable_sequence_parallel_moe_if_needed"))
        yield


class TestSchedulerConfigMaxNumBatchedTokens(unittest.TestCase):
    """Test cases for scheduler_config.max_num_batched_tokens assignment logic."""

    def test_max_num_batched_tokens_set_to_8192_when_v1_scheduler_enabled(self):
        """
        Test that max_num_batched_tokens is set to 8192 when:
        1. scheduler_config.max_num_batched_tokens is None
        2. ENABLE_V1_KVCACHE_SCHEDULER is enabled (value is truthy)

        This test covers the line:
        self.scheduler_config.max_num_batched_tokens = 8192
        """
        mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph = _create_mock_configs()

        with _patch_env_and_config(enable_v1_scheduler=1):
            fd_config = _create_fd_config_instance(
                mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph
            )
            fd_config.postprocess()

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
        mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph = _create_mock_configs()
        original_value = 4096
        mock_scheduler.max_num_batched_tokens = original_value

        with _patch_env_and_config(enable_v1_scheduler=1):
            fd_config = _create_fd_config_instance(
                mock_scheduler, mock_model, mock_cache, mock_parallel, mock_load, mock_graph
            )
            fd_config.postprocess()

            self.assertEqual(
                fd_config.scheduler_config.max_num_batched_tokens,
                original_value,
                "max_num_batched_tokens should not be overwritten when already set",
            )


if __name__ == "__main__":
    unittest.main()
