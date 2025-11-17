"""
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
"""

import unittest
from unittest.mock import Mock

from fastdeploy import envs
from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ParallelConfig,
    SchedulerConfig,
)
from fastdeploy.utils import get_host_ip


class TestConfig(unittest.TestCase):
    def test_fdconfig_nnode(self):
        parallel_config = ParallelConfig({"tensor_parallel_size": 16, "expert_parallel_size": 1})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        model_config = Mock()
        model_config.max_model_len = 512
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            ips=[get_host_ip(), "0.0.0.0"],
            test_mode=True,
        )
        assert fd_config.nnode == 2
        assert fd_config.is_master is True

    def test_fdconfig_ips(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        model_config = Mock()
        model_config.max_model_len = 512
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        assert fd_config.master_ip == "0.0.0.0"

    def test_fdconfig_max_num_tokens(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        load_config = LoadConfig({})
        cache_config.enable_chunked_prefill = True
        scheduler_config = SchedulerConfig({})
        model_config: Mock = Mock()
        model_config.max_model_len = 512

        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            assert fd_config.scheduler_config.max_num_batched_tokens == 2048

        cache_config.enable_chunked_prefill = False
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            assert fd_config.scheduler_config.max_num_batched_tokens == 8192

    def test_fdconfig_init_cache(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        cache_config.cache_transfer_protocol = "rdma,ipc"
        cache_config.pd_comm_port = "2334"
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        scheduler_config.splitwise_role = "prefill"
        model_config: Mock = Mock()
        model_config.max_model_len = 512

        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            test_mode=True,
        )
        fd_config.init_cache_info()
        assert fd_config.disaggregate_info["role"] == "prefill"


if __name__ == "__main__":
    unittest.main()
