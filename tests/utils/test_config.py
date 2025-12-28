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

import random
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
        model_config.architectures = ["test_model"]
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
        model_config.architectures = ["test_model"]
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
        model_config.architectures = ["test_model"]

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
        model_config.architectures = ["test_model"]

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
        assert fd_config.register_info is not None

    def test_fdconfig_postprocess_ports(self):
        data_parallel_size = 4
        tensor_parallel_size = 2
        local_data_parallel_id = random.randint(0, data_parallel_size - 1)
        engine_worker_queue_ports = [random.randint(8000, 65535) for _ in range(data_parallel_size)]
        cache_queue_ports = [random.randint(8000, 65535) for _ in range(data_parallel_size)]
        pd_comm_ports = [random.randint(8000, 65535) for _ in range(data_parallel_size)]
        rdma_comm_ports = [random.randint(8000, 65535) for _ in range(data_parallel_size * tensor_parallel_size)]

        parallel_config = ParallelConfig(
            {
                "engine_worker_queue_port": ",".join(map(str, engine_worker_queue_ports)),
                "data_parallel_size": data_parallel_size,
                "tensor_parallel_size": tensor_parallel_size,
                "local_data_parallel_id": local_data_parallel_id,
            }
        )
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig(
            {
                "cache_queue_port": ",".join(map(str, cache_queue_ports)),
                "pd_comm_port": ",".join(map(str, pd_comm_ports)),
                "rdma_comm_ports": ",".join(map(str, rdma_comm_ports)),
            }
        )
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        model_config: Mock = Mock()
        model_config.max_model_len = 512
        model_config.architectures = ["test_model"]

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
        assert (
            fd_config.parallel_config.local_engine_worker_queue_port
            == engine_worker_queue_ports[local_data_parallel_id]
        )
        assert fd_config.cache_config.local_cache_queue_port == cache_queue_ports[local_data_parallel_id]
        assert fd_config.cache_config.local_pd_comm_port == pd_comm_ports[local_data_parallel_id]
        assert (
            fd_config.cache_config.local_rdma_comm_ports
            == rdma_comm_ports[
                local_data_parallel_id * tensor_parallel_size : (local_data_parallel_id + 1) * tensor_parallel_size
            ]
        )


if __name__ == "__main__":
    unittest.main()
