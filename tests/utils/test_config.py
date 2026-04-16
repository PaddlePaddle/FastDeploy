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
        model_config.mm_max_tokens_per_item = None
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
        model_config.mm_max_tokens_per_item = None
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
        model_config.mm_max_tokens_per_item = None

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
        model_config.mm_max_tokens_per_item = None

        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
            model_config=model_config,
            test_mode=True,
        )
        fd_config.init_pd_info()
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
        model_config.mm_max_tokens_per_item = None

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

    def test_fdconfig_get_cache_bytes(self):
        """Test CacheConfig.get_cache_bytes static method for various dtypes."""
        # Test float32/fp32 variants
        for dtype in ["float32", "fp32"]:
            assert CacheConfig.get_cache_bytes(dtype) == 4

        # Test float16/bf16/fp16 variants
        for dtype in ["float16", "bf16", "fp16"]:
            assert CacheConfig.get_cache_bytes(dtype) == 2

        # Test 8-bit types
        for dtype in ["uint8", "int8", "float8", "fp8"]:
            assert CacheConfig.get_cache_bytes(dtype) == 1

        # Test int4
        assert CacheConfig.get_cache_bytes("int4") == 0.5

        # Test unsupported dtype raises ValueError
        with self.assertRaises(ValueError) as ctx:
            CacheConfig.get_cache_bytes("bf11")
        assert "Unsupported cache dtype" in str(ctx.exception)

    def test_fdconfig_num_cpu_blocks(self):
        """Test num_cpu_blocks calculation with swap_space."""
        # Create mock model config with required attributes
        model_config = Mock()
        model_config.num_key_value_heads = 32
        model_config.num_attention_heads = 32
        model_config.head_dim = 128
        model_config.num_hidden_layers = 24
        model_config.quantization = None
        model_config.quantization_config = None

        # Test case 1: swap_space is None -> num_cpu_blocks = 0
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "bfloat16",
                "swap_space": None,
            }
        )
        assert cache_config.num_cpu_blocks == 0

        # Test case 2: swap_space = 1GB
        # bytes_per_block = head_num * head_dim * byte_size * kv_factor * block_size * num_hidden_layers
        #                 = 32 * 128 * 2 * 2 * 64 * 24 = 25165824 bytes
        # num_cpu_blocks = 1 * 1024^3 / 25165824 = 42
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "bfloat16",
                "swap_space": 1,
            }
        )
        expected_blocks = int(1 * 1024**3 / (32 * 128 * 2 * 2 * 64 * 24))
        assert cache_config.num_cpu_blocks == expected_blocks
        assert cache_config.num_cpu_blocks == 42

        # Test case 3: swap_space = 2GB
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "bfloat16",
                "swap_space": 2,
            }
        )
        assert cache_config.num_cpu_blocks == 85

        # Test case 4: with fp32 dtype (4 bytes)
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "float32",
                "swap_space": 1,
            }
        )
        expected_blocks = int(1 * 1024**3 / (32 * 128 * 4 * 2 * 64 * 24))
        assert cache_config.num_cpu_blocks == expected_blocks
        assert cache_config.num_cpu_blocks == 21

        # Test case 5: with int8 dtype (1 byte)
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "int8",
                "swap_space": 1,
            }
        )
        expected_blocks = int(1 * 1024**3 / (32 * 128 * 1 * 2 * 64 * 24))
        assert cache_config.num_cpu_blocks == expected_blocks
        assert cache_config.num_cpu_blocks == 85

        # Test case 6: num_cpu_blocks is explicitly set (not affected by swap_space)
        cache_config = CacheConfig(
            {
                "model_cfg": model_config,
                "cache_dtype": "bfloat16",
                "swap_space": 10,
                "num_cpu_blocks": 100,
            }
        )
        assert cache_config.num_cpu_blocks == 100

        # Test case 7: with num_key_value_heads (GQA)
        model_config_with_gqa = Mock()
        model_config_with_gqa.num_key_value_heads = 8  # GQA
        model_config_with_gqa.num_attention_heads = 32
        model_config_with_gqa.head_dim = 128
        model_config_with_gqa.num_hidden_layers = 24
        model_config_with_gqa.quantization = None
        model_config_with_gqa.quantization_config = None

        cache_config = CacheConfig(
            {
                "model_cfg": model_config_with_gqa,
                "cache_dtype": "bfloat16",
                "swap_space": 1,
            }
        )
        # bytes_per_block = 8 * 128 * 2 * 2 * 64 * 24 = 6291456 bytes
        # num_cpu_blocks = 1 * 1024^3 / 6291456 = 170
        expected_blocks = int(1 * 1024**3 / (8 * 128 * 2 * 2 * 64 * 24))
        assert cache_config.num_cpu_blocks == expected_blocks
        assert cache_config.num_cpu_blocks == 170


if __name__ == "__main__":
    unittest.main()
