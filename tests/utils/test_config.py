import unittest

from fastdeploy import envs
from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ParallelConfig,
    SchedulerConfig,
)


class TestConfig(unittest.TestCase):
    def test_fdconfig_nnode(self):
        parallel_config = ParallelConfig({"tensor_parallel_size": 16, "expert_parallel_size": 1})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            ips=["1.1.1.1", "0.0.0.0"],
            test_mode=True,
        )
        assert fd_config.nnode == 2
        assert fd_config.is_master is False

    def test_fdconfig_ips(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        load_config = LoadConfig({})
        scheduler_config = SchedulerConfig({})
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
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
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
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
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            load_config=load_config,
            scheduler_config=scheduler_config,
            splitwise_role="prefill",
            test_mode=True,
        )
        fd_config.init_cache_info()
        assert fd_config.disaggregate_info["role"] == "prefill"


if __name__ == "__main__":
    unittest.main()
