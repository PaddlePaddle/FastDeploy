import unittest
import logging
from unittest.mock import patch

from fastdeploy import envs
from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
)


class TestConfig(unittest.TestCase):
    def test_fdconfig_nnode(self):
        parallel_config = ParallelConfig({"tensor_parallel_size": 16, "expert_parallel_size": 1})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            ips=["1.1.1.1", "0.0.0.0"],
            test_mode=True,
        )
        assert fd_config.nnode == 2
        assert fd_config.is_master is False

    def test_fdconfig_ips(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        assert fd_config.master_ip == "0.0.0.0"

    def test_fdconfig_max_num_tokens(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        cache_config.enable_chunked_prefill = True
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            assert fd_config.max_num_batched_tokens == 2048

        cache_config.enable_chunked_prefill = False
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            ips="0.0.0.0",
            test_mode=True,
        )
        if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            assert fd_config.max_num_batched_tokens == 8192

    def test_fdconfig_init_cache(self):
        parallel_config = ParallelConfig({})
        graph_opt_config = GraphOptimizationConfig({})
        cache_config = CacheConfig({})
        cache_config.cache_transfer_protocol = "rdma,ipc"
        cache_config.pd_comm_port = "2334"
        fd_config = FDConfig(
            parallel_config=parallel_config,
            graph_opt_config=graph_opt_config,
            cache_config=cache_config,
            splitwise_role="prefill",
            test_mode=True,
        )
        fd_config.init_cache_info()
        assert fd_config.disaggregate_info["role"] == "prefill"

    def test_gpu_memory_utilization_warning(self):
        """Test that a warning is issued when gpu_memory_utilization >= 0.95"""
        with patch('fastdeploy.utils.console_logger') as mock_logger:
            # Test case 1: gpu_memory_utilization = 0.95 should trigger warning
            cache_config = CacheConfig({"gpu_memory_utilization": 0.95})
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("0.95", warning_call)
            self.assertIn("out-of-memory", warning_call)
            self.assertIn("below 0.9", warning_call)
            
            # Reset mock
            mock_logger.reset_mock()
            
            # Test case 2: gpu_memory_utilization = 0.99 should trigger warning
            cache_config = CacheConfig({"gpu_memory_utilization": 0.99})
            mock_logger.warning.assert_called_once()
            
            # Reset mock
            mock_logger.reset_mock()
            
            # Test case 3: gpu_memory_utilization = 0.9 should NOT trigger warning
            cache_config = CacheConfig({"gpu_memory_utilization": 0.9})
            mock_logger.warning.assert_not_called()
            
            # Reset mock
            mock_logger.reset_mock()
            
            # Test case 4: gpu_memory_utilization = 0.8 should NOT trigger warning
            cache_config = CacheConfig({"gpu_memory_utilization": 0.8})
            mock_logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
