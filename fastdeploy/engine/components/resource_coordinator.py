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
Resource Coordinator - Manages KV cache and block allocation.

This component is part of the new modular architecture and handles:
- Resource manager initialization
- Block allocation and deallocation
- Cache configuration management
"""

from typing import Optional

from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1
from fastdeploy.engine.sched.scheduler_metrics_logger import SchedulerMetricsLogger
from fastdeploy.utils import envs, llm_logger


class ResourceCoordinator:
    """
    Manages KV cache resources for the engine.

    This component is used in the new modular architecture.
    The old architecture (_use_new_architecture=False) uses the original EngineService code.
    """

    def __init__(self, cfg):
        """
        Initialize resource coordinator.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self._resource_manager: Optional[ResourceManager] = None
        self.cache_config = cfg.cache_config
        self.scheduler_metrics_logger: Optional[SchedulerMetricsLogger] = None
        self.llm_logger = llm_logger

    def init_resource_manager(self, dp_rank: int = 0):
        """
        Initialize the resource manager.
        Corresponds to EngineService resource manager initialization
        """
        self.scheduler_metrics_logger = SchedulerMetricsLogger(
            enabled=True,
            dp_rank=dp_rank,
        )

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.llm_logger.info("Use V1 KVCache Scheduler")
            self._resource_manager = ResourceManagerV1(
                self.cfg.scheduler_config.max_num_seqs,
                self.cfg,
                self.cfg.parallel_config.tensor_parallel_size,
                self.cfg.scheduler_config.splitwise_role,
                self.cfg.parallel_config.local_data_parallel_id,
            )
        else:
            self.llm_logger.info("Use V0 KVCache Scheduler")
            self._resource_manager = ResourceManager(
                self.cfg.scheduler_config.max_num_seqs,
                self.cfg,
                self.cfg.parallel_config.tensor_parallel_size,
                self.cfg.scheduler_config.splitwise_role,
                self.cfg.parallel_config.local_data_parallel_id,
            )

        # Set metrics logger
        self._resource_manager.scheduler_metrics_logger = self.scheduler_metrics_logger

    def start(self, dp_rank: int = 0):
        """
        Start resource coordinator services.

        Args:
            dp_rank: Data parallel rank
        """
        if self._resource_manager is None:
            self.init_resource_manager(dp_rank)

    def available_block_num(self) -> int:
        """
        Get the number of available KV cache blocks.

        Returns:
            Number of available blocks
        """
        if self._resource_manager is None:
            return 0
        return self._resource_manager.available_block_num()

    def check_and_free_block_tables(self):
        """Free block tables for completed requests."""
        if self._resource_manager is not None:
            self._resource_manager.check_and_free_block_tables()

    def reset_cache_config(self, cache_config):
        """
        Reset the cache configuration.

        Args:
            cache_config: New cache configuration
        """
        if self._resource_manager is not None:
            self._resource_manager.reset_cache_config(cache_config)

    @property
    def resource_manager(self) -> Optional[ResourceManager]:
        """Get the resource manager instance."""
        return self._resource_manager

    @property
    def real_bsz(self) -> int:
        """Get real batch size."""
        if self._resource_manager is not None:
            return self._resource_manager.real_bsz
        return 1
