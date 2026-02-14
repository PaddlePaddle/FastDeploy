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
Process Manager - Manages worker processes lifecycle.

This component is part of the new modular architecture and handles:
- Worker process startup
- Cache process startup
- Process health monitoring
- Process cleanup and restart
"""

from multiprocessing import Process
from typing import List, Optional

from fastdeploy.utils import llm_logger


class ProcessManager:
    """
    Manages worker processes for the engine.

    This component is used in the new modular architecture.
    The old architecture (_use_new_architecture=False) uses the original EngineService code.
    """

    def __init__(self, cfg, ipc_manager):
        """
        Initialize process manager.

        Args:
            cfg: Configuration object
            ipc_manager: IPCManager instance for communication setup
        """
        self.cfg = cfg
        self.ipc = ipc_manager

        self._worker_procs: List[Process] = []
        self._cache_procs: List[Process] = []
        self._running = False

        # Process configuration
        self.ipc_signal_suffix = None
        self.cache_manager_processes = None

        # Worker management attributes
        self.do_profile = 1 if self.cfg.cache_config.num_gpu_blocks_override is None else 0

        self.llm_logger = llm_logger

    def start_workers(self):
        """
        Start all worker processes.
        This is a stub - the actual implementation will be migrated from
        EngineService._start_worker_service() in a future phase.
        """
        # TODO: Migrate from EngineService._start_worker_service()
        # This involves:
        # 1. Setting ipc_signal_suffix
        # 2. Initializing worker signals via ipc_manager
        # 3. Launching components (scheduler, cache_manager, expert_service)
        # 4. Starting cache service (if needed)
        # 5. Launching worker processes via paddle.distributed.launch
        # 6. Waiting for worker initialization and model loading
        pass

    def start_cache_service(self, device_ids: List[str], suffix: int) -> List[Process]:
        """
        Start cache service processes.
        This is a stub - the actual implementation will be migrated from
        EngineService.start_cache_service() in a future phase.

        Args:
            device_ids: List of device IDs for cache processes
            suffix: Suffix for identifying cache processes

        Returns:
            List of started cache processes
        """
        # TODO: Migrate from EngineService.start_cache_service()
        # This involves:
        # 1. Setting environment variables for cache processes
        # 2. Launching cache_manager processes
        # 3. Handling different splitwise roles (prefill/decode/mixed)
        return []

    def stop_workers(self):
        """
        Stop all worker and cache processes.
        This is a stub - the actual implementation will be migrated from
        EngineService cleanup code in a future phase.
        """
        # TODO: Cleanup all processes
        pass

    def check_worker_health(self) -> bool:
        """
        Check the health status of worker processes.
        This is a stub - the actual implementation will be migrated from
        EngineService worker health check code in a future phase.

        Returns:
            True if all workers are healthy, False otherwise
        """
        # TODO: Implement health check logic
        # Based on FD_WORKER_ALIVE_TIMEOUT and worker_healthy_live_signal
        return False

    @property
    def worker_proc(self) -> Optional[Process]:
        """
        Get the first worker process (for compatibility).

        Returns:
            First worker process or None if no workers running
        """
        return self._worker_procs[0] if self._worker_procs else None

    @worker_proc.setter
    def worker_proc(self, value: Optional[Process]):
        """Set worker_proc for compatibility."""
        if self._worker_procs:
            self._worker_procs[0] = value
        else:
            self._worker_procs = [value] if value else []

    @property
    def cache_procs(self) -> List[Process]:
        """Get all cache processes."""
        return self._cache_procs

    @property
    def cache_manager_processes(self) -> Optional[List[Process]]:
        """Get cache manager processes (alias for compatibility)."""
        return self._cache_procs

    @cache_manager_processes.setter
    def cache_manager_processes(self, value: Optional[List[Process]]):
        """Set cache manager processes (alias for compatibility)."""
        self._cache_procs = value or []
