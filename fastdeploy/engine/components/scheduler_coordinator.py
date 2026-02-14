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
Scheduler Coordinator - Manages request scheduling and dispatch.

This component is part of the new modular architecture and handles:
- Scheduler initialization
- Request scheduling loop
- Task dispatch to workers
- Result collection
- Data processor management
"""

from typing import TYPE_CHECKING, Optional

from fastdeploy.model_executor.guided_decoding import schema_checker
from fastdeploy.plugins.token_processor import load_token_processor_plugins
from fastdeploy.utils import envs, llm_logger

if TYPE_CHECKING:
    from fastdeploy.engine.sched.scheduler import BaseScheduler
    from fastdeploy.input.preprocess import InputPreprocessor


class SchedulerCoordinator:
    """
    Manages request scheduling and dispatch for the engine.

    This component is used in the new modular architecture.
    The old architecture (_use_new_architecture=False) uses the original EngineService code.
    """

    def __init__(self, cfg, resource_coordinator, ipc_manager):
        """
        Initialize scheduler coordinator.

        Args:
            cfg: Configuration object
            resource_coordinator: ResourceCoordinator instance
            ipc_manager: IPCManager instance
        """
        self.cfg = cfg
        self.resource = resource_coordinator
        self.ipc = ipc_manager

        self._scheduler: Optional["BaseScheduler"] = None
        self._data_processor: Optional["InputPreprocessor"] = None

        # Partial chunked tokens configuration
        self.partial_chunked_tokens = []

        # Metrics logger (shared with resource coordinator)
        self.scheduler_metrics_logger = None

        # Token processor
        self.token_processor = None

        # Structured outputs checker
        self.guided_decoding_checker = None

        # Split connector
        self.split_connector = None

        # BOS client and multimodal config
        self.bos_client = None
        self.mm_max_tokens_per_item = None

        self.llm_logger = llm_logger
        self.enable_decode_cache_task = envs.FD_ENABLE_CACHE_TASK == "1"

    def init_scheduler(self):
        """
        Initialize the scheduler.
        Corresponds to EngineService scheduler initialization
        """
        self._scheduler = self.cfg.scheduler_config.scheduler()

    def init_data_processor(self):
        """
        Initialize the data processor.
        Corresponds to EngineService.create_data_processor()
        """
        from fastdeploy.input.preprocess import InputPreprocessor

        # Create input processor with proper parameters matching old architecture
        self._input_processor = InputPreprocessor(
            self.cfg.model_config,
            self.cfg.structured_outputs_config.reasoning_parser,
            self.cfg.limit_mm_per_prompt,
            self.cfg.mm_processor_kwargs,
            self.cfg.tool_parser,
        )
        # Create the actual data processor
        self._data_processor = self._input_processor.create_processor()

        # Initialize multimodal if needed
        if self.cfg.model_config.enable_mm:
            self._data_processor.init_mm_processor()

        # Initialize guided decoding checker
        if self.cfg.structured_outputs_config.guided_decoding_backend != "off":
            self.guided_decoding_checker = schema_checker(
                self.cfg.structured_outputs_config.guided_decoding_backend,
                disable_any_whitespace=self.cfg.structured_outputs_config.disable_any_whitespace,
            )

    def init_token_processor(self):
        """
        Initialize the token processor.
        """
        from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector

        try:
            TokenProcessor = load_token_processor_plugins()
            self.llm_logger.info(f"TokenProcessor plugin {TokenProcessor} loaded")
        except:
            from fastdeploy.output.token_processor import TokenProcessor

        self.split_connector = SplitwiseConnector(
            self.cfg, self.ipc.engine_worker_queue, self.resource.resource_manager
        )
        self.token_processor = TokenProcessor(
            cfg=self.cfg,
            cached_generated_tokens=self._scheduler,
            engine_worker_queue=self.ipc.engine_worker_queue,
            split_connector=self.split_connector,
        )
        self.token_processor.set_resource_manager(self.resource.resource_manager)
        self.token_processor.set_scheduler_metrics_logger(self.resource.scheduler_metrics_logger)

    def init_partial_chunked_tokens(self):
        """Initialize partial chunked tokens configuration."""
        self.partial_chunked_tokens = [0] * (self.cfg.max_num_partial_prefills + 1)
        for idx in range(1, self.cfg.max_num_partial_prefills + 1):
            self.partial_chunked_tokens[idx] = (
                (self.cfg.scheduler_config.max_num_batched_tokens // idx)
                // self.cfg.cache_config.block_size
                * self.cfg.cache_config.block_size
            )

    def start(self):
        """Start scheduler services."""
        self.init_scheduler()
        # Schedule loop will be started separately by EngineService.start()
        # The loop methods (_schedule_request_to_worker, etc.) remain in EngineService
        # until they can be properly isolated.

    def stop(self):
        """Stop scheduler services."""
        # TokenProcessor doesn't have a stop method in the current implementation
        # This will be handled at the adapter level if needed

    @property
    def scheduler(self) -> Optional["BaseScheduler"]:
        """Get the scheduler instance."""
        return self._scheduler

    @property
    def data_processor(self) -> Optional["InputPreprocessor"]:
        """Get the data processor instance."""
        return self._data_processor
