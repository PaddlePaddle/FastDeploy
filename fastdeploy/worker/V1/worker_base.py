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
from abc import ABC, abstractmethod
from typing import Optional

from paddle import Tensor, nn

from fastdeploy.config import LLMConfig
from fastdeploy.scheduler.scheduler_batch import WorkerBatch
from fastdeploy.worker.output import ModelRunnerOutput
from fastdeploy.worker.V1.model_runner_base import ModelRunnerBase


class WorkerBase(ABC):
    """
        Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
        Worker interface that allows inference framwork to cleanly separate implementations for different harware.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        local_rank: int,
        rank: int,
    ) -> None:
        """
        Initizalize common worker components.

        Args:
             llm_config:
             local_rank:
             rank:
        """
        # Set Configuration
        self.llm_config = llm_config
        self.model_config = llm_config.model_config
        self.cache_config = llm_config.cache_config
        self.lora_config = llm_config.lora_config
        self.load_config = llm_config.load_config
        self.parallel_config = llm_config.parallel_config
        self.device_config = llm_config.device_config
        # ... config

        # Device and Runner
        self.device: Optional[str]  # gpu, xpu ...
        self.local_rank = local_rank
        self.rank = rank
        self.model_runner: Optional[ModelRunnerBase]

    @abstractmethod
    def init_device(self) -> None:
        """ Initialize the device state."""
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initizlize the KV Cache with the given size in blocks."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """ Get the model loaded by worker."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """load model from local or remote"""
        raise NotImplementedError

    @abstractmethod
    def execute_model(
            self, worker_batch: WorkerBatch) -> Optional[ModelRunnerOutput]:
        """ """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_spec(self) -> dict[str, Tensor]:
        """ """
        raise NotImplementedError

    @abstractmethod
    def graph_optimize_and_warm_up_model(self) -> None:
        """Prepare model for execution through grpah optimizaiton(CudaGrpah/CINN) or warmup."""
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return NotImplementedError
