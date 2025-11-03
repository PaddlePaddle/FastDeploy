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

from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelRunnerOutput


class WorkerBase(ABC):
    """
    Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
    Worker interface that allows inference framework to cleanly separate implementations for different hardware.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        local_rank: int,
        rank: int,
    ) -> None:
        """
        Initizalize common worker components.

        Args:
             fd_config:
             local_rank:
             rank:
        """
        # Set Configuration
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.load_config = fd_config.load_config
        self.parallel_config = fd_config.parallel_config
        self.device_config = fd_config.device_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        # ... config

        # Device and Runner
        self.device: Optional[str]  # gpu, xpu ...
        self.local_rank = local_rank
        self.rank = rank
        self.model_runner: Optional[ModelRunnerBase]

    @abstractmethod
    def init_device(self) -> None:
        """Initialize the device state."""
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initizlize the KV Cache with the given size in blocks."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """Get the model loaded by worker."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """load model from local or remote"""
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
        model_forward_batch=None,
    ) -> Optional[ModelRunnerOutput]:
        """ """
        raise NotImplementedError

    @abstractmethod
    def graph_optimize_and_warm_up_model(self) -> None:
        """Prepare model for execution through graph optimizaiton(CudaGrpah/CINN) or warmup."""
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return NotImplementedError

    def exist_prefill(self):
        """check whether prefill stage exist."""
        return True
