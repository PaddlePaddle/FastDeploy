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

from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.utils import get_logger
from fastdeploy.worker.output import ModelRunnerOutput

logger = get_logger("model_runner_base", "model_runner_base.log")


class ModelRunnerBase(ABC):
    """
    Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
    ModelRunner interface abstracts the model execution logic that
    contain input preparation, token generation, and tokenprocessing.
    """

    def __init__(self, fd_config: FDConfig, device: str) -> None:
        """
        Initialize FDConfig
        """
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.load_config = fd_config.load_config
        self.device_config = fd_config.device_config
        self.speculative_config = fd_config.speculative_config
        self.parallel_config = fd_config.parallel_config
        self.graph_opt_config = fd_config.graph_opt_config
        self.quant_config = fd_config.quant_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        # ... config

        self.device = device

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model from local path or remote(will download) path
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """
        Get current model
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
    ) -> ModelRunnerOutput:
        """
        Execute model with and get output
        """
        raise NotImplementedError

    @abstractmethod
    def profile_run(self) -> None:
        """
        Execute a forward pass with dummy inputs to profile the memory usage of the model."
        """
        raise NotImplementedError
