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

from fastdeploy.config import LLMConfig
from fastdeploy.model_runner.output import ModelRunnerOutput
from fastdeploy.utils import get_logger

logger = get_logger("model_runner_base", "model_runner_base.log")


class ModelRunnerBase(ABC):
    """
        Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
        ModelRunner interface abstracts the model execution logic that
        contain input preparation, token generation, and tokenprocessing.
    """

    def __init__(self, llm_config: LLMConfig, device: str) -> None:
        self.llm_conig = llm_config
        self.model_config = llm_config.model_config
        self.lora_config = llm_config.lora_config
        self.load_config = llm_config.load_config
        self.device_config = llm_config.device_config
        self.speculative_config = llm_config.speculative_config
        self.kv_cache_config = self.llm_config.kv_cache_config
        self.parallel_config = self.llm_config.parallel_config
        # ... config

        self.device = device

    @abstractmethod
    def load_model(self) -> None:
        """ """
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """ """
        raise NotImplementedError

    @abstractmethod
    def execute_model(self, ) -> ModelRunnerOutput:
        """ """
        raise NotImplementedError

    @abstractmethod
    def profile_run(self) -> None:
        """ """
        raise NotImplementedError
