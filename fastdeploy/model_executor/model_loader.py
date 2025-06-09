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

import paddle
from paddle import nn

from fastdeploy.config import LLMConfig, LoadConfig, ModelConfig


# TODO(gongshaotian): implement real interface to replace this
def get_model(llm_config: LLMConfig) -> nn.Layer:
    """ load or download model """
    model_path = llm_config.load_config.model_path
    model = paddle.load(model_path, return_numpy=True)
    return model


class BaseModelLoader(ABC):
    """ Base class for model loaders. """

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, load_config: ModelConfig) -> None:
        """ Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, llm_config: LLMConfig) -> nn.Layer:
        """ Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """ ModelLoader that can load registered models """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(self, llm_config: LLMConfig) -> nn.Layer:
        pass
