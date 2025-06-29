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

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.models.ernie4_5_moe import \
    Ernie4_5_PretrainedModel
from fastdeploy.model_executor.models.ernie4_5_mtp import \
    Ernie4_5_MTPPretrainedModel
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.models.qwen2 import Qwen2PretrainedModel
from fastdeploy.model_executor.models.qwen3 import Qwen3PretrainedModel
from fastdeploy.model_executor.models.qwen3moe import Qwen3MoePretrainedModel
from fastdeploy.model_executor.models.utils import load_checkpoint

MODEL_CLASSES = {
    "Ernie4_5_MoeForCausalLM": Ernie4_5_PretrainedModel,
    "Ernie4_5_MTPForCausalLM": Ernie4_5_MTPPretrainedModel,
    "Qwen2ForCausalLM": Qwen2PretrainedModel,
    "Qwen3ForCausalLM": Qwen3PretrainedModel,
    "Qwen3MoeForCausalLM": Qwen3MoePretrainedModel,
    "Ernie4_5_ForCausalLM": Ernie4_5_PretrainedModel
}


def get_model_from_loader(fd_config: FDConfig) -> nn.Layer:
    """ load or download model """
    model_loader = DefaultModelLoader(fd_config.load_config)
    model = model_loader.load_model(fd_config)
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
    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        """ Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """ ModelLoader that can load registered models """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        context = paddle.LazyGuard()
        architectures = fd_config.model_config.architectures[0]

        # TODO(gongshaotian): Now, only support safetensor

        model_class = MODEL_CLASSES[architectures]
        state_dict = load_checkpoint(
            fd_config.parallel_config.model_name_or_path,
            model_class,
            fd_config.model_config,
            return_numpy=True)
        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(fd_config)

        model.eval()
        model.set_state_dict(state_dict)

        return model
