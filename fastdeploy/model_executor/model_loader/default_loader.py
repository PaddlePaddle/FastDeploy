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

import contextlib

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    load_composite_checkpoint,
    measure_time,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.platforms import current_platform


class DefaultModelLoader(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("Load the model and weights using DefaultModelLoader")

    def download_model(self, model_config: ModelConfig) -> None:
        """download_model"""
        pass

    def clean_memory_fragments(self, state_dict: dict) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda() or current_platform.is_maca():
            if state_dict:
                for k, v in state_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v.value().get_tensor()._clear()
            paddle.device.empty_cache()
            paddle.device.synchronize()

    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, architectures: str) -> None:
        model_class = ModelRegistry.get_pretrain_cls(architectures)

        state_dict = load_composite_checkpoint(
            fd_config.model_config.model,
            model_class,
            fd_config,
            return_numpy=True,
        )

        model.set_state_dict(state_dict)
        self.clean_memory_fragments(state_dict)

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        logger.info(f"Starting to load model {architectures}")
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

            if fd_config.speculative_config.model_type != "mtp":
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MoeForCausalLM")
            else:
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MTPForCausalLM")

            architectures = architectures + "RL"
            context = paddle.LazyGuard()
        else:
            context = contextlib.nullcontext()

        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(fd_config)

        model.eval()

        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model

        # TODO(gongshaotian): Now, only support safetensor
        self.load_weights(model, fd_config, architectures)
        return model
