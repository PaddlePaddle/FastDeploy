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

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    get_weight_iterator,
    is_weight_cache_enabled,
    load_weights_form_cache,
    measure_time,
    save_model,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.platforms import current_platform


class DefaultModelLoaderV1(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.synchronize()

    @save_model()
    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, enable_cache: bool = False) -> None:
        weights_iterator = get_weight_iterator(fd_config.model_config.model)
        if enable_cache:
            load_weights_form_cache(model, weights_iterator)
        else:
            model.load_weights(weights_iterator)
        self.clean_memory_fragments()

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        logger.info(f"Starting to load model {architectures}")
        context = paddle.LazyGuard()
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

            architectures = architectures + "RL"

        enable_cache, _, weight_cache_context = is_weight_cache_enabled(fd_config)
        with weight_cache_context:
            with context:
                model_cls = ModelRegistry.get_class(architectures)
                model = model_cls(fd_config)

        model.eval()
        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model
        self.load_weights(model, fd_config, enable_cache)
        return model
