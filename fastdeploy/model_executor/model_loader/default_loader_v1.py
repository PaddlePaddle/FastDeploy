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
from typing_extensions import assert_never

from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.utils import get_logger

logger = get_logger("default_loader_v1", "default_loader_v1.log")

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    fast_weights_iterator,
    get_all_safetensors,
    measure_time,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.registry import model_registry
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

    @measure_time
    def load_weights(self, model, fd_config: FDConfig) -> None:
        _, safetensor_files = get_all_safetensors(fd_config.model_config.model)
        weights_iterator = fast_weights_iterator(safetensor_files)
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

        with context:
            model_cls = model_registry.get_class(architectures)

            model = model_cls(fd_config)

        convert_type = fd_config.model_config.convert_type
        logger.info(f"convert_type:{convert_type}")
        if convert_type == "none":
            pass
        elif convert_type == "embed":
            logger.info("Converting to embedding model.")
            model_cls = as_embedding_model(model_cls)
        else:
            assert_never(convert_type)

        model.eval()

        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model
        self.load_weights(model, fd_config)
        return model
