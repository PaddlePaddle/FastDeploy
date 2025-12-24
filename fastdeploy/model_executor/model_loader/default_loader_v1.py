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

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (
    get_weight_iterator,
    is_weight_cache_enabled,
    load_weights_from_cache,
    measure_time,
    save_model,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.utils import process_final_after_loading
from fastdeploy.platforms import current_platform


class DefaultModelLoaderV1(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda() or current_platform.is_maca():
            paddle.device.empty_cache()
            paddle.device.synchronize()

    @save_model()
    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, enable_cache: bool = False) -> None:
        weights_iterator = get_weight_iterator(fd_config.model_config.model)
        if enable_cache:
            load_weights_from_cache(model, weights_iterator)
        else:
            model.load_weights(weights_iterator)
        if fd_config.speculative_config.model_type != "mtp":
            process_final_after_loading(model, fd_config)

        self.clean_memory_fragments()

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        context = paddle.LazyGuard()
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

            if fd_config.speculative_config.model_type != "mtp":
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MoeForCausalLM")
            else:
                architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MTPForCausalLM")

            architectures = architectures + "RL"

        enable_cache, _, weight_cache_context = is_weight_cache_enabled(fd_config)
        fd_config.model_config.enable_cache = enable_cache
        with weight_cache_context:
            with context:
                model_cls = ModelRegistry.get_class(architectures)
                convert_type = fd_config.model_config.convert_type
                if convert_type == "none":
                    pass
                elif convert_type == "embed":
                    model_cls = as_embedding_model(model_cls)
                else:
                    assert_never(convert_type)

                model = model_cls(fd_config)
                if fd_config.load_config.dynamic_load_weight or fd_config.model_config.enable_cache:
                    process_final_after_loading(model, fd_config)

        model.eval()
        # RL model not need set_state_dict
        if fd_config.load_config.dynamic_load_weight:
            return model
        self.load_weights(model, fd_config, enable_cache)
        return model
