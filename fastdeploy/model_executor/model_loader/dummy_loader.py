"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import time

import paddle
from paddle import nn
from paddleformers.utils.log import logger
from typing_extensions import assert_never

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import is_weight_cache_enabled
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.utils import process_final_after_loading


class DummyModelLoader(BaseModelLoader):
    """Model loader that initializes model weights with random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("Load the model and initialize dummy weights")

    def download_model(self, model_config: ModelConfig) -> None:
        """download_model"""
        pass

    def _initialize_dummy_weights(
        self,
        model: nn.Module,
        low: float = -1e-3,
        high: float = 1e-3,
    ) -> None:
        float_dtypes = (
            paddle.float16,
            paddle.float32,
            paddle.float64,
            paddle.bfloat16,
        )
        float8_dtypes = (
            paddle.float8_e4m3fn,
            paddle.float8_e5m2,
        )
        with paddle.no_grad():
            for _, param in model.named_parameters():
                if param is None:
                    continue
                if not param.shape or 0 in param.shape:
                    continue
                if param.dtype in float8_dtypes:
                    tmp = (high - low) * paddle.randn(param.shape, dtype=paddle.float16) + low
                    param.copy_(tmp.cast(param.dtype), False)
                elif param.dtype in float_dtypes:
                    param.set_value((high - low) * paddle.randn(param.shape, dtype=param.dtype) + low)
                else:
                    param.set_value(paddle.zeros(param.shape, dtype=param.dtype))

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        start_dummy_weight_time = time.time()
        architectures = fd_config.model_config.architectures[0]
        context = paddle.LazyGuard()
        if fd_config.load_config.dynamic_load_weight:
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

        model.eval()
        self._initialize_dummy_weights(model)
        process_final_after_loading(model, fd_config)
        logger.info("dummy weight cost time: {}s".format(time.time() - start_dummy_weight_time))
        return model
