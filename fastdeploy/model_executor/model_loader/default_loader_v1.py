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
    fast_weights_iterator,
    get_all_safetensors,
    measure_time,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.models.utils import default_load_weights_into_param
from fastdeploy.platforms import current_platform


class DefaultModelLoaderV1(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        assert not load_config.is_inflight_quant, (
            "Dynamic quantization requires running with --load_choices 'inflight_quant' "
            "or load_choices='inflight_quant'."
        )
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.synchronize()

    def _load_weights_into_param(self, model, processed_weights_iterator):
        """_load_weights_into_param"""
        for loaded_weight_name, _, model_param, preprocessed_weight, shard_id, expert_id in processed_weights_iterator:
            load_weights_into_param = getattr(
                model_param, "load_weights_into_param", default_load_weights_into_param()
            )
            if expert_id is not None:
                load_weights_into_param(model_param, preprocessed_weight, expert_id, shard_id)
            else:
                load_weights_into_param(model_param, preprocessed_weight, shard_id)
        if hasattr(model, "after_load_weights"):
            model.after_load_weights()

    @measure_time
    def _load_weights(self, model, fd_config: FDConfig) -> None:
        _, safetensor_files = get_all_safetensors(fd_config.model_config.model)
        weights_iterator = fast_weights_iterator(safetensor_files)
        params_dict = dict(model.named_parameters())
        processed_weights_iterator = model.processed_weights(weights_iterator, params_dict)
        self._load_weights_into_param(model, processed_weights_iterator)
        self._clean_memory_fragments()

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        architectures = fd_config.model_config.architectures[0]
        logger.info(f"Starting to load model {architectures}")
        if fd_config.load_config.dynamic_load_weight:
            # register rl model
            import fastdeploy.rl  # noqa

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

        self._load_weights(model, fd_config)
        return model
