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
    ORI_WEIGHT_NAME,
    QUANT_SCALE_NAME,
    QUANT_WEIGHT_NAME,
    fast_weights_iterator,
    get_all_safetensors,
    measure_time,
)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.models.utils import default_load_weights_into_param
from fastdeploy.model_executor.utils import switch_config_context
from fastdeploy.platforms import current_platform


class InflightQuantModelLoader(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        assert load_config.is_inflight_quant, "InflightQuantModelLoader can only be used for dynamic quantization."
        logger.info("Load the model and weights using InflightModelLoader")

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _clean_memory_fragments(self) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.synchronize()

    def create_model(self, fd_config, architectures):
        with paddle.LazyGuard():
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(fd_config)
        model.eval()
        return model

    def _get_quantized_weights_iterator(self, quantized_params_dict, fd_config: FDConfig):
        """
        Construct an unquantized model, perform weight preprocessing (e.g., tensor parallel splitting)
        on its parameters, and return an iterator of quantized weights.

        Args:
            quantized_params_dict (dict): A dictionary containing quantized parameter names and their tensors.
            fd_config (FDConfig): Configuration object with settings needed for weight processing.

        Returns:
            Iterator: Yields tuples of (weight_name, quantized_weight_tensor) for each quantized weight.
        """

        # 1.Create an unquantized model
        with switch_config_context(fd_config, "quant_config", None):
            architectures = fd_config.model_config.architectures[0]
            with paddle.LazyGuard():
                model_cls = ModelRegistry.get_class(architectures)
                unquantized_model = model_cls(fd_config)
            unquantized_model.eval()
        # 2.Get weight iterator
        _, safetensor_files = get_all_safetensors(fd_config.model_config.model)
        weights_iterator = fast_weights_iterator(safetensor_files)
        # 3.Get an iterator over the processed weights (e.g., tensor parallel splitting) .
        unquantized_params_dict = dict(unquantized_model.named_parameters())
        processed_weights_iterator = unquantized_model.processed_weights(weights_iterator, unquantized_params_dict)
        # 4.Quantize using the parameter that has a quantization method.
        for loaded_weight_name, model_param_name, _, preprocessed_weight, _, _ in processed_weights_iterator:
            if model_param_name in quantized_params_dict:
                yield loaded_weight_name, preprocessed_weight
            else:
                model_quant_weight_name = model_param_name.replace(ORI_WEIGHT_NAME, QUANT_WEIGHT_NAME)
                model_param = quantized_params_dict[model_quant_weight_name]
                quant_method = getattr(model_param, "quant_method", None)
                assert quant_method is not None, f"{model_quant_weight_name} lacks an implementation of quant_method."
                quant_weight_name = loaded_weight_name.replace(ORI_WEIGHT_NAME, QUANT_WEIGHT_NAME)
                quant_res = quant_method(preprocessed_weight)
                if len(quant_res) == 2:
                    quant_scale_name = loaded_weight_name.replace(ORI_WEIGHT_NAME, QUANT_SCALE_NAME)
                    quant_weight = quant_res[0]
                    weight_scale = quant_res[1]
                    yield quant_weight_name, quant_weight
                    yield quant_scale_name, weight_scale
                else:
                    yield quant_weight_name, quant_weight

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
        quantized_params_dict = dict(model.named_parameters())
        quanted_weights_iterator = self._get_quantized_weights_iterator(quantized_params_dict, fd_config)
        processed_weights_iterator = model.processed_weights(
            quanted_weights_iterator, quantized_params_dict, is_processed=True
        )
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
