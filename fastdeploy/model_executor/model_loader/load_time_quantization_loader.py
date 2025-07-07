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

from functools import partial
from typing import Dict, Generator, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (deal_state_dict,
                                                         get_all_safetensors,
                                                         tp_weights_iterator)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.model_loader.utils import get_pretrain_cls
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.models.quant_utils import (
    apply_quant_action, check_quantization_prerequisites,
    get_quant_layer_instance_map)
from fastdeploy.model_executor.models.utils import switch_config_context
from fastdeploy.platforms import current_platform


class LoadTimeQuantizationModelLoader(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info(
            "Load the model and weights using LoadTimeQuantizationModelLoader")

    def download_model(self, model_config: ModelConfig) -> None:
        """download_model"""
        pass

    def _clean_memory_fragments(self, state_dict: dict) -> None:
        """clean_memory_fragments"""
        if current_platform.is_cuda():
            if state_dict:
                for k, v in state_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v.value().get_tensor()._clear()
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()

    def _get_quantized_weights(
        self,
        weight_iterator: Generator[tuple[str, Union[paddle.Tensor,
                                                    np.ndarray]], None, None],
        quant_filtered_map: Dict[str, partial],
        quant_layer_instance_map: Dict[str, nn.Layer],
    ) -> Dict[str, paddle.Tensor]:
        """_get_quantized_weights"""
        state_dict = {}
        for key, weight in weight_iterator:
            if not isinstance(weight, paddle.Tensor):
                weight = paddle.Tensor(weight, zero_copy=True)
                weight = weight._copy_to(
                    paddle.framework._current_expected_place(), False)
            if key in quant_filtered_map:
                apply_quant_action(
                    quant_filtered_map,
                    key,
                    weight,
                    state_dict,
                    quant_layer_instance_map,
                )
            else:
                state_dict[key] = weight
        deal_state_dict(state_dict)
        paddle.device.cuda.empty_cache()
        paddle.device.cuda.synchronize()
        return state_dict

    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        """load_model"""
        assert not fd_config.model_config.is_quantized
        with switch_config_context(fd_config.model_config, "is_quantized",
                                   True):
            context = paddle.LazyGuard()
            architectures = fd_config.model_config.architectures[0]
            model_class = get_pretrain_cls(architectures)
            model_cls = ModelRegistry.get_class(architectures)

            with context:
                model = model_cls(fd_config)
            model.eval()
            safetensor_keys, safetensor_files = get_all_safetensors(
                fd_config.parallel_config.model_name_or_path)
            if not self.load_config.use_fastsafetensor:
                logger.info(
                    "Tip: For faster model loading, consider enabling FastSafeTensor by setting:\n"
                    "  export FD_USE_FASTSAFETENSOR=1")
            weights_iterator = tp_weights_iterator(
                model_class,
                fd_config,
                safetensor_keys,
                safetensor_files,
                use_fastsafetensor=self.load_config.use_fastsafetensor,
            )
            model_dict = dict(model.named_sublayers())
            quant_filtered_map = {}

            check_quantization_prerequisites(
                fd_config,
                model_class,
                quant_filtered_map,
                safetensor_keys,
                model_dict,
            )
            need_quant = True if quant_filtered_map else False
            quant_layer_instance_map = {}
            if need_quant:
                quant_layer_instance_map = get_quant_layer_instance_map(
                    model_class, model_dict)
            need_quant = True if need_quant and quant_layer_instance_map else False

            assert need_quant, "Quantization must be enabled"

            state_dict = self._get_quantized_weights(weights_iterator,
                                                     quant_filtered_map,
                                                     quant_layer_instance_map)
            model.set_state_dict(state_dict)
            self._clean_memory_fragments(state_dict)
            return model
