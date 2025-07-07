"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from typing import Dict, Optional

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.utils import switch_level_context


def get_quant_layer_instance_map(
        cls: PretrainedModel,
        model_dict: Dict[str, nn.Layer]) -> Dict[str, nn.Layer]:
    """get_quant_layer_instance_map"""
    suffix_set = set(cls.quant_need_find_layer_list)
    quant_layer_map = {}

    remaining_suffixes = set(suffix_set)

    for key, layer in model_dict.items():
        for suffix in list(remaining_suffixes):
            if key.endswith(suffix):
                quant_layer_map[suffix] = layer
                remaining_suffixes.remove(suffix)
                break

        if not remaining_suffixes:
            break

    if not quant_layer_map:
        logger.error(
            "quant_map should not be empty. "
            "Pre-quantization is required, but _get_quantization_mappings is not implemented."
        )

    return quant_layer_map


def apply_quant_action(
    quant_map: Dict[str, partial],
    key: str,
    tensor: paddle.Tensor,
    state_dict: Dict[str, paddle.Tensor],
    quant_layer_instance_map: Dict[str, nn.Layer],
) -> None:
    """
    apply_quant_action
    """
    action = quant_map.pop(key)
    quant_weight_tensor, weight_quanter_tensor = action(
        key, tensor, quant_layer_instance_map)
    if quant_weight_tensor._is_initialized():
        quant_weight_key = key.replace("weight", "quant_weight")
        state_dict[quant_weight_key] = quant_weight_tensor
    if weight_quanter_tensor._is_initialized():
        weight_quanter_key = key.replace("weight", "weight_scale")
        state_dict[weight_quanter_key] = weight_quanter_tensor


@switch_level_context("WARNING")
def check_quantization_prerequisites(
    fd_config: FDConfig,
    cls: PretrainedModel,
    quant_filtered_map: Dict[str, partial],
    safetensor_keys: list[str],
    model_dict: Optional[Dict[str, nn.Layer]] = None,
) -> None:
    """check_quantization_prerequisites"""
    if not hasattr(cls, "_get_quantization_mappings"):
        raise NotImplementedError(
            f"Class {cls.__name__} must implement method '_get_quantization_mappings'"
        )
    quant_map = cls._get_quantization_mappings(fd_config)
    if not quant_map:
        logger.error("quant_map should not be empty. \
        pre-quantization required, but _get_quantization_mappings is not implemented."
                     )
    else:
        filtered_quant_map = cls._resolve_prefix_keys(quant_map.keys(),
                                                      safetensor_keys)
        for k, v in filtered_quant_map.items():
            quant_filtered_map[v] = quant_map.pop(k)
        if not filtered_quant_map:
            logger.info("No weights to quantize; filtered_quant_map is empty.")
        else:
            if not model_dict:
                logger.error(
                    "Missing required argument 'model_dict' when calling tp_weights_iterator."
                )


def quantization_func(fd_config: FDConfig):
    """quantization_func"""

    def fn(
        key: str,
        tensor: paddle.Tensor,
        quant_layer_instance_map: Dict[str, nn.Layer],
        quant_layer_key: str = "",
    ):
        """fn"""
        quant_layer = quant_layer_instance_map[quant_layer_key]
        quant_method = fd_config.quant_config.get_quant_method(quant_layer)
        if quant_method is None:
            raise ValueError("quant_method should not be None.")
        try:
            (quanted_weight_tensor, weight_scale_tensor) = (
                quant_method.apply_weight_quantization(tensor))
        except Exception:
            raise ValueError(
                f"{key} Expected apply_weight_quantization is missing from {quant_method}"
            )
        return quanted_weight_tensor, weight_scale_tensor

    return fn
