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
from abc import abstractmethod
from typing import Optional

import paddle

from .utils import xpu_quant_weight

from fastdeploy.model_executor.layers.quantization.quant_base import QuantConfigBase
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig, WeightOnlyLinearMethod

class XPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on XPU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def process_loaded_weights(self, layer, weight) -> None:
        """
        loaded_weights using xpu special quantization
        """
        quanted_weight_tensor, weight_scale_tensor = xpu_quant_weight(
            weight.cpu().numpy())
        layer.linear_weight.set_value(quanted_weight_tensor)
        layer.linear_weight_scale.set_value(
            weight_scale_tensor.astype(paddle.get_default_dtype()))
