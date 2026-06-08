"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional

from paddleformers.utils.log import logger


from ..moe import FusedMoE
from .quant_base import QuantConfigBase, QuantMethodBase

QUANT_SCALING_FACTOR = 6


class WFP4AFP8Config(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, weight_scale_dict, act_scale_dict, is_permuted, is_quantized) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.quant_max_bound = 6
        self.quant_min_bound = -6
        self.quant_round_type = 1
        self.is_permuted = is_permuted
        self.is_quantized = is_quantized
        self.is_checkpoint_bf16 = not is_quantized

    def name(self) -> str:
        return "wfp4afp8"

    @classmethod
    def from_config(cls, config: dict) -> "WFP4AFP8Config":
        weight_scale_dict = config.get("weight_scale_dict", None)
        act_scale_dict = config.get("act_scale_dict", None)
        is_permuted = config.get("is_permuted", True)
        is_quantized = config.get("is_quantized", False)
        return cls(weight_scale_dict, act_scale_dict, is_permuted, is_quantized)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        logger.info("Currently only support DeepGEMMMegaMoE for wfp4afp8")
        if isinstance(layer, FusedMoE):
            from fastdeploy.model_executor.layers.moe.fused_moe_deepgemm_backend import (
                DeepGemmMegaMoEMethod,
            )

            return DeepGemmMegaMoEMethod(self)
        else:
            raise NotImplementedError(f"wfp4afp8 quant method not supported for {type(layer)}")
