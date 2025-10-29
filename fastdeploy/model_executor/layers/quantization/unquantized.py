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

from fastdeploy.model_executor.layers.linear import UnquantizedLinearMethod
from fastdeploy.platforms import current_platform

from ..moe import FusedMoE
from .quant_base import QuantConfigBase


class UnquantizedConfig(QuantConfigBase):
    """
    Quantization config for unquantized
    """

    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "unquantized"

    @classmethod
    def from_config(cls, config: dict) -> "UnquantizedConfig":
        return cls()

    def get_quant_method(self, layer):
        if current_platform.is_cuda():
            if isinstance(layer, FusedMoE):
                from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
                    CutlassMoEMethod,
                )

                return CutlassMoEMethod(self)
            else:
                return UnquantizedLinearMethod()
        else:
            raise RuntimeError("Unsupported platform {}".format(current_platform))
