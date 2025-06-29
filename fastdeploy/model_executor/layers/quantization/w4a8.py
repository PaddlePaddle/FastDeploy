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
from typing import Optional

from ..moe import FusedMoE
from .quant_base import QuantConfigBase, QuantMethodBase


class W4A8Config(QuantConfigBase):
    """
    quantization config for weight 4bits and activation 8bits
    """

    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "w4a8"

    @classmethod
    def from_config(cls, config: dict) -> "W4A8Config":
        return cls()

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import CutlassW4A8MoEMethod
            return CutlassW4A8MoEMethod(self)
        else:
            raise ValueError(f"Unsupported layer type {type(layer)} for w4a8")
