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

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.moe.moe import FusedMoE

from . import get_quantization_config
from .quant_base import QuantConfigBase, QuantMethodBase


class MixQuantConfig(QuantConfigBase):
    """
    Quantization config for layers that has different quantization methods.
    """

    def __init__(
        self,
        dense_quant_type: str,
        moe_quant_type: str,
        kv_cache_quant_type: str = None,
        image_moe_quant_type: str = None,
        is_channel_wise: bool = False,
        has_zero_point: bool = False,
        is_permuted: bool = True,
        is_quantized: bool = False,
        hadamard_block_size: int = 128,
        moe_dynamic_quant: bool = False,
    ) -> None:
        super().__init__()
        self.dense_quant_type = dense_quant_type
        self.moe_quant_type = moe_quant_type
        self.kv_cache_quant_type = kv_cache_quant_type
        if image_moe_quant_type is None:
            self.image_moe_quant_type = moe_quant_type
        else:
            self.image_moe_quant_type = image_moe_quant_type
        self.is_channel_wise = is_channel_wise
        self.has_zero_point = has_zero_point
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0
        self.is_permuted = is_permuted
        self.is_checkpoint_bf16 = not is_quantized
        self.is_quantized = is_quantized
        self.hadamard_block_size = hadamard_block_size
        self.moe_dynamic_quant = moe_dynamic_quant

    def name(self) -> str:
        return "mix_quant"

    @classmethod
    def from_config(cls, config: dict) -> "MixQuantConfig":
        return cls(
            config.get("dense_quant_type", None),
            config.get("moe_quant_type", None),
            config.get("kv_cache_quant_type", None),
            config.get("image_moe_quant_type", None),
            config.get("is_channel_wise", False),
            config.get("has_zero_point", False),
            config.get("is_permuted", True),
            config.get("is_quantized", False),
            config.get("hadamard_block_size", 128),
            config.get("moe_dynamic_quant", False),
        )

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            if layer.moe_tag == "Image":
                if self.image_moe_quant_type is not None:
                    return (
                        get_quantization_config(self.image_moe_quant_type)
                        .from_config(
                            {
                                "is_permuted": self.is_permuted,
                                "is_quantized": not self.is_checkpoint_bf16,
                                "hadamard_block_size": self.hadamard_block_size,
                            }
                        )
                        .get_quant_method(layer)
                    )
                else:
                    return None
            else:
                if self.moe_quant_type is not None:
                    return (
                        get_quantization_config(self.moe_quant_type)
                        .from_config(
                            {
                                "is_permuted": self.is_permuted,
                                "is_quantized": not self.is_checkpoint_bf16,
                                "hadamard_block_size": self.hadamard_block_size,
                            }
                        )
                        .get_quant_method(layer)
                    )
                else:
                    return None
        elif isinstance(layer, Attention):
            if self.kv_cache_quant_type is not None:
                return (
                    get_quantization_config("kvcache")
                    .from_config(self.kv_cache_quant_type, self.is_channel_wise, self.has_zero_point)
                    .get_quant_method(layer)
                )
            else:
                return None
        else:
            if self.dense_quant_type is not None:
                return (
                    get_quantization_config(self.dense_quant_type)
                    .from_config({"is_quantized": not self.is_checkpoint_bf16})
                    .get_quant_method(layer)
                )
            else:
                return None
