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
from . import get_quantization_config
from .quant_base import QuantConfigBase, QuantMethodBase


class WINT2Config(QuantConfigBase):
    """
    Quantization config for wint8 linear and w4w2 MoE.
    """

    def __init__(
        self,
        dense_quant_type: str,
        dense_quant_granularity: str,
        moe_quant_type: str,
        moe_w4_quant_type: str,
        moe_w4_quant_granularity: str,
        moe_w4_quant_start_layer: int,
        moe_w4_quant_end_layer: int,
        moe_w2_quant_type: str,
        moe_w2_quant_granularity: str,
        moe_w2_quant_group_size: int,
        moe_w2_quant_start_layer: int,
        moe_w2_quant_end_layer: int,
    ) -> None:
        super().__init__()
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0

        # wint2 quantization config
        self.dense_quant_type = dense_quant_type
        self.dense_quant_granularity = dense_quant_granularity
        self.moe_quant_type = moe_quant_type
        self.moe_w4_quant_type = moe_w4_quant_type
        self.moe_w4_quant_granularity = moe_w4_quant_granularity
        self.moe_w4_quant_start_layer = moe_w4_quant_start_layer
        self.moe_w4_quant_end_layer = moe_w4_quant_end_layer
        self.moe_w2_quant_type = moe_w2_quant_type
        self.moe_w2_quant_granularity = moe_w2_quant_granularity
        self.moe_w2_quant_group_size = moe_w2_quant_group_size
        self.moe_w2_quant_start_layer = moe_w2_quant_start_layer
        self.moe_w2_quant_end_layer = moe_w2_quant_end_layer

    def name(self) -> str:
        """
        Get the name of the quantization configuration.
        Returns:
            str: The name of the quantization configuration.
        """
        return "wint2"

    @classmethod
    def from_config(cls, config: dict) -> "WINT2Config":
        """
        Create a new instance of `WINT2Config` using the provided configuration dictionary.
        Args:
            config (dict): A dictionary containing the configuration parameters for the new instance.

        Returns:
            WINT2Config: The newly created instance of `WINT2Config`.
        """

        dense_quant_type = config.get("dense_quant_type", "wint8")
        dense_quant_granularity = config.get("dense_quant_granularity", "per_channel")

        moe_quant_config = config.get("moe_quant_config", {})
        moe_quant_type = moe_quant_config.get("quant_type", "w4w2")

        moe_w4_quant_config = moe_quant_config.get("moe_w4_quant_config", {})
        moe_w4_quant_type = moe_w4_quant_config.get("quant_type", "wint4")
        moe_w4_quant_granularity = moe_w4_quant_config.get("quant_granularity", "per_channel")
        moe_w4_quant_start_layer = moe_w4_quant_config.get("quant_start_layer", 0)
        moe_w4_quant_end_layer = moe_w4_quant_config.get("quant_end_layer", 6)

        moe_w2_quant_config = moe_quant_config.get("moe_w2_quant_config", {})
        moe_w2_quant_type = moe_w2_quant_config.get("quant_type", "wint2")
        moe_w2_quant_granularity = moe_w2_quant_config.get("quant_granularity", "pp_acc")
        moe_w2_quant_group_size = moe_w2_quant_config.get("quant_group_size", 0)
        moe_w2_quant_start_layer = moe_w2_quant_config.get("quant_start_layer", 0)
        moe_w2_quant_end_layer = moe_w2_quant_config.get("quant_end_layer", 0)

        return cls(
            dense_quant_type,
            dense_quant_granularity,
            moe_quant_type,
            moe_w4_quant_type,
            moe_w4_quant_granularity,
            moe_w4_quant_start_layer,
            moe_w4_quant_end_layer,
            moe_w2_quant_type,
            moe_w2_quant_granularity,
            moe_w2_quant_group_size,
            moe_w2_quant_start_layer,
            moe_w2_quant_end_layer,
        )

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        Get the quantization method associated with the given layer based on the current quantization configuration.
        Args:
            layer (Layer): The layer for which the quantization method should be retrieved.

        Returns:
            QuantMethodBase: The quantization method associated with the given layer.
        """
        if isinstance(layer, FusedMoE):
            if layer.layer_idx <= self.moe_w4_quant_end_layer:
                return get_quantization_config(self.moe_w4_quant_type).from_config({}).get_quant_method(layer)
            else:
                from fastdeploy.model_executor.layers.moe.fused_moe_wint2_backend import (
                    CutlassWint2FusedMoeMethod,
                )

                return CutlassWint2FusedMoeMethod(self)
        else:
            return get_quantization_config(self.dense_quant_type).from_config({}).get_quant_method(layer)
