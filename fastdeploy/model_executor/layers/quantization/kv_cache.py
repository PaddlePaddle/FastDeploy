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
from enum import Enum
from typing import Optional

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.utils import get_tensor

from ..utils import create_and_set_parameter
from .quant_base import QuantConfigBase, QuantMethodBase


class KvCacheQuantzationTypes(str, Enum):
    """
    KvCacheQuantzationTypes
    """
    INT8 = "int8"
    FP8 = "float8_e4m3fn"
    INT8_ZP = "int8_zp"
    FP8_ZP = "float8_e4m3fn_zp"


class KvCacheQuantConfig(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, kv_cache_quant_type: str) -> None:
        """
        __init__
        """
        super().__init__()
        self.kv_cache_quant_type = kv_cache_quant_type

        try:
            self.quant_type = KvCacheQuantzationTypes(kv_cache_quant_type)
        except ValueError:
            raise ValueError(f'Invalid Kvcache type: {kv_cache_quant_type}')

        self.has_zero_point = "zp" in kv_cache_quant_type

        if self.quant_type == KvCacheQuantzationTypes.INT8 or self.quant_type == KvCacheQuantzationTypes.INT8_ZP:
            self.max_bound = 127.0
        elif self.quant_type == KvCacheQuantzationTypes.FP8 or self.quant_type == KvCacheQuantzationTypes.FP8_ZP:
            self.max_bound = 448.0
        else:
            raise ValueError(f'Invalid Kvcache type: {kv_cache_quant_type}')

    def name(self) -> str:
        """
        get_name
        """
        return "kvcache"

    @classmethod
    def from_config(cls, kv_cache_quant_type: str) -> "KvCacheQuantConfig":
        """
        from_config
        """
        return cls(kv_cache_quant_type)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        get_quant_method
        """
        return KVCacheMethodBase(self)


class KVCacheMethodBase(QuantMethodBase):
    """
    KVCacheMethodBase
    """

    def __init__(
        self,
        quant_config: KvCacheQuantConfig,
    ) -> None:
        """
        KVCacheMethodBase __init__
        """
        super().__init__()
        self.cache_quant_config = quant_config

    def load_zp(self, layer: nn.Layer, state_dict):
        """
        load_zp
        """
        cache_k_zeropoint = get_tensor(state_dict.pop(self.cache_k_zp_name))
        cache_v_zeropoint = get_tensor(state_dict.pop(self.cache_v_zp_name))

        create_and_set_parameter(layer, "cache_k_zp", cache_k_zeropoint)
        create_and_set_parameter(layer, "cache_v_zp", cache_v_zeropoint)

    def load_scale(self, layer: nn.Layer, state_dict):
        """
        load_scale
        """
        cache_k_scale_tensor = get_tensor(
            state_dict.pop(self.cache_k_scale_name)).cast(
                paddle.get_default_dtype()).reshape_([-1])
        cache_v_scale_tensor = get_tensor(
            state_dict.pop(self.cache_v_scale_name)).cast(
                paddle.get_default_dtype()).reshape_([-1])

        cache_k_scale = self.cache_quant_config.max_bound / cache_k_scale_tensor
        cache_v_scale = self.cache_quant_config.max_bound / cache_v_scale_tensor
        cache_k_out_scale = cache_k_scale_tensor / self.cache_quant_config.max_bound
        cache_v_out_scale = cache_v_scale_tensor / self.cache_quant_config.max_bound

        create_and_set_parameter(layer, "cache_k_scale", cache_k_scale)
        create_and_set_parameter(layer, "cache_v_scale", cache_v_scale)
        create_and_set_parameter(layer, "cache_k_out_scale", cache_k_out_scale)
        create_and_set_parameter(layer, "cache_v_out_scale", cache_v_out_scale)

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        create_weights
        """
        self.prefix = layer.prefix
        self.cache_k_scale_name = layer.prefix + ".cachek_matmul.activation_scale"
        self.cache_v_scale_name = layer.prefix + ".cachev_matmul.activation_scale"
        self.cache_k_zp_name = layer.prefix + ".cachek_matmul.activation_zero_point"
        self.cache_v_zp_name = layer.prefix + ".cachev_matmul.activation_zero_point"

        if self.cache_quant_config.quant_type == KvCacheQuantzationTypes.INT8:
            setattr(layer, "cache_quant_type_str", "cache_int8")
            setattr(layer, "quant_max_bound", 127.0)
            setattr(layer, "quant_min_bound", -127.0)
        elif self.cache_quant_config.quant_type == KvCacheQuantzationTypes.FP8:
            setattr(layer, "cache_quant_type_str", "cache_fp8")
            setattr(layer, "quant_max_bound", 448.0)
            setattr(layer, "quant_min_bound", -448.0)
        else:
            raise NotImplementedError(f"{self.cache_quant_config.quant_type} is not implemented")

        self.load_scale(layer, state_dict)
        if self.cache_quant_config.has_zero_point:
            self.load_zp(layer, state_dict)

    def apply(self, layer):
        """
        apply
        """
        raise RuntimeError(
            f"{self.__class__.__name__}.apply should not be called.")
