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
from fastdeploy.model_executor.utils import set_weight_attrs
from fastdeploy.platforms import current_platform

from .quant_base import QuantConfigBase, QuantMethodBase


class KvCacheQuantzationTypes(str, Enum):
    """
    KvCacheQuantzationTypes
    """

    INT8 = "int8"
    FP8 = "float8_e4m3fn"
    BLOCK_WISE_FP8 = "block_wise_fp8"
    INT8_ZP = "int8_zp"
    INT4_ZP = "int4_zp"
    FP8_ZP = "float8_e4m3fn_zp"


class KvCacheQuantConfig(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, kv_cache_quant_type: str, is_channel_wise: bool, has_zero_point: bool) -> None:
        """
        __init__
        """
        super().__init__()
        self.kv_cache_quant_type = kv_cache_quant_type
        self.is_channel_wise = is_channel_wise
        self.has_zero_point = has_zero_point

        try:
            self.quant_type = KvCacheQuantzationTypes(kv_cache_quant_type)
        except ValueError:
            raise ValueError(f"Invalid Kvcache type: {kv_cache_quant_type}")

        if "zp" in kv_cache_quant_type:
            self.has_zero_point = True

        if self.quant_type == KvCacheQuantzationTypes.INT8 or self.quant_type == KvCacheQuantzationTypes.INT8_ZP:
            self.max_bound = 127.0
            self.is_channel_wise = True
        elif (
            self.quant_type == KvCacheQuantzationTypes.FP8
            or self.quant_type == KvCacheQuantzationTypes.FP8_ZP
            or self.quant_type == KvCacheQuantzationTypes.BLOCK_WISE_FP8
        ):
            self.max_bound = 448.0
        elif self.quant_type == KvCacheQuantzationTypes.INT4_ZP:
            self.max_bound = 7.0
        else:
            raise ValueError(f"Invalid Kvcache type: {kv_cache_quant_type}")

    def name(self) -> str:
        """
        get_name
        """
        return "kvcache"

    @classmethod
    def from_config(
        cls, kv_cache_quant_type: str, is_channel_wise: bool, has_zero_point: bool
    ) -> "KvCacheQuantConfig":
        """
        from_config
        """
        return cls(kv_cache_quant_type, is_channel_wise, has_zero_point)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        get_quant_method
        """
        if current_platform.is_xpu():
            from fastdeploy.model_executor.layers.backends.xpu.quantization.kv_cache import (
                XPUKVCacheMethodBase,
            )

            return XPUKVCacheMethodBase(self)
        else:
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
        cache_k_zeropoint = get_tensor(state_dict.pop(self.cache_k_zp_name)).cast(paddle.get_default_dtype())
        cache_v_zeropoint = get_tensor(state_dict.pop(self.cache_v_zp_name)).cast(paddle.get_default_dtype())

        layer.cache_k_zp.set_value(cache_k_zeropoint)
        layer.cache_v_zp.set_value(cache_v_zeropoint)

    def load_scale(self, layer: nn.Layer, state_dict):
        """
        load_scale
        """
        cache_k_scale_tensor = (
            get_tensor(state_dict.pop(self.cache_k_scale_name)).cast(paddle.get_default_dtype()).reshape_([-1])
        )
        cache_v_scale_tensor = (
            get_tensor(state_dict.pop(self.cache_v_scale_name)).cast(paddle.get_default_dtype()).reshape_([-1])
        )

        if self.cache_quant_config.has_zero_point:  # cache_int4_zp
            cache_k_scale = 1.0 / cache_k_scale_tensor
            cache_v_scale = 1.0 / cache_v_scale_tensor
            cache_k_out_scale = cache_k_scale_tensor
            cache_v_out_scale = cache_v_scale_tensor
        else:
            cache_k_scale = self.cache_quant_config.max_bound / cache_k_scale_tensor
            cache_v_scale = self.cache_quant_config.max_bound / cache_v_scale_tensor
            cache_k_out_scale = cache_k_scale_tensor / self.cache_quant_config.max_bound
            cache_v_out_scale = cache_v_scale_tensor / self.cache_quant_config.max_bound

        layer.cache_k_scale.set_value(cache_k_scale)
        layer.cache_v_scale.set_value(cache_v_scale)
        layer.cache_k_out_scale.set_value(cache_k_out_scale)
        layer.cache_v_out_scale.set_value(cache_v_out_scale)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        create_weights
        """
        if self.cache_quant_config.quant_type == KvCacheQuantzationTypes.INT8:
            layer.cache_quant_type_str = "cache_int8"
            layer.quant_max_bound = 127.0
            layer.quant_min_bound = -127.0
        elif self.cache_quant_config.quant_type == KvCacheQuantzationTypes.FP8:
            layer.cache_quant_type_str = "cache_fp8"
            layer.quant_max_bound = 448.0
            layer.quant_min_bound = -448.0
        elif self.cache_quant_config.quant_type == KvCacheQuantzationTypes.INT4_ZP:
            layer.cache_quant_type_str = "cache_int4_zp"
            layer.quant_max_bound = 7.0
            layer.quant_min_bound = -7.0
        elif self.cache_quant_config.quant_type == KvCacheQuantzationTypes.BLOCK_WISE_FP8:
            layer.cache_quant_type_str = "block_wise_fp8"
            layer.quant_max_bound = 448.0
            layer.quant_min_bound = -448.0
        else:
            raise NotImplementedError(f"{self.cache_quant_config.quant_type} is not implemented")

        scale_shape = [layer.fd_config.model_config.num_key_value_heads]
        if self.cache_quant_config.is_channel_wise:
            scale_shape = [layer.kv_num_heads * layer.head_dim]

        layer.cache_k_scale = layer.create_parameter(
            shape=scale_shape,
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.cache_v_scale = layer.create_parameter(
            shape=scale_shape,
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        set_weight_attrs(
            layer.cache_k_scale,
            {
                **extra_weight_attrs,
            },
        )
        set_weight_attrs(
            layer.cache_v_scale,
            {
                **extra_weight_attrs,
            },
        )

        layer.cache_k_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.cache_v_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        if self.cache_quant_config.has_zero_point:
            layer.cache_k_zp = layer.create_parameter(
                shape=scale_shape,
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.cache_v_zp = layer.create_parameter(
                shape=scale_shape,
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            set_weight_attrs(
                layer.cache_k_zp,
                {
                    **extra_weight_attrs,
                },
            )
            set_weight_attrs(
                layer.cache_v_zp,
                {
                    **extra_weight_attrs,
                },
            )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        use for loader v0
        """
        self.prefix = layer.prefix
        self.cache_k_scale_name = layer.prefix + ".cachek_matmul.activation_scale"
        self.cache_v_scale_name = layer.prefix + ".cachev_matmul.activation_scale"
        self.cache_k_zp_name = layer.prefix + ".cachek_matmul.activation_zero_point"
        self.cache_v_zp_name = layer.prefix + ".cachev_matmul.activation_zero_point"

        if "block_wise" not in layer.cache_quant_type_str:
            self.load_scale(layer, state_dict)
            if self.cache_quant_config.has_zero_point:
                self.load_zp(layer, state_dict)

    def process_weights_after_loading(self, layer: nn.Layer):
        """
        use for loader v1
        """
        if layer.cache_k_scale._is_initialized():
            layer.cache_k_out_scale.set_value(1 / layer.cache_k_scale)
        if layer.cache_v_scale._is_initialized():
            layer.cache_v_out_scale.set_value(1 / layer.cache_v_scale)

    def apply(self, layer):
        """
        apply
        """
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")
