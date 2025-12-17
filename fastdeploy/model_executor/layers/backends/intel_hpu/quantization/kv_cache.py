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

from fastdeploy.model_executor.layers.quantization.quant_base import (
    QuantConfigBase,
    QuantMethodBase,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import set_weight_attrs


class KvCacheQuantzationTypes(str, Enum):
    """
    KvCacheQuantzationTypes
    """

    FP8 = "float8_e4m3fn"
    FP8_E4M3 = "float8_e4m3"


class HPUKvCacheQuantConfig(QuantConfigBase):
    """
    quantization config for weight fp8
    """

    def __init__(self, kv_cache_quant_type: str, is_channel_wise: bool, has_zero_point: bool) -> None:
        """
        __init__
        """
        super().__init__()
        self.kv_cache_quant_type = kv_cache_quant_type

        try:
            self.quant_type = KvCacheQuantzationTypes(kv_cache_quant_type)
        except ValueError:
            raise ValueError(f"Invalid Kvcache type: {kv_cache_quant_type}")

        if self.quant_type == KvCacheQuantzationTypes.FP8_E4M3:
            self.max_bound = 240.0
        elif self.quant_type == KvCacheQuantzationTypes.FP8:
            self.max_bound = 448.0
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
    ) -> "HPUKvCacheQuantConfig":
        """
        from_config
        """
        return cls(kv_cache_quant_type, is_channel_wise, has_zero_point)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        get_quant_method
        """
        return HPUKVCacheMethodBase(self)


class HPUKVCacheMethodBase(QuantMethodBase):
    """
    HPUKVCacheMethodBase: HPU need scale in fp32 format but GPU define all scale in bf16 format
    """

    def __init__(
        self,
        quant_config: HPUKvCacheQuantConfig,
    ) -> None:
        """
        HPUKVCacheMethodBase __init__
        """
        super().__init__()
        self.cache_quant_config = quant_config

    def load_scale(self, layer: nn.Layer, state_dict):
        """
        load_scale
        """

        cache_k_scale_tensor = get_tensor(state_dict.pop(self.cache_k_scale_name)).cast("float32").reshape_([-1])
        cache_v_scale_tensor = get_tensor(state_dict.pop(self.cache_v_scale_name)).cast("float32").reshape_([-1])
        q_scale_tensor = get_tensor(state_dict.pop(self.q_scale_name)).cast("float32").reshape_([-1])
        s_scale_tensor = get_tensor(state_dict.pop(self.s_scale_name)).cast("float32").reshape_([-1])

        cache_k_scale = self.cache_quant_config.max_bound / cache_k_scale_tensor
        cache_v_scale = self.cache_quant_config.max_bound / cache_v_scale_tensor
        cache_k_out_scale = cache_k_scale_tensor / self.cache_quant_config.max_bound
        cache_v_out_scale = cache_v_scale_tensor / self.cache_quant_config.max_bound
        q_scale = self.cache_quant_config.max_bound / q_scale_tensor
        q_out_scale = q_scale_tensor / self.cache_quant_config.max_bound
        s_scale = self.cache_quant_config.max_bound / s_scale_tensor
        s_out_scale = s_scale_tensor / self.cache_quant_config.max_bound
        scaling_factor = layer.head_dim**-0.5
        q_scaling_scale = self.cache_quant_config.max_bound / (q_scale_tensor * scaling_factor)
        q_scaling_out_scale = (q_scale_tensor * scaling_factor) / self.cache_quant_config.max_bound

        layer.cache_k_scale.set_value(cache_k_scale)
        layer.cache_v_scale.set_value(cache_v_scale)
        layer.cache_k_out_scale.set_value(cache_k_out_scale)
        layer.cache_v_out_scale.set_value(cache_v_out_scale)
        layer.q_scale.set_value(q_scale)
        layer.q_out_scale.set_value(q_out_scale)
        layer.q_scaling_scale.set_value(q_scaling_scale)
        layer.q_scaling_out_scale.set_value(q_scaling_out_scale)
        layer.s_scale.set_value(s_scale)
        layer.s_out_scale.set_value(s_out_scale)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        create_weights
        """
        if self.cache_quant_config.quant_type == KvCacheQuantzationTypes.FP8_E4M3:
            layer.cache_quant_type_str = "cache_fp8_sdpa_fp8"
            layer.quant_max_bound = 240.0
            layer.quant_min_bound = -240.0
        else:
            raise NotImplementedError(f"{self.cache_quant_config.quant_type} is not implemented")

        scale_shape = [1]

        layer.cache_k_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.cache_v_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
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
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.cache_v_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        layer.q_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.q_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.q_scaling_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.q_scaling_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.s_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.s_out_scale = layer.create_parameter(
            shape=scale_shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        use for loader v0
        """
        self.prefix = layer.prefix
        self.cache_k_scale_name = layer.prefix + ".cachek_matmul.activation_scale"
        self.cache_v_scale_name = layer.prefix + ".cachev_matmul.activation_scale"
        self.q_scale_name = layer.prefix + ".q_matmul.activation_scale"
        self.s_scale_name = layer.prefix + ".s_matmul.activation_scale"

        self.load_scale(layer, state_dict)

    def process_weights_after_loading(self, layer: nn.Layer):
        """
        use for loader v1
        """
        # cache_k_out_scale is the reciprocal of cache_k_scale
        if layer.cache_k_scale._is_initialized():
            layer.cache_k_out_scale.set_value(1 / layer.cache_k_scale)  # cache_k_out_scale
        if layer.cache_v_scale._is_initialized():
            layer.cache_v_out_scale.set_value(1 / layer.cache_v_scale)

    def apply(self, layer):
        """
        apply
        """
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")
