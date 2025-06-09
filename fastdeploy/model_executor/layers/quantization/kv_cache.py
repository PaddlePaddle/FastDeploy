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
from paddle import nn
import os
import paddle
from .quant_base import QuantConfigBase, QuantMethodBase
from typing import Optional


class KvCacheQuantConfig(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, cachekv_scale_dict) -> None:
        """
        __init__
        """
        super().__init__()
        self.cachekv_scale_dict = cachekv_scale_dict

    def get_name(self) -> str:
        """
        get_name
        """
        return "kvcache"

    @classmethod
    def from_config(cls, config: dict) -> "KvCacheQuantConfig":
        """
        from_config
        """
        cachekv_scale_dict = config["cachekv_scale_dict"]
        return cls(cachekv_scale_dict)

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
        self.quant_config = quant_config

    def load_zp(self, layer: nn.Layer):
        """
        load_zp
        """
        if self.cache_k_zp_name in self.quant_config.cachekv_scale_dict:
            cache_k_zp = paddle.cast(
                paddle.to_tensor(
                    self.quant_config.cachekv_scale_dict[self.cache_k_zp_name]
                ),
                self.cache_scale_dtype,
            )
        else:
            cache_k_zp = paddle.zeros(
                (
                    [self.kv_num_heads * self.head_dim]
                    if self.quant_config.is_channel_wise
                    else [self.kv_num_heads]
                ),
                dtype=self.cache_scale_dtype,
            )
        if self.cache_v_zp_name in self.quant_config.cachekv_scale_dict:
            cache_v_zp = paddle.cast(
                paddle.to_tensor(
                    self.quant_config.cachekv_scale_dict[self.cache_v_zp_name]
                ),
                self.cache_scale_dtype,
            )
        else:
            cache_v_zp = paddle.zeros(
                (
                    [self.kv_num_heads * self.head_dim]
                    if self.quant_config.is_channel_wise
                    else [self.kv_num_heads]
                ),
                dtype=self.cache_scale_dtype,
            )
        layer.cache_k_zp.set_value(cache_k_zp)
        layer.cache_v_zp.set_value(cache_v_zp)

    def load_scale(self, layer: nn.Layer):
        """
        load_scale
        """
        if self.cache_k_scale_name in self.quant_config.cachekv_scale_dict:
            cache_k_scale = paddle.cast(
                paddle.to_tensor(
                    self.quant_config.cachekv_scale_dict[self.cache_k_scale_name]
                ),
                self.cache_scale_dtype,
            )
            cache_k_out_scale = 1.0 / cache_k_scale
        else:
            raise KeyError(
                f"{self.cache_k_scale_name} not found in scale dict")

        if self.cache_v_scale_name in self.quant_config.cachekv_scale_dict:
            cache_v_scale = paddle.cast(
                paddle.to_tensor(
                    self.quant_config.cachekv_scale_dict[self.cache_v_scale_name]
                ),
                self.cache_scale_dtype,
            )
            cache_v_out_scale = 1.0 / cache_v_scale
        else:
            raise KeyError(
                f"{self.cache_v_scale_name} not found in scale dict")

        if self.cache_v_scale_name in self.quant_config.cachekv_scale_dict:
            cache_v_scale = paddle.cast(
                paddle.to_tensor(
                    self.quant_config.cachekv_scale_dict[self.cache_v_scale_name]
                ),
                self.cache_scale_dtype,
            )
            cache_v_out_scale = 1.0 / cache_v_scale
        else:
            raise KeyError(
                f"{self.cache_v_scale_name} not found in scale dict")

        layer.cache_k_scale.set_value(cache_k_scale)
        layer.cache_v_scale.set_value(cache_v_scale)
        layer.cache_k_out_scale.set_value(cache_k_out_scale)
        layer.cache_v_out_scale.set_value(cache_v_out_scale)

    def create_scale(self, layer: nn.Layer):
        """
        create_scale
        """
        layer.cache_k_scale = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        layer.cache_v_scale = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        layer.cache_k_out_scale = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            attr=None,
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        layer.cache_v_out_scale = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            attr=None,
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )

    def create_zp(self, layer: nn.Layer):
        """
        create_zp
        """
        layer.cache_k_zp = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )
        layer.cache_v_zp = layer.create_parameter(
            shape=(
                [layer.kv_num_heads * layer.head_dim]
                if self.quant_config.is_channel_wise
                else [layer.kv_num_heads]
            ),
            dtype=self.cache_scale_dtype,
            is_bias=False,
        )

    def create_weights(self, layer: nn.Layer):
        """
        create_weights
        """
        self.prefix = layer.prefix
        self.cache_k_scale_name = layer.prefix + ".cachek_matmul.activation_quanter"
        self.cache_v_scale_name = layer.prefix + ".cachev_matmul.activation_quanter"
        self.cache_k_zp_name = layer.cache_k_scale_name + ".zero_point"
        self.cache_v_zp_name = layer.cache_v_scale_name + ".zero_point"

        layer.cache_k_zp = None
        layer.cache_v_zp = None
        layer.cache_k_scale = None
        layer.cache_v_scale = None
        layer.cache_k_out_scale = None
        layer.cache_v_out_scale = None

        self._dtype = layer._dtype
        if self._dtype != "bfloat16" and self._dtype != "float16" and self._dtype == "float32":
            raise ValueError(
                f"Just support float32, float16 and \
                    bfloat16 as default dtype, but received {self._dtype}"
            )
        self.cache_scale_dtype = (
            self._dtype if self.quant_config.use_append_attn else "float32"
        )

        if not self.quant_config.use_dynamic_cachekv_quant:
            if (
                self.quant_config.cachekv_dtype == "int8"
                or self.quant_config.cachekv_dtype == "int4"
                or self.quant_config.cachekv_dtype == "float8_e4m3fn"
            ):
                self.create_scale(layer)
                self.load_scale(layer)
                if self.quant_config.has_zero_point:
                    self.create_zp(layer)
                    self.load_zp(layer)
        layer.cache_quant_type_str = self.quant_config.cache_quant_type

    def apply(self, layer):
        """
        apply
        """
        raise RuntimeError(
            f"{self.__class__.__name__}.apply should not be called.")

