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

from typing import Optional

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.kv_cache import (
    KvCacheQuantzationTypes,
)
from fastdeploy.model_executor.layers.quantization.quant_base import (
    QuantConfigBase,
    QuantMethodBase,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import set_weight_attrs


class XPUKvCacheQuantConfig(QuantConfigBase):
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

        if self.quant_type == KvCacheQuantzationTypes.INT8:
            self.max_bound = 127.0
            self.is_channel_wise = True
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
    ) -> "XPUKvCacheQuantConfig":
        """
        from_config
        """
        return cls(kv_cache_quant_type, is_channel_wise, has_zero_point)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        get_quant_method
        """
        return XPUKVCacheMethodBase(self)


class XPUKVCacheMethodBase(QuantMethodBase):
    """
    XPUKVCacheMethodBase: XPU need scale in fp32 format but GPU define all scale in bf16 format
    """

    def __init__(
        self,
        quant_config: XPUKvCacheQuantConfig,
    ) -> None:
        """
        XPUKVCacheMethodBase __init__
        """
        super().__init__()
        self.cache_quant_config = quant_config

    def load_zp(self, layer: nn.Layer, state_dict):
        """
        load_zp
        """
        cache_k_zeropoint = get_tensor(state_dict.pop(self.cache_k_zp_name)).cast("float32")
        cache_v_zeropoint = get_tensor(state_dict.pop(self.cache_v_zp_name)).cast("float32")

        layer.cache_k_zp.set_value(cache_k_zeropoint)
        layer.cache_v_zp.set_value(cache_v_zeropoint)

    def load_scale(self, layer: nn.Layer, state_dict):
        """
        load_scale
        """

        cache_k_scale_tensor = get_tensor(state_dict.pop(self.cache_k_scale_name)).cast("float32").reshape_([-1])
        cache_v_scale_tensor = get_tensor(state_dict.pop(self.cache_v_scale_name)).cast("float32").reshape_([-1])

        if self.cache_quant_config.quant_type == KvCacheQuantzationTypes.INT8:
            # cache_k_scale and cache_v_scale are used to quantize the KV Cache, while cache_k_out_scale and cache_v_out_scale are used for inverse quantization
            cache_k_scale = self.cache_quant_config.max_bound / cache_k_scale_tensor
            cache_v_scale = self.cache_quant_config.max_bound / cache_v_scale_tensor
            cache_k_out_scale = cache_k_scale_tensor
            cache_v_out_scale = cache_v_scale_tensor
        else:
            raise NotImplementedError(f"{self.cache_quant_config.quant_type} is not implemented")

        # W4A8 model need kv_scale in bf16 format
        layer.cache_k_scale.set_value(paddle.cast(cache_k_scale, paddle.get_default_dtype()))
        layer.cache_v_scale.set_value(paddle.cast(cache_v_scale, paddle.get_default_dtype()))

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
        else:
            raise NotImplementedError(f"{self.cache_quant_config.quant_type} is not implemented")

        scale_shape = [layer.fd_config.model_config.num_key_value_heads]
        if self.cache_quant_config.is_channel_wise:
            scale_shape = [layer.kv_num_heads * layer.head_dim]
            # Custom weight_loader for C8+TP: the safetensors scale/zp shape is
            # [1, num_kv_heads, 1, head_dim]. We must split along the kv_heads
            # dimension (dim=1), not the last dimension. The default_weight_loader
            # treats output_dim as boolean and always splits along dim=-1, which
            # is incorrect for 4D tensors where we need to split along dim=1.
            fd_config = layer.fd_config
            total_kv_heads = fd_config.model_config.num_key_value_heads
            tp_size = fd_config.parallel_config.tensor_parallel_size
            tp_rank = fd_config.parallel_config.tensor_parallel_rank
            max_bound = self.cache_quant_config.max_bound

            def _kv_scale_weight_loader(
                param,
                loaded_weight,
                shard_id=None,
                _total_kv_heads=total_kv_heads,
                _tp_size=tp_size,
                _tp_rank=tp_rank,
                _max_bound=max_bound,
            ):
                loaded_weight = get_tensor(loaded_weight).cast("float32")
                # TP split along kv_heads dimension
                if _tp_size > 1 and not fd_config.load_config.is_pre_sharded:
                    head_dim = loaded_weight.numel() // _total_kv_heads
                    loaded_weight = loaded_weight.reshape([_total_kv_heads, head_dim])
                    assert (
                        _total_kv_heads % _tp_size == 0
                    ), f"num_kv_heads ({_total_kv_heads}) must be divisible by tp_size ({_tp_size})"
                    kv_heads_per_rank = _total_kv_heads // _tp_size
                    start = _tp_rank * kv_heads_per_rank
                    end = start + kv_heads_per_rank
                    loaded_weight = loaded_weight[start:end, :]
                loaded_weight = paddle.clip(loaded_weight, min=1e-8)
                loaded_weight = (_max_bound / loaded_weight).reshape(param.shape).cast(param.dtype)
                param.copy_(loaded_weight, False)

            def _kv_zp_weight_loader(
                param, loaded_weight, shard_id=None, _total_kv_heads=total_kv_heads, _tp_size=tp_size, _tp_rank=tp_rank
            ):
                loaded_weight = get_tensor(loaded_weight).cast(param.dtype)
                # TP split along kv_heads dimension
                if _tp_size > 1 and not fd_config.load_config.is_pre_sharded:
                    head_dim = loaded_weight.numel() // _total_kv_heads
                    loaded_weight = loaded_weight.reshape([_total_kv_heads, head_dim])
                    kv_heads_per_rank = _total_kv_heads // _tp_size
                    start = _tp_rank * kv_heads_per_rank
                    end = start + kv_heads_per_rank
                    loaded_weight = loaded_weight[start:end, :]
                loaded_weight = loaded_weight.reshape(param.shape)
                param.copy_(loaded_weight, False)

            scale_weight_attrs = {**extra_weight_attrs, "weight_loader": _kv_scale_weight_loader}
            zp_weight_attrs = {**extra_weight_attrs, "weight_loader": _kv_zp_weight_loader}
        else:
            scale_weight_attrs = extra_weight_attrs
            zp_weight_attrs = extra_weight_attrs

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
                **scale_weight_attrs,
            },
        )
        set_weight_attrs(
            layer.cache_v_scale,
            {
                **scale_weight_attrs,
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

        if self.cache_quant_config.has_zero_point:
            layer.cache_k_zp = layer.create_parameter(
                shape=scale_shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.cache_v_zp = layer.create_parameter(
                shape=scale_shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            set_weight_attrs(
                layer.cache_k_zp,
                {
                    **zp_weight_attrs,
                },
            )
            set_weight_attrs(
                layer.cache_v_zp,
                {
                    **zp_weight_attrs,
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
        # cache_k_out_scale is the reciprocal of cache_k_scale
        if layer.cache_k_scale._is_initialized():
            layer.cache_k_out_scale.set_value(
                self.cache_quant_config.max_bound / layer.cache_k_scale.cast("float32").reshape_([-1])
            )
        if layer.cache_v_scale._is_initialized():
            layer.cache_v_out_scale.set_value(
                self.cache_quant_config.max_bound / layer.cache_v_scale.cast("float32").reshape_([-1])
            )

    def apply(self, layer):
        """
        apply
        """
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")
