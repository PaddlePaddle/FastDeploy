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

import os
from abc import abstractmethod
from typing import Optional

import paddle
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy import envs
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.utils import TensorTracker, free_tensor, set_weight_attrs
from fastdeploy.platforms import current_platform

from ..moe import FusedMoE
from ..utils import get_tensor
from .quant_base import QuantConfigBase, QuantMethodBase


class WeightOnlyConfig(QuantConfigBase):
    """
    Quantization config for weight only
    Args:
        algo: The quant algorithm("weight_only_int8" or "weight_only_int4") used for weight only linear layer
    """

    def __init__(
        self,
        algo: str,
        is_checkpoint_bf16: bool = False,
    ) -> None:
        super().__init__()
        self.algo = algo
        # arch (int): The compute arch for target device. For example, A100 is 80, v100 is 70,
        # if you do not assign arch, we will get arch from your device, default: None.
        self.weight_only_linear_arch = os.getenv("FLAGS_weight_only_linear_arch")
        if self.weight_only_linear_arch is not None:
            self.weight_only_linear_arch = int(self.weight_only_linear_arch)
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self) -> str:
        return "weight_only"

    @classmethod
    def from_config(cls, config: dict) -> "WeightOnlyConfig":
        algo = config["algo"]
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(algo, is_checkpoint_bf16)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if current_platform.is_xpu():
            from fastdeploy.model_executor.layers.backends import (
                XPUWeightOnlyLinearMethod,
            )
            from fastdeploy.model_executor.layers.moe.fused_moe_xpu_backend import (
                XPUWeightOnlyMoEMethod,
            )

            if isinstance(layer, FusedMoE):
                return XPUWeightOnlyMoEMethod(self)
            else:
                return XPUWeightOnlyLinearMethod(self)
        elif current_platform.is_gcu():
            from fastdeploy.model_executor.layers.backends import (
                GCUWeightOnlyLinearMethod,
                GCUWeightOnlyMoEMethod,
            )

            if isinstance(layer, FusedMoE):
                return GCUWeightOnlyMoEMethod(self)
            else:
                return GCUWeightOnlyLinearMethod(self)
        elif current_platform.is_dcu():
            if isinstance(layer, FusedMoE):
                from fastdeploy.model_executor.layers.backends import (
                    DCUTritonWeightOnlyMoEMethod,
                )

                return DCUTritonWeightOnlyMoEMethod(self)
            else:
                from fastdeploy.model_executor.layers.backends import (
                    DCUWeightOnlyLinearMethod,
                )

                return DCUWeightOnlyLinearMethod(self)
        elif current_platform.is_maca():
            if isinstance(layer, FusedMoE):
                from fastdeploy.model_executor.layers.backends import (
                    MetaxTritonWeightOnlyMoEMethod,
                )

                return MetaxTritonWeightOnlyMoEMethod(self)
            else:

                return GPUWeightOnlyLinearMethod(self)
        else:
            if isinstance(layer, FusedMoE):
                if layer.use_method == "cutlass":
                    from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
                        CutlassWeightOnlyMoEMethod,
                    )

                    return CutlassWeightOnlyMoEMethod(self)
                elif layer.use_method == "triton":
                    from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
                        TritonWeightOnlyMoEMethod,
                    )

                    return TritonWeightOnlyMoEMethod(self)
                elif layer.use_method == "marlin":
                    from fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend import (
                        MarlinWeightOnlyMoEMethod,
                    )

                    return MarlinWeightOnlyMoEMethod(self)
                else:
                    raise ValueError(f"Unsupported MOE backend {layer.use_method}")
            else:
                from fastdeploy.model_executor.layers.quantization.ops.machete_mm import (
                    _ENABLE_MACHETE,
                )

                if (
                    _ENABLE_MACHETE
                    and envs.FD_USE_MACHETE == "1"
                    and layer.weight_shape[1]
                    and layer.weight_shape[1] % 128 == 0
                    and not layer.add_bias
                ):
                    return MacheteWeightOnlyLinearMethod(self)
                return GPUWeightOnlyLinearMethod(self)


class WINT8Config(WeightOnlyConfig):
    """
    weight only int8 config
    """

    def __init__(self, is_checkpoint_bf16: bool = False) -> None:
        super().__init__("weight_only_int8", is_checkpoint_bf16)

    @classmethod
    def from_config(cls, config: dict) -> "WINT8Config":
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(is_checkpoint_bf16)

    def name(self) -> str:
        return "wint8"


class WINT4Config(WeightOnlyConfig):
    """
    weight only int4 config
    """

    def __init__(
        self,
        is_checkpoint_bf16: bool = False,
    ) -> None:
        super().__init__("weight_only_int4", is_checkpoint_bf16)

    @classmethod
    def from_config(cls, config: dict) -> "WINT4Config":
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(is_checkpoint_bf16)

    def name(self) -> str:
        return "wint4"


class WeightOnlyLinearMethod(QuantMethodBase):
    """
    Weight only quantization method for linear layer
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer, **extra_weight_attrs):
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            layer.weight = layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            extra_weight_attrs["weight_need_transpose"] = extra_weight_attrs.get("model_format") == "torch"
            quant_attrs = extra_weight_attrs
            if (
                isinstance(layer, MergedColumnParallelLinear)
                or isinstance(layer, QKVParallelLinear)
                or isinstance(layer, MergedReplicatedLinear)
            ):
                quant_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(
                        shape=layer.weight_shape, output_dim=extra_weight_attrs.get("output_dim", True)
                    ),
                }
            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )
        else:
            if isinstance(self, MacheteWeightOnlyLinearMethod):
                weight_scale_shape = [1, layer.weight_shape[1]]
                if self.quant_config.name() == "wint4":
                    layer.weight_shape[0] //= 8
                else:
                    layer.weight_shape[0] //= 4
                layer.weight_dtype = "int32"
            else:
                # The scale shape should be equal to the output dim of weight using Per-Channel Quantization.
                weight_scale_shape = [layer.weight_shape[1]]
                layer.weight_shape.reverse()
                if self.quant_config.name() == "wint4":
                    layer.weight_shape[0] //= 2
                layer.weight_dtype = "int8"

            layer.weight = layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            output_dim = extra_weight_attrs.get("output_dim")
            output_dim = not output_dim
            weight_loader = extra_weight_attrs.get("weight_loader")
            set_weight_attrs(
                layer.weight,
                {
                    "weight_loader": weight_loader,
                    "output_dim": output_dim,
                    "weight_need_transpose": not extra_weight_attrs.get("model_format") == "torch",
                },
            )

            layer.weight_scale = layer.create_parameter(
                shape=weight_scale_shape,
                dtype=layer._dtype,
                is_bias=False,
            )

            set_weight_attrs(
                layer.weight_scale,
                {
                    "weight_loader": weight_loader,
                    "output_dim": output_dim,
                },
            )

    def process_weights_after_loading(self, layer) -> None:
        if not self.quant_config.is_checkpoint_bf16:
            return
        if isinstance(self, MacheteWeightOnlyLinearMethod):
            from fastdeploy.model_executor.layers.quantization.ops import (
                machete_quantize_and_pack,
            )

            quanted_weight_tensor, weight_scale_tensor = machete_quantize_and_pack(
                w=layer.weight,
                atype=layer._dtype,
                quant_type="uint4b8" if self.quant_config.name() == "wint4" else "uint8b128",
            )
        else:
            quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                layer.weight,
                algo=self.quant_config.algo,
                arch=self.quant_config.weight_only_linear_arch,
            )

        free_tensor(layer.weight)

        layer.weight = layer.create_parameter(
            shape=quanted_weight_tensor.shape,
            dtype="int8" if not isinstance(self, MacheteWeightOnlyLinearMethod) else "int32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale = layer.create_parameter(
            shape=weight_scale_tensor.shape,
            dtype=layer._dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale.copy_(weight_scale_tensor, False)

    @abstractmethod
    def process_loaded_weights(self, layer, weights) -> None:
        raise NotImplementedError

    def apply(self, layer, x):
        linear_out = weight_only_linear(
            x,
            weight=layer.weight,
            bias=layer.bias if layer.add_bias else None,
            weight_scale=layer.weight_scale,
            weight_dtype=("int8" if self.quant_config.name() == "wint8" else "int4"),
            arch=self.quant_config.weight_only_linear_arch,
        )
        return linear_out


class GPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on GPU
    The weights are loaded in the BF16 numerical format. After loading, the quantization coefficients will be computed,
    and the weights will be quantized to int8 or int4.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def process_prequanted_weights(self, layer, state_dict, is_rearrange: bool = False) -> None:
        """
        Process pre-quantized weights before applying them to the model
        Args:
            layer: The layer that owns the weights
            quant_weight: The quantized weights
            weight_scale: The scale of the quantized weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))
        layer.weight.set_value(quant_weight)
        layer.weight_scale.set_value(weight_scale.astype(paddle.get_default_dtype()))

    def process_loaded_weights(self, layer, weight) -> None:

        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
            weight,
            algo=self.quant_config.algo,
            arch=self.quant_config.weight_only_linear_arch,
        )
        if current_platform.is_maca():
            quanted_weight_tensor = paddle.transpose(quanted_weight_tensor, [1, 0])
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor.astype(paddle.get_default_dtype()))


class MacheteWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on GPU using Machete
    The weights are loaded in the BF16 numerical format. After loading, the quantization coefficients will be computed,
    and the weights will be quantized to int8 or int4.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def process_prequanted_weights(self, layer, state_dict) -> None:
        pass

    def process_loaded_weights(self, layer, weight) -> None:
        from fastdeploy.model_executor.layers.quantization.ops import (
            machete_quantize_and_pack,
        )

        quanted_weight_tensor, weight_scale_tensor = machete_quantize_and_pack(
            w=weight,
            atype=layer._dtype,
            quant_type="uint4b8" if self.quant_config.name() == "wint4" else "uint8b128",
        )
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor.astype(paddle.get_default_dtype()))

    def apply(self, layer, x):
        assert layer.bias is None, "Machete weight only linear method does not support bias."
        from fastdeploy.model_executor.layers.quantization.ops import machete_wint_mm

        linear_out = machete_wint_mm(
            x,
            w_prepack=layer.weight,
            w_g_s=layer.weight_scale,
            weight_dtype="uint4b8" if self.quant_config.name() == "wint4" else "uint8b128",
        )

        return linear_out
