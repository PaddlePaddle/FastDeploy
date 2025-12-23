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

import copy
from typing import Optional

import paddle

from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.ops import (
    cutlass_scaled_mm,
    scaled_fp8_quant,
)
from fastdeploy.model_executor.layers.quantization.quant_base import (
    QuantConfigBase,
    QuantMethodBase,
)
from fastdeploy.model_executor.layers.utils import per_token_cast_to_fp8
from fastdeploy.model_executor.utils import (
    TensorTracker,
    process_weight_transpose,
    set_weight_attrs,
)


class WFP8AFP8Config(QuantConfigBase):
    """
    Quantization config for weight and activation with FP8.
    """

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: list[int] = [-1, 1],
        is_checkpoint_bf16: bool = False,
    ) -> None:
        super().__init__()
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self) -> str:
        """ """
        return "wfp8afp8"

    @classmethod
    def from_config(cls, config: dict) -> "WFP8AFP8Config":
        """ """
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(is_checkpoint_bf16=is_checkpoint_bf16)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """ """
        if isinstance(layer, FusedMoE):
            from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
                Wfp8Afp8MoEMethod,
            )

            return Wfp8Afp8MoEMethod(self)
        else:
            return WFP8AFP8LinearMethod(self)


class WFP8AFP8LinearMethod(QuantMethodBase):
    """
    Weight and activation quantization method for linear layer with FP8
    """

    def __init__(
        self,
        quant_config: WFP8AFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.use_per_token_if_dynamic = True

    def create_weights(self, layer, **extra_weight_attrs):
        """ """
        weight_shape = layer.weight_shape
        weight_block_size = self.quant_config.weight_block_size
        assert len(weight_shape) == 2 and len(weight_block_size) == 2
        scale_shape = copy.deepcopy(weight_shape)
        for i in range(len(weight_shape)):
            scale_shape[i] = (
                (weight_shape[i] + weight_block_size[i] - 1) // weight_block_size[i] if weight_block_size[i] > 0 else 1
            )
        scale_shape = scale_shape[::-1]
        self.model_format = extra_weight_attrs.get("model_format")
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            weight_shape = weight_shape[::-1] if self.model_format == "torch" else weight_shape
            layer.weight = layer.create_parameter(
                shape=weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            quant_attrs = extra_weight_attrs
            if (
                isinstance(layer, MergedColumnParallelLinear)
                or isinstance(layer, QKVParallelLinear)
                or isinstance(layer, MergedReplicatedLinear)
            ):
                tensor_output_dim = (self.model_format == "torch") ^ quant_attrs.get("output_dim", True)
                quant_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=weight_shape, output_dim=tensor_output_dim),
                }
            if self.model_format == "torch" and "output_dim" in quant_attrs:
                quant_attrs["output_dim"] = not quant_attrs["output_dim"]
            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )
        else:
            layer.weight_shape.reverse()
            layer.weight_dtype = "float8_e4m3fn"
            # TODO(YuanRisheng): set weight logic should be moved to process_loaded_weights func
            self.skip_quant = False
            layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.weight_scale = layer.create_parameter(
                shape=scale_shape,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

    def process_weights_after_loading(self, layer) -> None:
        if not self.quant_config.is_checkpoint_bf16:
            return

        def _process_quantize():
            weight_tensor = layer.weight.transpose([1, 0]).contiguous()
            assert self.quant_config.weight_block_size == [-1, 1]
            qweight, weight_scale = per_token_cast_to_fp8(weight_tensor)

            if hasattr(layer.weight, "tensor_track"):
                layer.weight.tensor_track = None
            layer.weight.value().get_tensor()._clear()
            del layer.weight

            layer.weight = layer.create_parameter(
                shape=qweight.shape,
                dtype="float8_e4m3fn",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.weight_scale = layer.create_parameter(
                shape=weight_scale.shape,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.weight.copy_(qweight, False)
            layer.weight_scale.copy_(weight_scale, False)

        if self.quant_config.is_checkpoint_bf16:
            if self.model_format == "torch":
                process_weight_transpose(layer, "weight")
            _process_quantize()
        else:
            return

    def process_loaded_weights(self, layer, weights) -> None:
        """ """
        if self.skip_quant:
            weight_tensor = weights.cast(layer._dtype)
            layer.weight.set_value(weight_tensor)
            return
        if weights.dtype != paddle.float8_e4m3fn:
            self.use_per_token_if_dynamic = True
        weight_tensor = weights.transpose([1, 0]).contiguous()
        qweight, weight_scale = scaled_fp8_quant(
            weight_tensor,
            use_per_token_if_dynamic=False,
        )
        layer.weight.copy_(qweight, False)
        layer.weight_scale.set_value(weight_scale)

    def apply(self, layer, x):
        """ """
        if self.use_per_token_if_dynamic:
            out_type = x.dtype
            a_q, a_scales = scaled_fp8_quant(x, use_per_token_if_dynamic=self.use_per_token_if_dynamic)
            linear_out = cutlass_scaled_mm(
                a_q,
                layer.weight,
                a_scales,
                layer.weight_scale,
                out_type,
                layer.bias,
            )
        else:
            raise NotImplementedError
        return linear_out
