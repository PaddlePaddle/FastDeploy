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

import paddle
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVGateParallelLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WeightOnlyConfig,
    WeightOnlyLinearMethod,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import (
    TensorTracker,
    free_tensor,
    process_weight_transpose,
    set_weight_attrs,
)


class IluvatarWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)
        self.quant_config.weight_only_linear_arch = -1
        self.group_size = -1

    def create_weights(self, layer, **extra_weight_attrs):
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        self.model_format = extra_weight_attrs.get("model_format")
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            weight_shape = layer.weight_shape[::-1] if self.model_format == "torch" else layer.weight_shape
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
                or isinstance(layer, QKVGateParallelLinear)
            ):
                # Only MergedReplicatedLinear uses the default outdim.
                tensor_output_dim = (self.model_format == "torch") ^ quant_attrs.get("output_dim", True)
                quant_attrs = {
                    **quant_attrs,
                    "tensor_track": TensorTracker(shape=weight_shape, output_dim=tensor_output_dim),
                }

            if self.model_format == "torch" and "output_dim" in quant_attrs:
                quant_attrs["output_dim"] = not quant_attrs["output_dim"]

            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )
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

            if "output_dim" in extra_weight_attrs:
                extra_weight_attrs["output_dim"] = not extra_weight_attrs["output_dim"]
            set_weight_attrs(
                layer.weight,
                extra_weight_attrs,
            )

            layer.weight_scale = layer.create_parameter(
                shape=weight_scale_shape,
                dtype=layer._dtype,
                is_bias=False,
            )

            set_weight_attrs(
                layer.weight_scale,
                extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer) -> None:
        def _process_quantize():
            quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                layer.weight,
                algo=self.quant_config.algo,
                arch=self.quant_config.weight_only_linear_arch,
            )

            free_tensor(layer.weight)

            layer.weight = layer.create_parameter(
                shape=quanted_weight_tensor.shape,
                dtype="int8",
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

        if self.quant_config.is_checkpoint_bf16:
            if self.model_format == "torch":
                process_weight_transpose(layer, "weight")
            _process_quantize()
        else:
            return

    def process_loaded_weights(self, layer, weight) -> None:

        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
            weight,
            algo=self.quant_config.algo,
            arch=self.quant_config.weight_only_linear_arch,
        )
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor.astype(paddle.get_default_dtype()))

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

    def apply(self, layer, x):
        linear_out = weight_only_linear(
            x,
            weight=layer.weight,
            bias=layer.bias if layer.with_bias else None,
            weight_scale=layer.weight_scale,
            weight_dtype=("int8" if self.quant_config.name() == "wint8" else "int4"),
            arch=self.quant_config.weight_only_linear_arch,
        )
        return linear_out
