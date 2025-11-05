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
from paddle import nn

from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WeightOnlyConfig,
    WeightOnlyLinearMethod,
)
from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu
from fastdeploy.model_executor.utils import TensorTracker, free_tensor, set_weight_attrs


class XPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on XPU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)
        self.quant_config.weight_only_linear_arch = -1

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs) -> None:
        """
        Create weights for linear layer on XPU
        """
        # The scale shape should be equal to the output dim of weight using Per-Channel Quantization.
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
            # The scale shape should be equal to the output dim of weight using Per-Channel Quantization.
            weight_scale_shape = [layer.weight_shape[1]]
            layer.weight_shape.reverse()
            if self.quant_config.name() == "weight_only_int4":
                layer.weight_shape[0] //= 2
            layer.weight_dtype = "int8"
            layer.weight = layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.weight_scale = layer.create_parameter(
                shape=weight_scale_shape,
                dtype="float32",
                is_bias=False,
            )

    def process_loaded_weights(self, layer: nn.Layer, weight: paddle.Tensor) -> None:
        """
        loaded_weights using xpu special quantization
        """
        k, n = weight.shape
        quanted_weight_tensors = []
        weight_scale_tensors = []
        offset = 30720
        for i in range(0, n, offset):
            end_n = min(i + offset, n)
            weight_i = weight[:, i:end_n]
            quanted_weight_tensor, weight_scale_tensor = weight_quantize_xpu(weight_i, self.quant_config.algo, -1, -1)
            quanted_weight_tensors.append(quanted_weight_tensor)
            weight_scale_tensors.append(weight_scale_tensor)
        quanted_weight_tensor = paddle.concat(quanted_weight_tensors, axis=1)
        weight_scale_tensor = paddle.concat(weight_scale_tensors, axis=0)
        layer.weight.set_value(paddle.transpose(quanted_weight_tensor, [1, 0]))
        layer.weight_scale.set_value(weight_scale_tensor)

    def process_weights_after_loading(self, layer) -> None:
        if not self.quant_config.is_checkpoint_bf16:
            return

        quanted_weight_tensor, weight_scale_tensor = weight_quantize_xpu(layer.weight, self.quant_config.algo, -1, -1)

        free_tensor(layer.weight)

        layer.weight = layer.create_parameter(
            shape=quanted_weight_tensor.shape[::-1],
            dtype="int8",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale = layer.create_parameter(
            shape=weight_scale_tensor.shape,
            dtype=weight_scale_tensor.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight.set_value(paddle.transpose(quanted_weight_tensor, [1, 0]))
        layer.weight_scale.copy_(weight_scale_tensor, False)
