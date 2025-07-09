"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddleformers.utils.log import logger

from .fused_moe_backend_base import MoEMethodBase

class XPUMoEMethod(MoEMethodBase):
    """
    XPU MOE
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        # bf16
        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)
        for weights in [ffn1_weights, ffn2_weights]:
            for idx, weight in enumerate(weights):
                weights[idx] = weight.transpose([1, 0])
        stacked_ffn1_weights = paddle.stack(ffn1_weights, axis=0)
        stacked_ffn2_weights = paddle.stack(ffn2_weights, axis=0)
        for idx, weight_tensor in enumerate(
            [stacked_ffn1_weights, stacked_ffn2_weights]):
            weight_name = self.added_weight_attrs[idx]
            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(weight_tensor)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,  # ffn1 bias
            None,  # ffn2 bias
            None,  # ffn1 scale
            None,  # ffn2 scale
            None, # ffn1_in_scale
            "", # moe_quant_type
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError
