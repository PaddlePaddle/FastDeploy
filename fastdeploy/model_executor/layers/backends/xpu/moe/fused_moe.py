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

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import (
    UnquantizedFusedMoEMethod,
)
from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig
from fastdeploy.model_executor.ops.xpu import (
    ep_moe_expert_combine,
    ep_moe_expert_dispatch,
    moe_expert_ffn,
    weight_quantize_xpu,
)


class XPUMoEMethod(UnquantizedFusedMoEMethod):
    """
    XPU MOE
    """

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        for weights in [up_gate_proj_weights, down_proj_weights]:
            for idx, weight in enumerate(weights):
                weights[idx] = weight.transpose([1, 0])
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)

        layer.up_gate_proj_weight.set_value(stacked_up_gate_proj_weights)
        layer.down_proj_weight.set_value(stacked_down_proj_weights)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            gate.weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            None,  # up_gate_proj scale
            None,  # down_proj scale
            None,  # up_gate_proj_in_scale
            "",  # moe_quant_type
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.reduce_results and layer.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError


class XPUWeightOnlyMoEMethod(QuantMethodBase):
    """
    XPU Fused MoE Method.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle cutlass create weight process.
        """
        self.default_dtype = "float32"
        self.weight_dtype = "int8"

        if self.moe_quant_type in ["weight_only_int4", "w4a8"]:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size // 2,
            ]
        else:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
        if self.moe_quant_type in ["weight_only_int4", "w4a8"]:
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size // 2,
            ]
        else:
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size,
            ]

        setattr(
            layer,
            self.added_weight_attrs[0],
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_weight_attrs[1],
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        # weight_scale
        setattr(
            layer,
            self.added_scale_attrs[0],
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[1],
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle xpu load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1, -1
                )  # weight is [k,n]
                weight_list.append(quant_weight.transpose([1, 0]))  # transpose weight to [n,k]
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        XPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            gate.weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.reduce_results and layer.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out


class XPUWeightOnlyMoeEpMethod(XPUMoEMethod):
    """
    XPU Fused MoE EP Method.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)
        self.moe_quant_type = self.quant_config.algo
        self.weight_dtype = "int8"
        self.scale_dtype = "float32"

    def import_backend_ep_runner(self) -> None:
        from .ep import XPUEPDecoderRunner, XPUEPPrefillRunner

        self.EPPrefillRunner = XPUEPPrefillRunner
        self.EPDecoderRunner = XPUEPDecoderRunner

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        create weight process.
        """
        if self.moe_quant_type in ["weight_only_int8"]:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size,
            ]
        elif self.moe_quant_type in ["weight_only_int4"]:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size // 2,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate // 2,
            ]
        else:
            raise ValueError(f"Unsupported moe quant type: {self.moe_quant_type}")

        self.up_gate_proj_scale_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
        ]
        self.down_proj_scale_shape = [
            layer.num_local_experts,
            layer.hidden_size,
        ]

        setattr(
            layer,
            self.added_weight_attrs[0],
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_weight_attrs[1],
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[0],
            layer.create_parameter(
                shape=self.up_gate_proj_scale_shape,
                dtype=self.scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[1],
            layer.create_parameter(
                shape=self.down_proj_scale_shape,
                dtype=self.scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle xpu load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1, -1
                )  # quant_weight is [k,n]
                # transpose quant_weight to [n,k]
                weight_list.append(quant_weight.transpose([1, 0]))
                weight_scale_list.append(scale)

            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)
            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        gate_out = gate(x.cast("float32"))
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)
        # 2. Dynamic compute blockwise quantization scales
        # x, x_scale_tensor = fastdeploy.model_executor.ops.xpu.per_token_quant(x)
        x_scale_tensor = None
        # 3. EP Dispatch
        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            _,
        ) = self.ep_prefill_runner.dispatch(x, topk_idx, topk_weights, x_scale_tensor=x_scale_tensor)

        token_num_per_expert = recv_num_tokens_per_expert_list.numpy().tolist()
        token_all_num = sum(token_num_per_expert)

        # 4. Compute ffn
        if token_all_num > 0:
            moe_dispatch_scale = None
            (
                permute_input,
                permute_indices_per_token,
                token_num_lod,
                dst_weights,
                ffn1_act_scale_per_token,
            ) = ep_moe_expert_dispatch(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                moe_dispatch_scale,
                token_num_per_expert,
                token_all_num,
                self.moe_quant_type,
            )

            moe_ffn1_scale = None
            moe_ffn2_scale = None
            ffn_out = moe_expert_ffn(
                permute_input,
                token_num_lod,
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_weight_attrs[1]),
                None,
                None,
                moe_ffn1_scale,
                moe_ffn2_scale,
                getattr(layer, self.added_scale_attrs[0]),
                getattr(layer, self.added_scale_attrs[1]),
                None,
                None,
                self.moe_quant_type,
                -1,
                token_all_num,
            )

            # prmt back per rank
            recv_topk_weights_bf16 = recv_topk_weights.astype("bfloat16")
            tmp_ffn_out = ep_moe_expert_combine(
                ffn_out,
                permute_indices_per_token,
                recv_topk_weights_bf16,
                permute_indices_per_token.shape[0],
                ffn_out.shape[0],
                ffn_out.shape[1],
                permute_indices_per_token.shape[1],
            )

        else:
            tmp_ffn_out = paddle.empty(recv_x.shape, "bfloat16")

        # 5. EP combine
        handle = None
        return self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights)

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input,
        token_nums_per_expert,
        valid_token_num=-1,
        extra_ffn1_in_scale=None,
    ):
        """
        Calculate moe
        """
        # ffn1_in_scale = extra_ffn1_in_scale
        moe_ffn1_scale = None
        moe_ffn2_scale = None

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            getattr(layer, self.added_weight_attrs[0]),
            getattr(layer, self.added_weight_attrs[1]),
            None,
            None,
            moe_ffn1_scale,
            moe_ffn2_scale,
            getattr(layer, self.added_scale_attrs[0]),
            getattr(layer, self.added_scale_attrs[1]),
            None,
            None,
            self.moe_quant_type,
            -1,
            valid_token_num,
        )
        return ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        gate_out = gate(x.cast("float32"))

        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)

        # 2. EP Dispatch
        expertwise_scale = None
        use_fp8 = False
        (
            permute_input,
            token_nums_per_expert,
            handle,
            valid_token_num,
        ) = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights, expertwise_scale=expertwise_scale, use_fp8=use_fp8
        )

        # 3. Compute ffn
        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_nums_per_expert,
            valid_token_num,
        )

        # 4. EP combine
        return self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)
