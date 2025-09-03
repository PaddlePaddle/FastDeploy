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

from abc import abstractmethod

import paddle
from paddle import nn

from fastdeploy.model_executor.utils import default_weight_loader, set_weight_attrs
from fastdeploy.platforms import current_platform

from ..quantization.quant_base import QuantMethodBase


class MoEMethodBase(QuantMethodBase):
    """ """

    def __init__(self, quant_config):
        super().__init__()
        if quant_config is None:
            self.moe_quant_type = "w16a16"
        else:
            self.quant_config = quant_config
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.pack_num = 1

    def init_ep(self, layer: nn.Layer) -> None:
        """
        Init EP related module
        """
        if layer.ep_size > 1:
            if layer.fd_config.parallel_config.splitwise_role == "mixed":
                from .ep import EPDecoderRunner, EPPrefillRunner

                self.ep_prefill_runner = EPPrefillRunner(
                    layer.top_k,
                    layer.hidden_size,
                    layer.num_experts,
                    layer.fd_config.parallel_config.splitwise_role,
                    layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                    layer.ep_size,
                    layer.ep_rank,
                    layer.fd_config.model_config.redundant_experts_num,
                    ep_group=layer.fd_config.parallel_config.ep_group,
                )
                self.ep_decoder_runner = EPDecoderRunner(
                    layer.top_k,
                    layer.hidden_size,
                    layer.num_experts,
                    layer.fd_config.parallel_config.splitwise_role,
                    layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                    layer.ep_size,
                    layer.ep_rank,
                    layer.fd_config.model_config.redundant_experts_num,
                    ep_group=layer.fd_config.parallel_config.ep_group,
                )
            else:
                if layer.fd_config.parallel_config.moe_phase.phase == "prefill":
                    from .ep import EPPrefillRunner

                    self.ep_prefill_runner = EPPrefillRunner(
                        layer.top_k,
                        layer.hidden_size,
                        layer.num_experts,
                        layer.fd_config.parallel_config.splitwise_role,
                        layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                        layer.ep_size,
                        layer.ep_rank,
                        layer.fd_config.model_config.redundant_experts_num,
                        ep_group=layer.fd_config.parallel_config.ep_group,
                    )
                else:
                    from .ep import EPDecoderRunner

                    self.ep_decoder_runner = EPDecoderRunner(
                        layer.top_k,
                        layer.hidden_size,
                        layer.num_experts,
                        layer.fd_config.parallel_config.splitwise_role,
                        layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                        layer.ep_size,
                        layer.ep_rank,
                        layer.fd_config.model_config.redundant_experts_num,
                        ep_group=layer.fd_config.parallel_config.ep_group,
                    )

    def process_loaded_weights(self, layer, weights) -> None:
        """
        process_loaded_weights
        """
        pass

    def check(self, layer: nn.Layer, up_gate_proj_weights, down_proj_weights):
        """
        check layer is valid for this method
        """
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size // self.pack_num,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size // self.pack_num,
            layer.hidden_size,
        ]

    @abstractmethod
    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        raise NotImplementedError

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        if layer.ep_size > 1:
            if layer.fd_config.parallel_config.moe_phase.phase == "prefill":
                if layer.fd_config.parallel_config.splitwise_role == "mixed":
                    self.ep_prefill_runner.clean_low_latency_buffer()
                return self.apply_ep_prefill(layer, x, gate)
            else:
                if layer.fd_config.parallel_config.splitwise_role == "mixed":
                    self.ep_decoder_runner.clean_low_latency_buffer()
                return self.apply_ep_decode(layer, x, gate)
        else:
            return self.apply_tp(layer, x, gate)


class UnquantizedFusedMoEMethod(MoEMethodBase):
    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):

        if current_platform.is_cuda():
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size * 2,
            ]
            self.down_proj_weight_shape = [layer.num_local_experts, layer.moe_intermediate_size, layer.hidden_size]
            extra_weight_attrs = {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 1, "down": 0, "up": 1}}
        else:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
            self.down_proj_weight_shape = [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size]
            extra_weight_attrs = {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}}

        layer.up_gate_proj_weight = layer.create_parameter(
            shape=self.up_gate_proj_weight_shape,
            dtype=layer.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        layer.down_proj_weight = layer.create_parameter(
            shape=self.down_proj_weight_shape,
            dtype=layer.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        set_weight_attrs(
            layer.up_gate_proj_weight,
            {
                "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                "model_format": extra_weight_attrs.get("model_format", ""),
            },
        )
        set_weight_attrs(
            layer.down_proj_weight,
            {
                "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                "model_format": extra_weight_attrs.get("model_format", ""),
            },
        )
