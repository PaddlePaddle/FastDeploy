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

from fastdeploy.model_executor.utils import (
    TensorTracker,
    default_weight_loader,
    free_tensor,
    set_weight_attrs,
    weight_fully_copied,
)
from fastdeploy.platforms import current_platform

from ..quantization.quant_base import QuantMethodBase


class MoEMethodBase(QuantMethodBase):
    """ """

    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config
        if self.quant_config is None:
            self.moe_quant_type = "w16a16"
        elif hasattr(quant_config, "algo"):
            self.moe_quant_type = quant_config.algo
        else:
            self.moe_quant_type = quant_config.name()
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.added_in_scale_attrs = [
            "up_gate_proj_in_scale",
            "down_proj_in_scale",
        ]
        self.pack_num = 1
        self.ep_prefill_runner = None
        self.ep_decoder_runner = None

    def import_backend_ep_runner(self) -> None:
        """
        Different platform has different ep runner. Override this method to import the corresponding EP runner.
        """
        from .ep import EPDecoderRunner, EPPrefillRunner

        self.EPPrefillRunner = EPPrefillRunner
        self.EPDecoderRunner = EPDecoderRunner

    def init_ep(self, layer: nn.Layer) -> None:
        """
        Initialize EP (Expert Parallel) related modules.
        """
        if layer.ep_size <= 1:
            return

        # Lazy import to avoid circular dependency or unnecessary loading
        self.import_backend_ep_runner()

        # Common arguments for both runners
        common_args = {
            "top_k": layer.top_k,
            "hidden_size": layer.hidden_size,
            "num_experts": layer.num_experts,
            "splitwise_role": layer.fd_config.scheduler_config.splitwise_role,
            "num_max_dispatch_tokens_per_rank": layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
            "ep_size": layer.ep_size,
            "ep_rank": layer.ep_rank,
            "redundant_experts_num": layer.fd_config.model_config.redundant_experts_num,
            "ep_group": layer.fd_config.parallel_config.ep_group,
        }

        config = layer.fd_config
        splitwise_role = config.scheduler_config.splitwise_role
        load_strategy = config.load_config.load_strategy

        # For "mixed" splitwise role: conditionally initialize both or none
        if splitwise_role == "mixed":
            if load_strategy == "meta":
                # for RL init model without deepep buff
                return
            else:
                if current_platform.is_cuda():
                    self.ep_prefill_runner = self.EPPrefillRunner(
                        **common_args,
                        use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
                    )
                    self.ep_decoder_runner = self.EPDecoderRunner(
                        **common_args,
                        use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
                    )
                else:
                    self.ep_prefill_runner = self.EPPrefillRunner(**common_args)
                    self.ep_decoder_runner = self.EPDecoderRunner(**common_args)
            return

        # For non-mixed ep
        phase = config.model_config.moe_phase.phase
        if current_platform.is_cuda():
            if phase == "prefill":
                self.ep_prefill_runner = self.EPPrefillRunner(
                    **common_args,
                    use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
                )
            else:
                self.ep_decoder_runner = self.EPDecoderRunner(
                    **common_args,
                    use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
                )
        else:
            if phase == "prefill":
                self.ep_prefill_runner = self.EPPrefillRunner(**common_args)
            else:
                self.ep_decoder_runner = self.EPDecoderRunner(**common_args)

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
            is_moe_start_layer = layer.layer_idx == layer.fd_config.model_config.moe_layer_start_index
            if layer.fd_config.model_config.moe_phase.phase == "prefill":
                if layer.fd_config.scheduler_config.splitwise_role == "mixed" and is_moe_start_layer:
                    self.ep_prefill_runner.clean_low_latency_buffer()
                return self.apply_ep_prefill(layer, x, gate)
            else:
                if layer.fd_config.scheduler_config.splitwise_role == "mixed" and is_moe_start_layer:
                    self.ep_decoder_runner.clean_low_latency_buffer()
                return self.apply_ep_decode(layer, x, gate)
        else:
            return self.apply_tp(layer, x, gate)


class UnquantizedFusedMoEMethod(MoEMethodBase):
    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        num_experts = extra_weight_attrs.pop("num_experts")
        hidden_size = extra_weight_attrs.pop("hidden_size")
        moe_intermediate_size = extra_weight_attrs.pop("moe_intermediate_size")
        self.model_format = extra_weight_attrs.get("model_format")
        if current_platform.is_cuda() and self.model_format != "torch":
            self.up_gate_proj_weight_shape = [num_experts, hidden_size, moe_intermediate_size * 2]
            self.down_proj_weight_shape = [num_experts, moe_intermediate_size, hidden_size]
            extra_weight_attrs = {
                **(extra_weight_attrs or {}),
                "SHARD_ID_TO_SHARDED_DIM": {"gate": 1, "down": 0, "up": 1},
            }
        else:
            self.up_gate_proj_weight_shape = [num_experts, moe_intermediate_size * 2, hidden_size]
            self.down_proj_weight_shape = [num_experts, hidden_size, moe_intermediate_size]
            extra_weight_attrs = {
                **(extra_weight_attrs or {}),
                "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
            }

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
        extra_weight_attrs["weight_loader"] = extra_weight_attrs.get(
            "weight_loader", default_weight_loader(layer.fd_config)
        )
        if self.model_format != "torch":
            up_gate_proj_attrs = extra_weight_attrs
            down_proj_attrs = extra_weight_attrs
        else:
            up_gate_proj_attrs = {
                **extra_weight_attrs,
                "tensor_track": TensorTracker(
                    shape=layer.up_gate_proj_weight.shape,
                    output_dim=extra_weight_attrs["SHARD_ID_TO_SHARDED_DIM"]["gate"],
                ),
            }
            down_proj_attrs = {
                **extra_weight_attrs,
                "tensor_track": TensorTracker(
                    shape=layer.down_proj_weight.shape,
                    output_dim=extra_weight_attrs["SHARD_ID_TO_SHARDED_DIM"]["down"],
                ),
            }
        set_weight_attrs(
            layer.up_gate_proj_weight,
            up_gate_proj_attrs,
        )
        set_weight_attrs(
            layer.down_proj_weight,
            down_proj_attrs,
        )

        if layer.with_bias:
            # only pt model now
            layer.up_gate_proj_bias = layer.create_parameter(
                shape=[num_experts, moe_intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_bias = layer.create_parameter(
                shape=[num_experts, hidden_size],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            set_weight_attrs(
                layer.up_gate_proj_bias,
                {
                    "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                },
            )
            set_weight_attrs(
                layer.down_proj_bias,
                {
                    "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                },
            )

    def process_weights_after_loading(self, layer):
        if self.model_format != "torch":
            return
        if not weight_fully_copied(layer.up_gate_proj_weight) or not weight_fully_copied(layer.down_proj_weight):
            return
        up_gate_proj_weight_transpose = layer.up_gate_proj_weight.transpose([0, 2, 1])
        down_proj_weight_transpose = layer.down_proj_weight.transpose([0, 2, 1])
        up_gate_proj = layer.create_parameter(
            shape=up_gate_proj_weight_transpose.shape,
            dtype=up_gate_proj_weight_transpose.dtype,
            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02),
            is_bias=False,
        )
        up_gate_proj.copy_(up_gate_proj_weight_transpose, False)
        free_tensor(layer.up_gate_proj_weight)
        layer.up_gate_proj_weight = up_gate_proj
        down_proj = layer.create_parameter(
            shape=down_proj_weight_transpose.shape,
            dtype=down_proj_weight_transpose.dtype,
            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02),
            is_bias=False,
        )
        down_proj.copy_(down_proj_weight_transpose, False)
        free_tensor(layer.down_proj_weight)
        layer.down_proj_weight = down_proj
