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
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.worker.experts_manager import RedundantExpertManger


def get_moe_method():
    """
    return moe method based on device platform
    """
    from fastdeploy.platforms import current_platform

    if current_platform.is_cuda():
        from .fused_moe_cutlass_backend import CutlassMoEMethod

        return CutlassMoEMethod(None)
    elif current_platform.is_xpu():
        from .fused_moe_xpu_backend import XPUMoEMethod

        return XPUMoEMethod(None)
    elif current_platform.is_gcu():
        from fastdeploy.model_executor.layers.backends import GCUFusedMoeMethod

        return GCUFusedMoeMethod(None)
    raise NotImplementedError


class FusedMoE(nn.Layer):
    """
    FusedMoE is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        fd_config,
        reduce_results: bool = True,
        moe_intermediate_size: int = -1,
        num_experts: int = -1,
        expert_id_offset: int = 0,
        top_k: int = -1,
        topk_method: str = "",
        topk_group: int = -1,
        n_group: int = -1,
        routed_scaling_factor: float = 1.0,
        layer_idx: int = -1,
        moe_tag: str = "",
        weight_key_map: dict = {},
    ):
        """
        Initialize the Moe layer with given parameters.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        super().__init__()

        self.fd_config = fd_config
        self.layer_idx = layer_idx
        self.reduce_results = reduce_results

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.ep_rank = fd_config.parallel_config.expert_parallel_rank

        assert (self.tp_size >= 1 and self.ep_size == 1) or (
            self.tp_size == 1 and self.ep_size > 1
        ), "MoE only support parallelism on TP or EP dimension."

        self.hidden_size = fd_config.model_config.hidden_size
        self.num_experts = num_experts
        self.num_local_experts = self.num_experts // self.ep_size

        self.moe_intermediate_size = moe_intermediate_size // self.tp_size

        self.top_k = top_k
        self.weight_key_map = weight_key_map

        self.use_method = envs.FD_MOE_BACKEND.lower()
        self.gate_correction_bias = None
        self.moe_tag = moe_tag
        if self.ep_size > 1:
            expert_id_offset = expert_id_offset + self.ep_rank * self.num_local_experts

        self.expert_id_offset = expert_id_offset

        # used for deepseek_v3
        self.topk_method = topk_method
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor

        moe_quant_config = fd_config.quant_config
        self.moe_quant_type = None
        if moe_quant_config:
            self.quant_method = moe_quant_config.get_quant_method(self)
            self.moe_quant_type = moe_quant_config.name()
        else:
            # now, no quant method(w_fp16 a_fp16) can't get from quant_config, we will optimize it in future
            self.quant_method = get_moe_method()

        self.redundant_table_manger = None
        if self.ep_size > 1:
            if fd_config.model_config.enable_redundant_experts is True:
                self.redundant_table_manger = RedundantExpertManger(
                    n_routed_experts=fd_config.model_config.moe_num_experts,
                    num_hidden_layers=fd_config.model_config.num_hidden_layers,
                    redundant_experts_num=fd_config.model_config.redundant_experts_num,
                    ep_size=self.ep_size,
                )
            self.quant_method.init_ep(self)

        if fd_config.load_config.dynamic_load_weight:
            # It's for RL to build model
            self.init_moe_weights()

        logger.info(
            f"{moe_tag}MoE config is {num_experts=}[{expert_id_offset}, {expert_id_offset + self.num_local_experts}), \
        {top_k=}, hidden_size={self.hidden_size}, {moe_intermediate_size=}, \
            , ep_size={self.ep_size}, \
            tp_size={self.tp_size}."
        )

    def init_moe_weights(self):
        """
        Initialize the weight shapes and parameters for the MoE layer.
        Combines weight shape initialization and parameter creation into a single function.
        """
        # Initialize weight shapes
        self._dtype = self._helper.get_default_dtype()
        self.weight_dtype = self._dtype
        gate_weight_shape = [self.hidden_size, self.num_experts]
        gate_correction_bias_shape = [1, self.num_experts]

        self.gate_weight = self.create_parameter(
            shape=gate_weight_shape,
            dtype="float32",
        )
        if self.fd_config.model_config.moe_use_aux_free:
            self.gate_correction_bias = self.create_parameter(
                shape=gate_correction_bias_shape,
                dtype="float32",
            )
        up_gate_proj_output_dim = self.moe_intermediate_size * 2
        if self.moe_quant_type in ["fp8", "wint8"]:
            up_gate_proj_weight_shape = [
                self.num_local_experts,
                up_gate_proj_output_dim,
                self.hidden_size,
            ]
            down_proj_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ]
        else:
            up_gate_proj_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                up_gate_proj_output_dim,
            ]
            down_proj_weight_shape = [
                self.num_local_experts,
                self.moe_intermediate_size,
                self.hidden_size,
            ]

        # Create parameters
        if self.moe_quant_type == "fp8":
            # (TODO:gaoziyuan)
            pass
        elif self.moe_quant_type == "wint8":
            self.weight_dtype = "int8"
            self.init_weight_only_scale()

        # up_gate_proj parameters
        self.up_gate_proj_weight = self.create_parameter(
            shape=up_gate_proj_weight_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        # down_proj parameters
        self.down_proj_weight = self.create_parameter(
            shape=down_proj_weight_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """
        self.up_gate_proj_weight_scale = self.create_parameter(
            shape=[self.num_local_experts, self.moe_intermediate_size * 2],
            dtype=self._dtype,
        )
        self.down_proj_weight_scale = self.create_parameter(
            shape=[self.num_local_experts, self.hidden_size],
            dtype=self._dtype,
        )

    def load_experts_weight(
        self,
        state_dict: dict,
        up_gate_proj_expert_weight_key: str,
        down_proj_expert_weight_key: str,
    ):
        """
        Load experts weight from state_dict.
        Args:
            state_dict (dict): The state_dict of model.
            up_gate_proj_expert_weight_key (str): The key of up_gate_proj expert weight.
            down_proj_expert_weight_key (str): The key of down_proj expert weight.
        """
        logical_expert_ids = [
            i
            for i in range(
                self.expert_id_offset,
                self.expert_id_offset + self.num_local_experts,
            )
        ]
        ep_rank_to_expert_id_list = [i for i in range(self.num_experts)]
        if self.redundant_table_manger is not None:
            (
                ep_rank_to_expert_id_list,
                expert_id_to_ep_rank_array,
                expert_in_rank_num_list,
                tokens_per_expert_stats_list,
            ) = self.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(self.layer_idx)
            logical_expert_ids = ep_rank_to_expert_id_list[
                self.expert_id_offset : self.expert_id_offset + self.num_local_experts
            ]
        up_gate_proj_weights = []
        down_proj_weights = []
        is_ffn_merged = up_gate_proj_expert_weight_key.format(self.expert_id_offset) in state_dict
        if is_ffn_merged:
            for expert_idx in logical_expert_ids:
                down_proj_expert_weight_key_name = down_proj_expert_weight_key.format(expert_idx)
                up_gate_proj_expert_weight_key_name = up_gate_proj_expert_weight_key.format(expert_idx)
                up_gate_proj_weights.append(
                    get_tensor(
                        (
                            state_dict.pop(up_gate_proj_expert_weight_key_name)
                            if up_gate_proj_expert_weight_key_name in state_dict
                            else up_gate_proj_expert_weight_key_name
                        ),
                        self.fd_config.model_config.model,
                    )
                )
                down_proj_weights.append(
                    get_tensor(
                        (
                            state_dict.pop(down_proj_expert_weight_key_name)
                            if down_proj_expert_weight_key_name in state_dict
                            else down_proj_expert_weight_key_name
                        ),
                        self.fd_config.model_config.model,
                    )
                )
        else:
            gate_expert_weight_key = up_gate_proj_expert_weight_key.replace("up_gate_proj", "gate_proj")
            up_expert_weight_key = up_gate_proj_expert_weight_key.replace("up_gate_proj", "up_proj")
            for expert_idx in logical_expert_ids:
                gate_expert_weight_key_name = gate_expert_weight_key.format(expert_idx)
                up_expert_weight_key_name = up_expert_weight_key.format(expert_idx)
                down_proj_expert_weight_key_name = down_proj_expert_weight_key.format(expert_idx)
                gate = get_tensor(
                    (
                        state_dict.pop(gate_expert_weight_key_name)
                        if gate_expert_weight_key_name in state_dict
                        else gate_expert_weight_key_name
                    ),
                    self.fd_config.model_config.model,
                )
                up = get_tensor(
                    (
                        state_dict.pop(up_expert_weight_key_name)
                        if up_expert_weight_key_name in state_dict
                        else up_expert_weight_key_name
                    ),
                    self.fd_config.model_config.model,
                )
                up_gate_proj_weights.append(paddle.concat([gate, up], axis=-1))
                down_proj_weights.append(
                    get_tensor(
                        (
                            state_dict.pop(down_proj_expert_weight_key_name)
                            if down_proj_expert_weight_key_name in state_dict
                            else down_proj_expert_weight_key_name
                        ),
                        self.fd_config.model_config.model,
                    )
                )
        return up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list

    def extract_moe_ffn_weights(self, state_dict: dict):
        """
        Extract MoE FFN weights from state dict based on weight key mapping.

        Args:
            state_dict (dict): Model state dictionary containing the weights.

        Returns:
            tuple: A tuple containing two lists:
                - up_gate_proj_weights: List of tensors for first FFN layer weights
                - down_proj_weights: List of tensors for second FFN layer weights

        Raises:
            AssertionError: If required weight keys are missing or number of weights
                doesn't match number of local experts.
        """
        up_gate_proj_expert_weight_key = self.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = self.weight_key_map.get("down_proj_expert_weight_key", None)
        assert up_gate_proj_expert_weight_key is not None, "up_gate_proj_expert_weight_key should not be none."
        assert down_proj_expert_weight_key is not None, "down_proj_expert_weight_key should not be none."

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, _ = self.load_experts_weight(
            state_dict,
            up_gate_proj_expert_weight_key,
            down_proj_expert_weight_key,
        )
        assert (
            len(up_gate_proj_weights) == self.num_local_experts
        ), "up_gate_proj_weights length should be equal to num_local_experts."
        assert (
            len(down_proj_weights) == self.num_local_experts
        ), "down_proj_weights length should be equal to num_local_experts."

        return up_gate_proj_weights, down_proj_weights

    def extract_gate_correction_bias(self, gate_correction_bias_key, state_dict):
        """
        extract_gate_correction_bias function.
        """
        gate_correction_bias_tensor = get_tensor(state_dict.pop(gate_correction_bias_key)).astype("float32")
        return gate_correction_bias_tensor

    def load_state_dict(self, state_dict, is_rearrange: bool = False):
        """
        load_state_dict function.
        """
        if not is_rearrange:
            self.gate_correction_bias_key = self.weight_key_map.get("gate_correction_bias_key", None)
            if self.gate_correction_bias_key is not None and self.gate_correction_bias_key in state_dict:
                self.moe_use_gate_correction_bias = True
            else:
                self.moe_use_gate_correction_bias = False
            if self.moe_use_gate_correction_bias:
                gate_correction_bias_tensor = self.extract_gate_correction_bias(
                    self.gate_correction_bias_key, state_dict
                )
                self.gate_correction_bias = self.create_parameter(
                    shape=gate_correction_bias_tensor.shape,
                    dtype="float32",
                )
                self.gate_correction_bias.set_value(gate_correction_bias_tensor)

            gate_weight_key = self.weight_key_map.get("gate_weight_key", None)
            assert gate_weight_key is not None, "gate_weight_key should not be None, please check model checkpoints"

            gate_weight_tensor = get_tensor(state_dict.pop(gate_weight_key))

            self.gate_weight = self.create_parameter(
                shape=gate_weight_tensor.shape,
                dtype="float32",
            )
            self.gate_weight.set_value(gate_weight_tensor.astype("float32"))

        if self.fd_config.model_config.is_quantized:
            if getattr(self.fd_config.quant_config, "is_permuted", False):
                self.quant_method.process_prequanted_weights(self, state_dict)
            else:
                self.quant_method.create_weights(self, state_dict)
        else:
            self.quant_method.create_weights(self, state_dict)

    def forward(self, x: paddle.Tensor):
        """
        Defines the forward computation of the moe layer.

        Args:
            x (Tensor): Input tensor to the moe layer.

        Returns:
            Tensor: Output tensor.s

        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)
        out = self.quant_method.apply(self, x, gate_out)
        return out
