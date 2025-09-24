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

from typing import Optional

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import slice_fn
from fastdeploy.platforms import current_platform
from fastdeploy.worker.experts_manager import RedundantExpertManger

try:
    from fastdeploy.model_executor.ops.gpu import noaux_tc
except:
    logger.warning("import noaux_tc Failed!")


def get_moe_method():
    """
    return moe method based on device platform
    """

    if current_platform.is_cuda():
        from .fused_moe_cutlass_backend import CutlassMoEMethod

        return CutlassMoEMethod(None)
    elif current_platform.is_xpu():
        from fastdeploy.model_executor.layers.backends import XPUMoEMethod

        return XPUMoEMethod(None)
    elif current_platform.is_gcu():
        from fastdeploy.model_executor.layers.backends import GCUFusedMoeMethod

        return GCUFusedMoeMethod(None)
    elif current_platform.is_maca():
        from fastdeploy.model_executor.layers.backends import (
            MetaxTritonWeightOnlyMoEMethod,
        )

        return MetaxTritonWeightOnlyMoEMethod(None)
    elif current_platform.is_intel_hpu():
        from fastdeploy.model_executor.layers.backends import HpuMoEMethod

        return HpuMoEMethod(None)
        # return HpuTensorWiseFP8MoEMethod(None)
    raise NotImplementedError


def get_moe_scores(
    gating_output: paddle.Tensor,
    n_group,
    topk_group,
    top_k,
    routed_scaling_factor,
    e_score_correction_bias,
    renormalize: bool = False,
) -> paddle.Tensor:
    """
    compute moe scores using e_score_correction_bias.
    """
    scores = paddle.nn.functional.sigmoid(gating_output)
    assert e_score_correction_bias is not None, "e_score_correction_bias is none!"
    scores_with_bias = scores + e_score_correction_bias
    scores, topk_values, topk_idx = noaux_tc(
        scores,
        scores_with_bias,
        n_group if n_group > 0 else 1,
        topk_group if topk_group > 0 else 1,
        top_k,
        renormalize,
        routed_scaling_factor,
    )
    return scores, topk_values, topk_idx


class FusedMoE(nn.Layer):
    """
    FusedMoE is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        fd_config,
        reduce_results: bool = True,
        renormalize: bool = False,
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
        gate_correction_bias=None,
        redundant_table_manger: RedundantExpertManger = None,
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
        self.renormalize = renormalize
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.ep_rank = fd_config.parallel_config.expert_parallel_rank
        self.tp_group = fd_config.parallel_config.tp_group
        # NOTE(Zhenyu Li): just supports tp_size = 1 when ep_size > 1 in MOE now.
        if self.ep_size > 1:
            self.tp_size = 1
            self.tp_rank = 0

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
        self.moe_tag = moe_tag
        if self.ep_size > 1:
            expert_id_offset = expert_id_offset + self.ep_rank * self.num_local_experts

        self.expert_id_offset = expert_id_offset

        self.gate_correction_bias_key = self.weight_key_map.get("gate_correction_bias_key", None)
        if self.gate_correction_bias_key is not None:
            self.moe_use_gate_correction_bias = True
        else:
            self.moe_use_gate_correction_bias = False

        # used for deepseek_v3
        self.topk_method = topk_method
        self.topk_group = topk_group
        self.n_group = n_group
        self.routed_scaling_factor = routed_scaling_factor

        self._dtype = self._helper.get_default_dtype()
        self.weight_dtype = self._dtype

        moe_quant_config = fd_config.quant_config
        self.moe_quant_config = moe_quant_config
        self.moe_quant_type = None
        if moe_quant_config:
            self.quant_method = moe_quant_config.get_quant_method(self)
            self.moe_quant_type = moe_quant_config.name()
        else:
            self.quant_method = get_moe_method()
        self.redundant_table_manger = redundant_table_manger
        if self.ep_size > 1:
            self.quant_method.init_ep(self)

        # Merge normal and RL build model
        if gate_correction_bias is not None:
            self.gate_correction_bias = gate_correction_bias
        else:
            self.gate_correction_bias = None
        self.quant_method.create_weights(
            self, weight_loader=self.weight_loader, model_format=fd_config.model_config.model_format
        )

        logger.info(
            f"{moe_tag}MoE config is {num_experts=}[{expert_id_offset}, {expert_id_offset + self.num_local_experts}), \
        {top_k=}, hidden_size={self.hidden_size}, {moe_intermediate_size=}, \
            , ep_size={self.ep_size}, \
            tp_size={self.tp_size}."
        )

    def weight_loader(self, param, loaded_weight, expert_id, shard_id: Optional[str] = None):

        if hasattr(param, "SHARD_ID_TO_SHARDED_DIM"):
            SHARD_ID_TO_SHARDED_DIM = param.SHARD_ID_TO_SHARDED_DIM
        elif current_platform.is_cuda():
            SHARD_ID_TO_SHARDED_DIM = {"gate": 1, "down": 0, "up": 1}
        else:
            SHARD_ID_TO_SHARDED_DIM = {"gate": 0, "down": 1, "up": 0}

        if not param._is_initialized():
            param.initialize()

        if shard_id is None:
            # 1.gate up fused in disk
            weight_need_transpose = getattr(param, "weight_need_transpose", False)
            output_size = param[expert_id - self.expert_id_offset].shape[SHARD_ID_TO_SHARDED_DIM["gate"]]
            per_rank = output_size // 2
            start = self.tp_rank * per_rank
            loaded_weight_shard_gate = slice_fn(
                loaded_weight, weight_need_transpose ^ SHARD_ID_TO_SHARDED_DIM["gate"], start, start + per_rank
            )
            self._load_gate_up_weight(
                param, expert_id, loaded_weight_shard_gate, "gate", SHARD_ID_TO_SHARDED_DIM["gate"], is_sharded=True
            )
            start_up = output_size // 2 * self.tp_size + self.tp_rank * per_rank
            loaded_weight_shard_up = slice_fn(
                loaded_weight, weight_need_transpose ^ SHARD_ID_TO_SHARDED_DIM["up"], start_up, start_up + per_rank
            )
            self._load_gate_up_weight(
                param, expert_id, loaded_weight_shard_up, "up", SHARD_ID_TO_SHARDED_DIM["up"], is_sharded=True
            )
        else:
            # 2.gate up splited in disk
            assert shard_id in ["gate", "down", "up"]
            self._load_expert_weight(
                param=param,
                expert_id=expert_id,
                loaded_weight=loaded_weight,
                shard_id=shard_id,
                shard_dim=SHARD_ID_TO_SHARDED_DIM[shard_id],
            )

    def _load_gate_up_weight(self, param, expert_id, loaded_weight, shard_id, shard_dim=None, is_sharded=False):
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if self.tp_size > 1 and not is_sharded:
            tp_shard_dim = weight_need_transpose ^ shard_dim
            weight_dim = -1 if tp_shard_dim else 0
            if isinstance(loaded_weight, (np.ndarray, paddle.Tensor)):
                size = loaded_weight.shape[weight_dim]
            else:
                size = loaded_weight.get_shape()[weight_dim]
            block_size = size // self.tp_size
            shard_offset = self.tp_rank * block_size
            shard_size = (self.tp_rank + 1) * block_size
            loaded_weight = slice_fn(loaded_weight, tp_shard_dim, shard_offset, shard_size)
        loaded_weight = get_tensor(loaded_weight)
        expert_param = param[expert_id - self.expert_id_offset]
        dim = -1 if shard_dim else 0
        param_shard_size = expert_param.shape[dim] // 2
        if shard_id == "gate":
            param_shard_offset = 0
        else:
            # shard_id == "up":
            param_shard_offset = param_shard_size
        expert_param = slice_fn(
            expert_param, shard_dim, start=param_shard_offset, end=param_shard_offset + param_shard_size
        )
        if hasattr(param, "tensor_track"):
            # for dyn quant
            param.tensor_track.mark(
                start=param_shard_offset,
                end=param_shard_offset + param_shard_size,
                batch_id=expert_id - self.expert_id_offset,
            )

        # To ensure compatibility across backends, apply an extra transpose for GCU and XPU
        if expert_param.shape != loaded_weight.shape:
            loaded_weight = loaded_weight.transpose([1, 0])
        assert expert_param.shape == loaded_weight.shape, (
            f"Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({expert_param.shape})"
        )
        if expert_param.dtype != loaded_weight.dtype:
            if loaded_weight.dtype == paddle.int8 and expert_param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.view(expert_param.dtype)
            else:
                loaded_weight = loaded_weight.cast(expert_param.dtype)
        expert_param.copy_(loaded_weight, False)

    def _load_down_weight(self, param, expert_id, loaded_weight, shard_id, shard_dim=None):
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if self.tp_size > 1 and shard_dim is not None:
            tp_shard_dim = weight_need_transpose ^ shard_dim
            dim = -1 if tp_shard_dim else 0
            if isinstance(loaded_weight, paddle.Tensor):
                size = loaded_weight.shape[dim]
            else:
                size = loaded_weight.get_shape()[dim]
            block_size = size // self.tp_size
            shard_offset = self.tp_rank * block_size
            shard_size = (self.tp_rank + 1) * block_size
            loaded_weight = slice_fn(loaded_weight, tp_shard_dim, shard_offset, shard_size)
        loaded_weight = get_tensor(loaded_weight)
        expert_param = param[expert_id - self.expert_id_offset]
        if hasattr(param, "tensor_track"):
            # for dyn quant
            param.tensor_track.mark(start=0, batch_id=expert_id - self.expert_id_offset)
        # To ensure compatibility across backends, apply an extra transpose for GCU and XPU and opensource weight
        if expert_param.shape != loaded_weight.shape:
            loaded_weight = loaded_weight.transpose([1, 0])
        assert expert_param.shape == loaded_weight.shape, (
            f"Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({expert_param.shape})"
        )
        if expert_param.dtype != loaded_weight.dtype:
            if loaded_weight.dtype == paddle.int8 and expert_param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.view(expert_param.dtype)
            else:
                loaded_weight = loaded_weight.cast(expert_param.dtype)
        expert_param.copy_(loaded_weight, False)

    def _load_expert_weight(
        self,
        param,
        expert_id,
        loaded_weight,
        shard_id,
        shard_dim=None,
    ):
        if shard_id == "down":
            self._load_down_weight(param, expert_id, loaded_weight, shard_id, shard_dim)
        elif shard_id in ["gate", "up"]:
            self._load_gate_up_weight(param, expert_id, loaded_weight, shard_id, shard_dim)

    @classmethod
    def make_expert_params_mapping(
        cls,
        num_experts: int,
        ckpt_gate_proj_name: Optional[str] = None,
        ckpt_up_proj_name: Optional[str] = None,
        ckpt_down_proj_name: Optional[str] = None,
        ckpt_gate_up_proj_name: Optional[str] = None,
        param_gate_up_proj_name: Optional[str] = None,
        param_down_proj_name: Optional[str] = None,
        ckpt_expert_key_name: str = "experts",
        experts_offset: int = 0,
        num_experts_start_offset: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        param_name_maping = []

        if ckpt_gate_up_proj_name:
            param_name_maping.append((None, ckpt_gate_up_proj_name))
        if ckpt_gate_proj_name:
            param_name_maping.append(("gate", ckpt_gate_proj_name))
        if ckpt_down_proj_name:
            param_name_maping.append(("down", ckpt_down_proj_name))
        if ckpt_up_proj_name:
            param_name_maping.append(("up", ckpt_up_proj_name))

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    param_gate_up_proj_name
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name, ckpt_gate_up_proj_name]
                    else param_down_proj_name
                ),
                f"{ckpt_expert_key_name}.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(
                experts_offset + num_experts_start_offset, experts_offset + num_experts_start_offset + num_experts
            )
            for shard_id, weight_name in param_name_maping
        ]

    def load_experts_weight(
        self,
        state_dict: dict,
        up_gate_proj_expert_weight_key: str,
        down_proj_expert_weight_key: str,
        is_rearrange: bool = False,
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
        if isinstance(state_dict, list):
            state_dict = dict(state_dict)
        is_ffn_merged = (
            up_gate_proj_expert_weight_key.format(logical_expert_ids[0] if is_rearrange else self.expert_id_offset)
            in state_dict
        )
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

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            self.load_experts_weight(
                state_dict,
                up_gate_proj_expert_weight_key,
                down_proj_expert_weight_key,
            )
        )
        assert (
            len(up_gate_proj_weights) == self.num_local_experts
        ), "up_gate_proj_weights length should be equal to num_local_experts."
        assert (
            len(down_proj_weights) == self.num_local_experts
        ), "down_proj_weights length should be equal to num_local_experts."

        return up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list

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
        if self.fd_config.model_config.is_quantized:
            if getattr(self.fd_config.quant_config, "is_permuted", True):
                self.quant_method.process_prequanted_weights(self, state_dict, is_rearrange)
            else:
                self.quant_method.process_loaded_weights(self, state_dict)
        else:
            self.quant_method.process_loaded_weights(self, state_dict)

    def forward(self, x: paddle.Tensor, gate: nn.Layer):
        """
        Defines the forward computation of the moe layer.

        Args:
            x (Tensor): Input tensor to the moe layer.

        Returns:
            Tensor: Output tensor.s

        """
        out = self.quant_method.apply(self, x, gate)
        return out
