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

from __future__ import annotations

from functools import partial
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_optimization
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import \
    LayerIdPlaceholder as layerid
from fastdeploy.model_executor.models.utils import WeightMeta
from fastdeploy.worker.forward_meta import ForwardMeta


class Ernie4_5_MLP(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
            use_fast_ffn=True,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=None,
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        gate_up_out = self.gate_up_proj(hidden_states)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Ernie4_5_MoE(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, 'moe_quant_type'):
            moe_quant_type = fd_config.quant_config.moe_quant_type

        if moe_quant_type == "w4a8":
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_in_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "ffn2_expert_in_scale_key":
                f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4w2":
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_super_scales_key":
                f"{prefix}.experts.{{}}.up_gate_proj.super_scales",
                "ffn2_expert_super_scales_key":
                f"{prefix}.experts.{{}}.down_proj.super_scales",
                "ffn1_expert_code_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.code_scale",
                "ffn2_expert_code_scale_key":
                f"{prefix}.experts.{{}}.down_proj.code_scale",
                "ffn1_expert_code_zp_key":
                f"{prefix}.experts.{{}}.up_gate_proj.code_zp",
                "ffn2_expert_code_zp_key":
                f"{prefix}.experts.{{}}.down_proj.code_zp",
            }
        elif moe_quant_type == "tensor_wise_fp8" or (
                moe_quant_type == "block_wise_fp8"
                and fd_config.model_config.is_quantized):
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "ffn1_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "ffn2_expert_weight_scale_key":
                f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "ffn1_expert_in_scale_key":
                f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "ffn2_expert_in_scale_key":
                f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        else:
            weight_key_map = {
                "gate_weight_key":
                f"{prefix}.gate.weight",
                "gate_correction_bias_key":
                f"{prefix}.moe_statics.e_score_correction_bias",
                "ffn1_expert_weight_key":
                f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "ffn2_expert_weight_key":
                f"{prefix}.experts.{{}}.down_proj.weight",
            }

        self.fused_moe = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.moe_config.moe_intermediate_size,
            num_experts=fd_config.moe_config.num_experts,
            top_k=fd_config.moe_config.top_k,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

        self.num_shared_experts = fd_config.moe_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            shared_experts_hidden_dim = self.num_shared_experts * fd_config.moe_config.moe_intermediate_size
            self.shared_experts = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_hidden_dim,
                prefix=f"{prefix}.shared_experts",
            )

    def load_state_dict(self, state_dict):
        self.fused_moe.load_state_dict(state_dict)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        out = self.fused_moe(hidden_states)
        if self.num_shared_experts > 0:
            s_x = self.shared_experts(hidden_states)
            out = out + s_x
        return out


class Ernie4_5_Attention(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim *
            fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
        )
        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        qkv_out = self.qkv_proj(hidden_states)

        attn_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )

        output = self.o_proj(attn_out)

        return output


class Ernie4_5_DecoderLayer(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Ernie4_5_Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (fd_config.moe_config.num_experts is not None
                and layer_id >= fd_config.moe_config.moe_layer_start_index):
            self.mlp = Ernie4_5_MoE(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.ffn_hidden_size,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_graph_optimization
class Ernie4_5_Model(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie4_5_Model class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "ernie"

        self.embeddings = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"))

        self.hidden_layers = nn.LayerList([
            Ernie4_5_DecoderLayer(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ])

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{fd_config.model_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.embeddings.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.hidden_layers[i](forward_meta,
                                                            hidden_states,
                                                            residual)

        hidden_states = hidden_states + residual

        out = self.norm(hidden_states)

        return out


class Ernie4_5_MoeForCausalLM(ModelForCasualLM):
    """
    Ernie4_5_MoeForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_MoeForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.model = Ernie4_5_Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

    @classmethod
    def name(self):
        return "Ernie4_5_MoeForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray,
                                                         paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.model.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.out_linear.weight.set_value(
                self.model.embeddings.word_embeddings.weight.transpose([1, 0]))
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(self.fd_config.moe_config.moe_layer_start_index,
                       self.fd_config.model_config.num_layers):
            self.model.hidden_layers[i].mlp.fused_moe(fake_hidden_states)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.model(ids_remove_padding=ids_remove_padding,
                                   forward_meta=forward_meta)

        return hidden_states


class Ernie4_5_ForCausalLM(Ernie4_5_MoeForCausalLM):
    """
    Ernie4_5_ForCausalLM
    """

    @classmethod
    def name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5_ForCausalLM"


class Ernie4_5_PretrainedModel(PretrainedModel):
    """
    Ernie4_5_PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    weight_infos = [
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.weight",
                   True, tsm.GQA),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.weight",
                   False),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.weight",
            True, tsm.PairFused),
        WeightMeta(f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.weight",
                   False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.weight",
            True, tsm.PairFused),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.weight",
            False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.weight",
            True, tsm.PairFused),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.weight",
            False),
        WeightMeta(".embed_tokens.weight", False),
        WeightMeta("lm_head.weight", True),
        # quant tensorwise
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.quant_weight",
            True, tsm.GQA),
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.quant_weight",
            False),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.quant_weight",
            True, tsm.PairFused),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.quant_weight",
            False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.quant_weight",
            True, tsm.PairFused),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.quant_weight",
            False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.quant_weight",
            True, tsm.PairFused),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.quant_weight",
            False),
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("erine inference model _get_tensor_parallel_mappings")
        from fastdeploy.model_executor.models.tp_utils import (
            build_expanded_keys, has_prefix, split_or_merge_func_v1)

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim)

        def get_tensor_parallel_split_mappings(num_layers, moe_num_experts,
                                               moe_layer_start_index,
                                               prefix_name):
            base_actions = {}
            weight_infos = cls.weight_infos
            for (weight_name, is_column, extra) in weight_infos:
                params = {
                    "is_column": is_column,
                    **({
                        extra.value: True
                    } if extra else {})
                }

                if "lm_head.weight" in weight_name:
                    key = weight_name
                elif not has_prefix(prefix_name, weight_name):
                    key = f"{prefix_name}{weight_name}"
                else:
                    key = weight_name
                base_actions[key] = partial(fn, **params)
            final_actions = {}
            start_layer = (moe_layer_start_index
                           if moe_layer_start_index > 0 else num_layers)
            final_actions = build_expanded_keys(
                num_layers,
                moe_num_experts,
                start_layer,
                base_actions,
            )
            return final_actions

        moe_num_experts = 0
        if isinstance(config.moe_num_experts, list):
            moe_num_experts = sum(config.moe_num_experts)
        elif isinstance(config.moe_num_experts, int):
            moe_num_experts = config.moe_num_experts

        moe_layer_start_index = -1
        if isinstance(config.moe_layer_start_index, list):
            moe_layer_start_index = min(config.moe_layer_start_index)
        elif isinstance(config.moe_layer_start_index, int):
            moe_layer_start_index = config.moe_layer_start_index

        mappings = get_tensor_parallel_split_mappings(config.num_layers,
                                                      moe_num_experts,
                                                      moe_layer_start_index,
                                                      config.prefix_name)
        return mappings
