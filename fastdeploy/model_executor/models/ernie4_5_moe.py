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

import inspect
import re
from functools import partial
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
from fastdeploy.model_executor.models.utils import WeightMeta
from fastdeploy.platforms import current_platform
from fastdeploy.worker.experts_manager import RedundantExpertManger


class Ernie4_5_MLP(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
            reduce_results=reduce_results,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=None,
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        gate_up_out = self.up_gate_proj(hidden_states)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Ernie4_5_MoE(nn.Layer):
    def __init__(
        self, fd_config: FDConfig, layer_id: int, prefix: str, redundant_table_manger: RedundantExpertManger = None
    ) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, "moe_quant_type"):
            moe_quant_type = fd_config.quant_config.moe_quant_type

        if moe_quant_type == "w4a8" or moe_quant_type == "w4afp8":
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4w2":
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_super_scales_key": f"{prefix}.experts.{{}}.up_gate_proj.super_scales",
                "down_proj_expert_super_scales_key": f"{prefix}.experts.{{}}.down_proj.super_scales",
                "up_gate_proj_expert_code_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.code_scale",
                "down_proj_expert_code_scale_key": f"{prefix}.experts.{{}}.down_proj.code_scale",
                "up_gate_proj_expert_code_zp_key": f"{prefix}.experts.{{}}.up_gate_proj.code_zp",
                "down_proj_expert_code_zp_key": f"{prefix}.experts.{{}}.down_proj.code_zp",
            }
        elif moe_quant_type == "tensor_wise_fp8" or (
            moe_quant_type == "block_wise_fp8"
            and (fd_config.model_config.is_quantized or fd_config.model_config.is_moe_quantized)
        ):
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        else:
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            }

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.moe_num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        self.experts = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.moe_num_experts,
            top_k=fd_config.model_config.moe_k,
            layer_idx=layer_id,
            gate_correction_bias=None,
            redundant_table_manger=redundant_table_manger,
            weight_key_map=weight_key_map,
        )

        if fd_config.model_config.moe_use_aux_free:
            self.experts.gate_correction_bias = self.create_parameter(
                shape=[1, fd_config.model_config.moe_num_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
        else:
            self.experts.gate_correction_bias = None

        self.num_shared_experts = fd_config.model_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            shared_experts_hidden_dim = self.num_shared_experts * fd_config.model_config.moe_intermediate_size
            self.shared_experts = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_hidden_dim,
                prefix=f"{prefix}.shared_experts",
            )

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)
        if self.experts.gate_correction_bias is not None:
            gate_correction_bias_tensor = state_dict.pop(self.experts.gate_correction_bias_key)
            if self.experts.gate_correction_bias.shape != gate_correction_bias_tensor.shape:
                gate_correction_bias_tensor = gate_correction_bias_tensor.reshape(
                    self.experts.gate_correction_bias.shape
                )
            self.experts.gate_correction_bias.set_value(gate_correction_bias_tensor)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def update_state_dict(self, state_dict):
        self.experts.load_state_dict(state_dict, True)

    def forward(self, hidden_states: paddle.Tensor):
        out = self.experts(hidden_states, self.gate)
        if self.num_shared_experts > 0:
            s_x = self.shared_experts(hidden_states)
            out = out + s_x
        return out


class Ernie4_5_Attention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
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
        redundant_table_manger: RedundantExpertManger = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        self.self_attn = Ernie4_5_Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (
            getattr(fd_config.model_config, "moe_num_experts", None) is not None
            and layer_id >= fd_config.model_config.moe_layer_start_index
        ):
            self.mlp = Ernie4_5_MoE(
                fd_config=fd_config,
                layer_id=layer_id,
                redundant_table_manger=redundant_table_manger,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def update_state_dict(self, state_dict):
        self.mlp.update_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )

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

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "ernie"
        self.fd_config = fd_config
        self.redundant_table_manger = None
        if fd_config.eplb_config.enable_eplb is True:
            self.redundant_table_manger = RedundantExpertManger(
                n_routed_experts=fd_config.model_config.moe_num_experts,
                num_hidden_layers=fd_config.model_config.num_hidden_layers,
                redundant_experts_num=fd_config.model_config.redundant_experts_num,
                ep_size=fd_config.parallel_config.expert_parallel_size,
            )

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Ernie4_5_DecoderLayer(
                    fd_config=fd_config,
                    redundant_table_manger=self.redundant_table_manger,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def update_state_dict(self, state_dict):
        """
        Update model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        for i in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            logger.info(f"Start update layer {i}")
            self.layers[i].update_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        if current_platform.is_iluvatar() and forward_meta.attn_backend.mixed:
            hidden_states = forward_meta.attn_backend.transpose(hidden_states)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if current_platform.is_iluvatar() and forward_meta.attn_backend.mixed:
            out = forward_meta.attn_backend.reverse_transpose(out)

        return out


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_MoeForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
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
        self.ernie = Ernie4_5_Model(fd_config=fd_config)

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
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.ernie.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.ernie.embed_tokens.embeddings.weight})
        else:
            self.lm_head.load_state_dict(state_dict)

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.

        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """

        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
            rename_offline_ckpt_suffix_to_fd_suffix,
        )

        general_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("embed_tokens.embeddings", "embed_tokens", None, None),
            ("lm_head.linear", "lm_head", None, None),
            ("experts.gate_correction_bias", "moe_statics.e_score_correction_bias", None, None),
            ("qkv_proj", "q_proj", None, "q"),
            ("qkv_proj", "k_proj", None, "k"),
            ("qkv_proj", "v_proj", None, "v"),
            ("up_gate_proj", "gate_proj", None, "gate"),
            ("up_gate_proj", "up_proj", None, "up"),
            ("attn.cache_k_scale", "cachek_matmul.activation_scale", None, None),
            ("attn.cache_v_scale", "cachev_matmul.activation_scale", None, None),
            ("attn.cache_k_zp", "cachek_matmul.activation_zero_point", None, None),
            ("attn.cache_v_zp", "cachev_matmul.activation_zero_point", None, None),
        ]

        expert_params_mapping = []
        if getattr(self.fd_config.model_config, "moe_num_experts", None) is not None:
            if self.fd_config.parallel_config.expert_parallel_size > 1:
                num_experts = self.fd_config.parallel_config.num_experts_per_rank
                num_experts_start_offset = self.fd_config.parallel_config.num_experts_start_offset
            else:
                num_experts = self.fd_config.model_config.moe_num_experts
                num_experts_start_offset = 0

            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                num_experts=num_experts,
                ckpt_down_proj_name="down_proj",
                ckpt_gate_up_proj_name="up_gate_proj",
                ckpt_gate_proj_name="gate_proj",
                ckpt_up_proj_name="up_proj",
                param_gate_up_proj_name="experts.up_gate_proj_",
                param_down_proj_name="experts.down_proj_",
                num_experts_start_offset=num_experts_start_offset,
            )
        all_param_mapping = [
            (param, weight, exp, shard, False) for param, weight, exp, shard in general_params_mapping
        ] + [(param, weight, exp, shard, True) for param, weight, exp, shard in expert_params_mapping]
        checkpoint_to_fd_key_fn = rename_offline_ckpt_suffix_to_fd_suffix(
            fd_config=self.fd_config, ckpt_weight_suffix="quant_weight", ckpt_scale_suffix="weight_scale"
        )
        params_dict = dict(self.named_parameters())

        process_weights_after_loading_fn = process_weights_after_loading(
            dict(self.named_sublayers()), fd_config=self.fd_config
        )

        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded_weight_name = loaded_weight_name.replace("model", "ernie")
            for param_name, weight_name, exp_id, shard_id, is_moe in all_param_mapping:
                loaded_weight_name = checkpoint_to_fd_key_fn(loaded_weight_name, is_moe)
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                expert_id = exp_id
                shard_id = shard_id
                break
            else:
                expert_id = None
                shard_id = None
                loaded_weight_name = checkpoint_to_fd_key_fn(loaded_weight_name, is_moe=False)
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict.keys():
                    continue
                param = params_dict[model_param_name]

            # Get weight loader from parameter and set weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            sig = inspect.signature(weight_loader)
            if "expert_id" in sig.parameters:
                weight_loader(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            else:
                weight_loader(param, loaded_weight, shard_id)

            model_sublayer_name = re.sub(
                r"\.(up_gate_proj_weight|down_proj_weight|weight|cache_k_scale|cache_v_scale)$", "", model_param_name
            )
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(self.ernie.embed_tokens.embeddings.weight.transpose([1, 0]))

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.ernie.layers[i].mlp.experts(fake_hidden_states, self.ernie.layers[i].mlp.gate)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.ernie(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.ernie.clear_grpah_opt_backend(fd_config=self.fd_config)


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_ForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
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


@ModelRegistry.register_model_class(
    architecture="Ernie4_5ForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Ernie4_5ForCausalLM(Ernie4_5_ForCausalLM):
    """
    Ernie4_5ForCausalLM 0.3B-PT
    """

    @classmethod
    def name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5ForCausalLM"


class Ernie4_5_MoePretrainedModel(PretrainedModel):
    """
    Ernie4_5_MoePretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "Ernie4_5_MoeForCausalLM"

    weight_infos = [
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.weight",
            False,
        ),
        WeightMeta(".embed_tokens.weight", False),
        WeightMeta("lm_head.weight", True),
        # quant tensorwise
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.quant_weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.quant_weight",
            False,
        ),
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("ernie inference model _get_tensor_parallel_mappings")
        from fastdeploy.model_executor.models.tp_utils import (
            build_expanded_keys,
            has_prefix,
            split_or_merge_func_v1,
        )

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        def get_tensor_parallel_split_mappings(num_layers, moe_num_experts, moe_layer_start_index, prefix_name):
            base_actions = {}
            weight_infos = cls.weight_infos
            for weight_name, is_column, extra in weight_infos:
                params = {
                    "is_column": is_column,
                    **({extra.value: True} if extra else {}),
                }

                if "lm_head.weight" in weight_name:
                    key = weight_name
                elif not has_prefix(prefix_name, weight_name):
                    key = f"{prefix_name}{weight_name}"
                else:
                    key = weight_name
                base_actions[key] = partial(fn, **params)
            final_actions = {}
            start_layer = moe_layer_start_index if moe_layer_start_index > 0 else num_layers
            final_actions = build_expanded_keys(base_actions, num_layers, start_layer, moe_num_experts)
            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers,
            getattr(config, "moe_num_experts", 0),
            getattr(config, "moe_layer_start_index", -1),
            config.prefix_name,
        )
        return mappings


class Ernie4_5_PretrainedModel(Ernie4_5_MoePretrainedModel):
    """
    Ernie4_5_PretrainedModel
    """

    @classmethod
    def arch_name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5_ForCausalLM"


class Ernie4_5PretrainedModel(Ernie4_5_PretrainedModel):
    """
    Ernie4_5PretrainedModel 0.3B-PT
    """

    @classmethod
    def arch_name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5ForCausalLM"
