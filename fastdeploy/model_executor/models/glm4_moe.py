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

import re
from functools import partial

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
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


class Glm4MoeMLP(nn.Layer):
    """ """

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

    def forward(self, x):
        """ """
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Glm4Moe(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.expert_parallel_size = fd_config.parallel_config.expert_parallel_size
        self.tensor_parallel_size = fd_config.parallel_config.tensor_parallel_size
        self.tensor_parallel_rank = fd_config.parallel_config.tensor_parallel_rank
        self.tp_group = fd_config.parallel_config.tp_group

        self.use_ep = self.expert_parallel_size > 1
        self.use_tp = self.tensor_parallel_size > 1

        self.n_routed_experts: int = fd_config.model_config.n_routed_experts
        self.n_shared_experts: int = fd_config.model_config.n_shared_experts

        self.norm_topk_prob = fd_config.model_config.norm_topk_prob

        weight_key_map = {
            "gate_correction_bias_key": f"{prefix}.gate.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.n_routed_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )
        self.gate.e_score_correction_bias = self.create_parameter(
            shape=[1, fd_config.model_config.n_routed_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.experts = FusedMoE(
            fd_config,
            reduce_results=False,
            renormalize=self.norm_topk_prob,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.n_routed_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method="noaux_tc",
            topk_group=fd_config.model_config.topk_group,
            n_group=fd_config.model_config.n_group,
            routed_scaling_factor=fd_config.model_config.routed_scaling_factor,
            layer_idx=layer_id,
            gate_correction_bias=self.gate.e_score_correction_bias,
            weight_key_map=weight_key_map,
        )

        shared_experts_intermediate_size = self.n_shared_experts * fd_config.model_config.moe_intermediate_size

        self.shared_experts = Glm4MoeMLP(
            fd_config=fd_config,
            intermediate_size=shared_experts_intermediate_size,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )

    def forward(self, x):
        shared_experts_out = self.shared_experts(x)
        out = self.experts(x, self.gate)
        out = out + shared_experts_out
        # We do to TP all reduce after the sum of experts.
        if self.tensor_parallel_size > 1:
            out = tensor_model_parallel_all_reduce(out, self.tp_group)
        return out


class Glm4MoeAttention(nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        tp_size = fd_config.parallel_config.tensor_parallel_size
        self.fd_config = fd_config
        self.head_dim = fd_config.model_config.head_dim
        self.num_heads = fd_config.model_config.num_attention_heads // tp_size
        self.num_kv_heads = fd_config.model_config.num_key_value_heads // tp_size
        self.attention_bias = fd_config.model_config.attention_bias
        self.use_qk_norm = fd_config.model_config.use_qk_norm
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=self.attention_bias)

        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )

        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
            rms_norm_eps=fd_config.model_config.rms_norm_eps,
        )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                fd_config,
                hidden_size=self.head_dim,
                eps=fd_config.model_config.rms_norm_eps,
                prefix=f"{prefix}.q_norm",
                begin_norm_axis=2,
            )
            self.k_norm = RMSNorm(
                fd_config,
                hidden_size=self.head_dim,
                eps=fd_config.model_config.rms_norm_eps,
                prefix=f"{prefix}.k_norm",
                begin_norm_axis=2,
            )

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """ """
        qkv_out = self.qkv_proj(hidden_states)

        if self.use_qk_norm:
            q, k, v = qkv_out.split([self.q_size, self.kv_size, self.kv_size], axis=-1)
            q = self.q_norm(q.reshape([-1, self.num_heads, self.head_dim]))[0].reshape(q.shape)
            k = self.k_norm(k.reshape([-1, self.num_kv_heads, self.head_dim]))[0].reshape(k.shape)
            qkv_out = paddle.concat([q, k, v], axis=-1)

        atten_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(atten_out)
        return output


class Glm4MoeDecoderLayer(nn.Layer):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = Glm4MoeAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (
            fd_config.model_config.n_routed_experts is not None
            and layer_id >= fd_config.model_config.first_k_dense_replace
        ):
            self.mlp = Glm4Moe(fd_config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Glm4MoeMLP(
                fd_config,
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

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        """ """
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_graph_optimization
class Glm4MoeModel(nn.Layer):
    """ """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Qwen2Model class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Glm4MoeDecoderLayer(
                    fd_config,
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

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="Glm4MoeForCausalLM",
    module_name="glm4_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Glm4MoeForCausalLM(ModelForCasualLM):
    """
    Glm4MoeForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Glm4MoeForCausalLM, self).__init__(fd_config)

        self.model = Glm4MoeModel(fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(self):
        """ """
        return "Glm4MoeForCausalLM"

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
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
        ]
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        for loaded_weight_name, loaded_weight in weights_iterator:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts" in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        glm4_moe only support loader_v1.
        """
        assert False, "glm4_moe only support --load-choices default_v1."

    def compute_logits(self, hidden_states: paddle.Tensor):
        """ """
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.ones(
            shape=[1, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.first_k_dense_replace,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.model.layers[i].mlp.experts(fake_hidden_states, self.model.layers[i].mlp.gate)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        hidden_states = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class Glm4MoePretrainedModel(PretrainedModel):
    """
    Glm4MoePretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "Glm4MoeForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        logger.info("Glm4Moe inference model _get_tensor_parallel_mappings")

        from fastdeploy.model_executor.models.tp_utils import split_or_merge_func_v1

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            # Self Attention Layer which are need TP.
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            # MLP Layer
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

            # Moe Layer
            for expert_idx in range(config.n_routed_experts):
                base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)

            # Shared Expert Layer
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

            # MTP parts
            base_actions["layers.46.embed_tokens.weight"] = partial(fn, is_column=False)
            base_actions["layers.46.eh_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.46.shared_head.head.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings
