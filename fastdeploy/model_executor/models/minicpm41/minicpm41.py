# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import math
import re
from functools import partial
from typing import Dict

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
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
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.minicpm41.config_minicpm41 import (
    SUPPORTED_QUANTIZATIONS,
)
from fastdeploy.model_executor.models.minicpm41.hybrid_reasoning import (
    HybridReasoningMode,
    build_minicpm41_thinking_token_sequences,
)
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.utils import (
    WeightsMapper,
    default_weight_loader,
    process_weights_after_loading,
    process_weights_before_loading,
)


def minicpm41_config_value(model_config, name: str, default):
    value = getattr(model_config, name, None)
    if value is not None:
        return value

    pretrained_config = getattr(model_config, "pretrained_config", None)
    value = getattr(pretrained_config, name, None)
    if value is not None:
        return value

    return default


def minicpm41_embedding_scale(model_config) -> float:
    return float(minicpm41_config_value(model_config, "scale_emb", 1.0))


def minicpm41_residual_scale(model_config) -> float:
    num_hidden_layers = int(minicpm41_config_value(model_config, "num_hidden_layers", 1))
    scale_depth = float(minicpm41_config_value(model_config, "scale_depth", 1.0))
    return scale_depth / math.sqrt(num_hidden_layers)


def minicpm41_lm_head_scale(model_config) -> float:
    hidden_size = float(minicpm41_config_value(model_config, "hidden_size", 1.0))
    dim_model_base = float(minicpm41_config_value(model_config, "dim_model_base", hidden_size))
    return dim_model_base / hidden_size


class MiniCPM41MLP(nn.Layer):
    """MiniCPM4.1 feed-forward network."""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=fd_config.model_config.intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=getattr(self.up_gate_proj, "bias", None),
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x: paddle.Tensor, forward_meta: ForwardMeta):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        return self.down_proj(act_out)


class MiniCPM41Attention(nn.Layer):
    """MiniCPM4.1 grouped-query self attention."""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        self.qkv_proj = QKVParallelLinear(fd_config=fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
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
        attention_out = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        return self.o_proj(attention_out)


class MiniCPM41DecoderLayer(nn.Layer):
    """MiniCPM4.1 decoder block with muP residual scaling."""

    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])
        self.residual_scale = minicpm41_residual_scale(fd_config.model_config)

        self.self_attn = MiniCPM41Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = MiniCPM41MLP(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
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

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )
        hidden_states = self.self_attn(forward_meta=forward_meta, hidden_states=hidden_states)
        hidden_states = hidden_states * self.residual_scale

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_meta)
        hidden_states = hidden_states * self.residual_scale

        return hidden_states, residual


@support_graph_optimization
class MiniCPM41Model(nn.Layer):
    """MiniCPM4.1 decoder-only transformer."""

    def __init__(self, fd_config: FDConfig = None):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        self.embedding_scale = minicpm41_embedding_scale(fd_config.model_config)
        fd_config.model_config.pretrained_config.prefix_name = "minicpm41"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                MiniCPM41DecoderLayer(
                    fd_config=fd_config,
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
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        hidden_states = hidden_states * self.embedding_scale

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        return self.norm(hidden_states, residual)[0]


@ModelRegistry.register_model_class(
    architecture="MiniCPMForCausalLM",
    module_name="minicpm41.minicpm41",
    category=ModelCategory.TEXT_GENERATION | ModelCategory.REASONING,
)
class MiniCPM41ForCausalLM(ModelForCasualLM):
    """MiniCPM4.1-8B FastDeploy entry point."""

    supported_quantizations = SUPPORTED_QUANTIZATIONS

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        self.fd_config = fd_config
        self.hybrid_reasoning = HybridReasoningMode(fd_config)
        self.minicpm41 = MiniCPM41Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        self.lm_head_scale = minicpm41_lm_head_scale(fd_config.model_config)
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        self.process_weights_before_loading_fn = process_weights_before_loading(
            mapper=(
                WeightsMapper(orig_to_new_prefix={"model.": "minicpm41."})
                if self.fd_config.model_config.model_format == "torch"
                else None
            ),
        )

    @classmethod
    def name(self):
        return "MiniCPMForCausalLM"

    @staticmethod
    def build_thinking_token_sequences(tokenizer):
        """Expose MiniCPM4.1 tokenizer markers through the engine model hook."""
        return build_minicpm41_thinking_token_sequences(tokenizer)

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            loaded_weight_name = (
                self.process_weights_before_loading_fn(loaded_weight_name)
                if getattr(self, "process_weights_before_loading_fn", None)
                else loaded_weight_name
            )
            if loaded_weight_name is None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight|weight_scale)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if getattr(self, "tie_word_embeddings", False):
            self.lm_head.linear.weight.set_value(
                self.minicpm41.embed_tokens.embeddings.weight.transpose([1, 0]).astype(
                    self.lm_head.linear.weight.dtype
                )
            )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.minicpm41.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        hidden_states = hidden_states * self.lm_head_scale
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def get_logits_processors(self):
        """Return model-owned logits processors for the runner sampling path."""
        return [self.hybrid_reasoning]

    def forward(
        self,
        inputs: Dict,
        forward_meta: ForwardMeta,
    ):
        ids_remove_padding = inputs["ids_remove_padding"]
        return self.minicpm41(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def clear_graph_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.minicpm41.clear_graph_opt_backend(fd_config=self.fd_config)


class MiniCPM41PretrainedModel(PretrainedModel):
    """MiniCPM4.1 tensor-parallel conversion metadata."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(self):
        return "MiniCPMForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):
        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
            }

            if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        return get_tensor_parallel_split_mappings(config.num_hidden_layers)
