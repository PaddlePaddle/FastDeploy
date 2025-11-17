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
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
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


class GptOssAttention(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.head_dim = fd_config.model_config.head_dim
        self.num_key_value_heads = fd_config.model_config.num_key_value_heads
        self.prefix = prefix

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            with_bias=True,
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            with_bias=True,
            add_bias=True,
        )

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            use_neox_rotary_style=True,
            with_sinks=True,
        )

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        qkv_out = self.qkv_proj(hidden_states)
        attn_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(attn_out)
        return output


class GptOssMoe(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        hidden_size = fd_config.model_config.hidden_size
        num_local_experts = fd_config.model_config.num_local_experts

        self.router = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.router",
            input_size=hidden_size,
            output_size=num_local_experts,
            with_bias=True,
            skip_quant=True,
            weight_dtype="float32",
        )

        weight_key_map = {
            "gate_weight_key": f"{prefix}.router.weight",
            "gate_correction_bias_key": f"{prefix}.router.bias",
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.gate_up_proj",
            "up_gate_proj_expert_bias_key": f"{prefix}.experts.gate_up_proj_bias",
            "down_proj_expert_weight_key": f"{prefix}.experts.down_proj",
            "down_proj_expert_bias_key": f"{prefix}.experts.down_proj_bias",
        }

        self.experts = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.model_config.intermediate_size,
            num_experts=num_local_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
            with_bias=True,
            activation="swigluoai",
            model_format="",
        )

    def forward(self, hidden_states: paddle.Tensor):
        expert_output = self.experts(hidden_states, self.router)
        return expert_output


class GptOssDecoderLayer(nn.Layer):
    """
    Paddle equivalent of vLLM's TransformerBlock.
    """

    def __init__(self, fd_config: FDConfig, prefix: str = ""):
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])
        hidden_size = fd_config.model_config.hidden_size

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
        )
        self.self_attn = GptOssAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )
        self.mlp = GptOssMoe(fd_config, layer_id, prefix=f"{prefix}.mlp")

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

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_graph_optimization
class GptOssModel(nn.Layer):

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            prefix="model.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                GptOssDecoderLayer(
                    fd_config=fd_config,
                    prefix=f"model.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix="model.norm",
        )

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        hidden_states = self.norm(hidden_states, residual)[0]
        return hidden_states


@ModelRegistry.register_model_class(
    architecture="GptOssForCausalLM",
    module_name="gpt_oss",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class GptOssForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super(GptOssForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.model = GptOssModel(fd_config=fd_config)

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(self):
        return "GptOssForCausalLM"

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
            ("attn.sinks", "sinks", None),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]
        expert_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("up_gate_proj_weight", "gate_up_proj", None, None),
            ("up_gate_proj_bias", "gate_up_proj_bias", None, None),
            ("down_proj_weight", "down_proj", None, None),
            ("down_proj_bias", "down_proj_bias", None, None),
        ]
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
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Loads the model weights. The complex weight loading and sharding logic
        from vLLM's `load_weights` should be adapted here or handled by the
        FastDeploy framework when loading a checkpoint.
        """
        assert False, "gpt-oss only support --load_choices default_v1."

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        return logits

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        return hidden_states
