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

from __future__ import annotations

from functools import partial

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.graph_optimization.decorator import \
    support_graph_optimization
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (QKVParallelLinear,
                                                     RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2MLP
from fastdeploy.worker.forward_meta import ForwardMeta


class Qwen3MLP(Qwen2MLP):
    """
    """
    pass


class Qwen3Attention(nn.Layer):
    """
    """

    def __init__(self,
                 fd_config: FDConfig,
                 layer_id: int,
                 prefix: str = "") -> None:
        super().__init__()

        self.fd_config = fd_config

        self.head_dim = fd_config.model_config.head_dim
        nranks = fd_config.parallel_config.tensor_parallel_degree
        self.q_size = fd_config.model_config.num_attention_heads * self.head_dim // nranks
        self.kv_size = fd_config.model_config.num_key_value_heads * self.head_dim // nranks

        self.qkv_proj = QKVParallelLinear(fd_config=fd_config,
                                          prefix=f"{prefix}.qkv_proj",
                                          with_bias=False)

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim *
            fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
        )

        self.attn = Attention(fd_config=fd_config,
                              layer_id=layer_id,
                              prefix=prefix,
                              use_neox_rotary_style=True)

        self.q_norm = RMSNorm(fd_config=fd_config,
                              hidden_size=fd_config.model_config.head_dim,
                              eps=1e-6,
                              prefix=f"{prefix}.q_norm",
                              begin_norm_axis=2)
        self.k_norm = RMSNorm(fd_config=fd_config,
                              hidden_size=fd_config.model_config.head_dim,
                              eps=1e-6,
                              prefix=f"{prefix}.k_norm",
                              begin_norm_axis=2)

    def load_state_dict(self, state_dict):
        """
        """
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.q_norm.load_state_dict(state_dict)
        self.k_norm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """
        """
        qkv_out = self.qkv_proj(hidden_states)

        # origin_qkv_out = qkv_out
        q, k, v = qkv_out.split([self.q_size, self.kv_size, self.kv_size],
                                axis=-1)

        q_by_head = q.reshape(
            [*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim])
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.reshape(q.shape)

        k_by_head = k.reshape(
            [*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim])
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.reshape(k.shape)

        qkv_out = paddle.concat([q, k, v], axis=-1)

        atten_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(atten_out)
        return output


class Qwen3DecoderLayer(Qwen2DecoderLayer):
    """
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(fd_config, prefix)
        layer_id = int(prefix.split(sep='.')[-1])
        self.self_attn = Qwen3Attention(fd_config=fd_config,
                                        layer_id=layer_id,
                                        prefix=f"{prefix}.self_attn")


@support_graph_optimization
class Qwen3Model(nn.Layer):
    """
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Qwen3Model class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "model"
        fd_config.model_config.tie_word_embeddings = True

        self.embeddings = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList([
            Qwen3DecoderLayer(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ])

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-6,
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
            self.layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        """
        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta,
                                                     hidden_states, residual)

        hidden_states = hidden_states + residual

        out = self.norm(hidden_states)

        return out


class Qwen3ForCausalLM(ModelForCasualLM):
    """
    Qwen3ForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Qwen3ForCausalLM, self).__init__(fd_config)

        self.model = Qwen3Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"),
        )
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

    @classmethod
    def name(self):
        """
        """
        return "Qwen3ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
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
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """
        """
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        """
        hidden_states = self.model(ids_remove_padding=ids_remove_padding,
                                   forward_meta=forward_meta)

        return hidden_states


class Qwen3PretrainedModel(PretrainedModel):
    """
    Qwen3PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):

        from paddleformers.transformers.conversion_utils import \
            split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn,
                                                            is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear

            base_actions["layers.0.self_attn.q_proj.weight"] = partial(
                fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(
                fn, is_column=True)
            # if we have enough num_key_value_heads to split, then split it.
            if config.num_key_value_heads % config.tensor_parallel_degree == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(
                    fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(
                    fn, is_column=True)

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(
                fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(
                fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.",
                                                  f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_layers)
        return mappings
