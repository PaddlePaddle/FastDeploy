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
from paddlenlp.transformers import PretrainedModel

from fastdeploy.config import LLMConfig, ModelConfig
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.worker.model_runner import ForwardMeta


class Qwen2MLP(nn.Layer):
    """
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nranks = llm_config.parallel_config.mp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.up_gate_proj",
            with_bias=False,
            activation=llm_config.model_config.hidden_act,
            use_fast_ffn=True,
        )

        self.down_proj = RowParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.down_proj",
            input_size=(llm_config.model_config.ffn_hidden_size //
                        self.nranks),
            output_size=llm_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            llm_config=llm_config,
            bias=getattr(self.gate_up_proj, "linear_bias", None),
            act_method=llm_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x):
        """
        """
        gate_up_out = self.gate_up_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Qwen2Attention(nn.Layer):
    """
    """

    def __init__(self,
                 llm_config: LLMConfig,
                 layer_id: int,
                 prefix: str = "") -> None:
        super().__init__()

        nranks = llm_config.parallel_config.mp_size

        self.qkv_proj = QKVParallelLinear(llm_config=llm_config,
                                          prefix=f"{prefix}.qkv_proj",
                                          with_bias=True)

        self.o_proj = RowParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.o_proj",
            input_size=(llm_config.model_config.hidden_size // nranks),
            output_size=llm_config.model_config.hidden_size,
        )

        self.attn = Attention(llm_config=llm_config,
                              layer_id=layer_id,
                              prefix=prefix,
                              use_neox_rotary_style=True)

    def load_state_dict(self, state_dict):
        """
        """
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """
        """
        qkv_out = self.qkv_proj(hidden_states)

        atten_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(atten_out)
        return output


class Qwen2DecoderLayer(nn.Layer):
    """
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Qwen2Attention(
            llm_config=llm_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Qwen2MLP(
            llm_config=llm_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        """
        """
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
        """
        """
        # Self Attention
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

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen2Model(nn.Layer):
    """
    """

    def __init__(
        self,
        llm_config: LLMConfig = None,
    ):
        """
        Initializer for the Qwen2Model class.

        Args:

        """
        super().__init__()

        self.num_layers = llm_config.model_config.num_layers
        llm_config.model_config.prefix_name = "qwen2"

        self.embeddings = VocabParallelEmbedding(
            llm_config=llm_config,
            num_embeddings=llm_config.model_config.vocab_size,
            embedding_dim=llm_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{llm_config.model_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList([
            Qwen2DecoderLayer(
                llm_config=llm_config,
                prefix=f"{llm_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ])

        self.norm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{llm_config.model_config.prefix_name}.norm",
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


class Qwen2ForCausalLM(ModelForCasualLM):
    """
    Qwen2ForCausalLM
    """

    def __init__(self, llm_config: LLMConfig):
        """
        Args:
            llm_config (LLMConfig): Configurations for the LLM model.
        """
        super(Qwen2ForCausalLM, self).__init__(llm_config)

        self.model = Qwen2Model(llm_config=llm_config)

        self.ori_vocab_size = llm_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            llm_config=llm_config,
            embedding_dim=llm_config.model_config.hidden_size,
            num_embeddings=llm_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(self):
        """
        """
        return "Qwen2ForCausalLM"

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
        hidden_states = self.model(ids_remove_padding, forward_meta)

        return hidden_states


class Qwen2PretrainedModel(PretrainedModel):
    """
    Qwen2PretrainedModel
    """

    config_class = LLMConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn,
                                                            is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(
                    fn, is_column=True)
            else:
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
                    base_actions["layers.0.self_attn.k_proj.bias"] = partial(
                        fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.bias"] = partial(
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
