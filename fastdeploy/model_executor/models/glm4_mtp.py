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

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.mtp_linear import ParallelEHProjection
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.glm4_moe import Glm4MoeDecoderLayer
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)


class Glm4MTPPretrainedModel(PretrainedModel):
    """
    Glm4MTPPretrainedModel
    """

    config_class = FDConfig

    def _init_weights(self, layer):
        return None

    @classmethod
    def arch_name(self):
        return "Glm4MTPForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.info("Glm4MTP inference model _get_tensor_parallel_mappings")

        from fastdeploy.model_executor.models.tp_utils import split_or_merge_func_v1

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        def get_tensor_parallel_split_mappings(num_mtp_layers, mtp_start_layer_idx):
            final_actions = {}

            base_actions = {
                "layers.0.embed_tokens.weight": partial(fn, is_column=True),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            # Self Attention Layer which are need TP.
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.k_proj.bias"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.v_proj.bias"] = partial(fn, is_column=True)

            # Moe Layer
            for expert_idx in range(config.n_routed_experts):
                base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)

            base_actions["layers.0.eh_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.shared_head.head.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(mtp_start_layer_idx, mtp_start_layer_idx + num_mtp_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_nextn_predict_layers, config.start_layer_index)
        return mappings


class SharedHead(nn.Module):
    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.shared_head.norm",
        )
        self.head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix=f"{prefix}.shared_head.head",
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        # NOTE(wangyanpeng04): This only passes through the normalization layer and skips the head layer
        return self.norm(hidden_states)


class Glm4MTPLayer(nn.Layer):
    """
    Glm4MTPLayer
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
        prefix: str = "",
    ) -> None:
        """
        Initializer for the Glm4MTPLayer class.
        """
        super().__init__()

        self.enorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.enorm",
        )
        self.hnorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.hnorm",
        )
        self.eh_proj = ParallelEHProjection(
            fd_config,
            num_embeddings=fd_config.model_config.hidden_size,
            embedding_dim=fd_config.model_config.hidden_size * 2,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = SharedHead(
            fd_config,
            prefix=prefix,
        )
        self.mtp_block = Glm4MoeDecoderLayer(
            fd_config,
            prefix=prefix,
            is_mtp=True,
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        previous_hidden_states: paddle.Tensor,
        inputs_embedding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        forward
        """
        assert inputs_embedding is not None

        inputs_embedding = paddle.concat(
            [self.enorm(inputs_embedding)[0], self.hnorm(previous_hidden_states)[0]],
            axis=-1,
        )

        hidden_states = self.eh_proj(inputs_embedding)
        hidden_states, residual = self.mtp_block(forward_meta, hidden_states, residual=None)

        hidden_states = residual + hidden_states
        return hidden_states


@support_graph_optimization
class Glm4MTPModel(nn.Layer):
    """
    Glm4MTPModel
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ) -> None:
        super().__init__()

        self.mtp_start_layer_idx = fd_config.model_config.start_layer_index
        self.num_mtp_layers = fd_config.model_config.num_nextn_predict_layers

        assert self.num_mtp_layers == 1, f"Currently only supports single MTP layer, but got {self.num_mtp_layers}"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix=(
                f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{self.mtp_start_layer_idx}.embed_tokens"
            ),
        )

        self.layers = nn.LayerDict(
            {
                str(i): Glm4MTPLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(0, self.num_mtp_layers)
            }
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        previous_hidden_states: paddle.Tensor,
        forward_meta: ForwardMeta,
        inputs_embedding: paddle.Tensor = None,
    ):
        if inputs_embedding is None:
            inputs_embedding = self.embed_tokens(ids_remove_padding)

        # NOTE(wangyanpeng04): Currently only supports single MTP layer
        hidden_states = self.layers[str(0)](
            ids_remove_padding,
            previous_hidden_states,
            inputs_embedding,
            forward_meta,
        )

        return hidden_states


@ModelRegistry.register_model_class(
    architecture="Glm4MTPForCausalLM",
    module_name="glm4_mtp",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Glm4MTPForCausalLM(ModelForCasualLM):
    """
    Glm4MTPForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Glm4MTPForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.model = Glm4MTPModel(fd_config)
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = fd_config.speculative_config.sharing_model.lm_head
        self.mtp_start_layer_idx = fd_config.model_config.start_layer_index
        self.num_mtp_layers = fd_config.model_config.num_nextn_predict_layers

    @classmethod
    def name(self):
        return "Glm4MTPForCausalLM"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        """
        Load model parameters from a given weights_iterator object.

        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """

        from fastdeploy.model_executor.models.glm4_moe import Glm4MoeForCausalLM
        from fastdeploy.model_executor.utils import remap_weight_keys

        template = {
            "enorm": "enorm",
            "hnorm": "hnorm",
            "eh_proj": "eh_proj.linear",
            "shared_head.norm": "shared_head.norm",
            "shared_head.head": "shared_head.head.linear",
            "self_attn.q_proj": "mtp_block.self_attn.q_proj",
            "self_attn.k_proj": "mtp_block.self_attn.k_proj",
            "self_attn.v_proj": "mtp_block.self_attn.v_proj",
            "self_attn.o_proj": "mtp_block.self_attn.o_proj",
            "mlp": "mtp_block.mlp",
            "input_layernorm": "mtp_block.input_layernorm",
            "post_attention_layernorm": "mtp_block.post_attention_layernorm",
        }
        remap = {
            f"layers.{self.mtp_start_layer_idx}.embed_tokens": "embed_tokens.embeddings",
        }

        # NOTE (wangyanpeng) Here we need to map the layer_id of MTP weights to start from 0,
        # otherwise there will be out-of-bounds when accessing kv_cache in Attention
        for key, value in template.items():
            for mtp_layer_id in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers):
                remap[f"layers.{mtp_layer_id}.{key}"] = f"layers.{mtp_layer_id - self.mtp_start_layer_idx}.{value}"

        weights_iterator = remap_weight_keys(
            weights_iterator,
            remap,
            include_keys=[
                f"layers.{mtp_layer_id}"
                for mtp_layer_id in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
            ],
        )

        Glm4MoeForCausalLM.load_weights(
            self,
            weights_iterator,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        glm4_mtp only support loader_v1.
        """
        assert False, "glm4_mtp only support --load-choices default_v1."

    def compute_logits(self, hidden_state: paddle.Tensor, forward_meta: ForwardMeta):
        """
        compute_logits
        """
        logits = self.lm_head(hidden_state)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def empty_input_forward(self, forward_meta):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        self.model.layers[str(0)].mtp_block.mlp.experts(
            fake_hidden_states,
            self.model.layers[str(0)].mtp_block.mlp.gate,
            forward_meta,
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        previous_hidden_states: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        forward
        """
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            previous_hidden_states=previous_hidden_states,
            forward_meta=forward_meta,
        )

        return hidden_states
