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


class MiniCPM4MLP(nn.Layer):
    """ """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
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
        """ """
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x, forward_meta):
        """ """
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class MiniCPM4Attention(nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.qkv_proj = QKVParallelLinear(fd_config=fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.hidden_size,
        )

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

    def load_state_dict(self, state_dict):
        """ """
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """ """
        qkv_out = self.qkv_proj(hidden_states)

        atten_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )
        output = self.o_proj(atten_out)
        return output


class MiniCPM4DecoderLayer(nn.Layer):
    """MiniCPM4 decoder layer with μP residual scaling."""

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        self.self_attn = MiniCPM4Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = MiniCPM4MLP(
            fd_config=fd_config,
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

        # μP residual scaling: scale_depth / sqrt(num_hidden_layers)
        scale_depth = getattr(fd_config.model_config, "scale_depth", 1.0)
        num_hidden_layers = fd_config.model_config.num_hidden_layers
        self.residual_scale = scale_depth / math.sqrt(num_hidden_layers)

    def load_state_dict(self, state_dict):
        """ """
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
        """ """
        # Self Attention
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        # μP: scale attention output before residual add
        hidden_states = hidden_states * self.residual_scale

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states, forward_meta)

        # μP: scale MLP output before residual add
        hidden_states = hidden_states * self.residual_scale

        return hidden_states, residual


@support_graph_optimization
class MiniCPM4Model(nn.Layer):
    """ """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "minicpm4"

        # μP embedding scaling factor
        self.scale_emb = getattr(fd_config.model_config, "scale_emb", 1)

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                MiniCPM4DecoderLayer(
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

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):

        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        # μP: scale embeddings
        if self.scale_emb != 1:
            hidden_states = hidden_states * self.scale_emb

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="MiniCPMForCausalLM",
    module_name="minicpm4",
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniCPM4ForCausalLM(ModelForCasualLM):
    """
    MiniCPM4ForCausalLM — supports MiniCPM4 and MiniCPM4.1 series models.

    Key differences from Qwen2:
    - μP (Maximal Update Parametrization) scaling:
      * Embedding output scaled by `scale_emb` (default: 12)
      * Residual connections scaled by `scale_depth / sqrt(num_hidden_layers)` (default: 1.4)
      * LM head input scaled by `hidden_size / dim_model_base` (default: 4096/256 = 16)
    - No QKV bias (attention_bias=false)
    - LongRoPE position encoding
    """

    def __init__(self, fd_config: FDConfig):
        super(MiniCPM4ForCausalLM, self).__init__(fd_config)

        self.fd_config = fd_config
        self.minicpm4 = MiniCPM4Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        # μP: lm_head input scaling factor = hidden_size / dim_model_base
        dim_model_base = getattr(fd_config.model_config, "dim_model_base", None)
        hidden_size = fd_config.model_config.hidden_size
        if dim_model_base is not None and dim_model_base > 0:
            self.lm_head_scale = hidden_size / dim_model_base
        else:
            self.lm_head_scale = 1.0

        self.process_weights_before_loading_fn = process_weights_before_loading(
            mapper=(
                WeightsMapper(orig_to_new_prefix={"model.": "minicpm4."})
                if self.fd_config.model_config.model_format == "torch"
                else None
            ),
        )

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.

        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
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
            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)
        if getattr(self, "tie_word_embeddings", False):
            self.lm_head.linear.weight.set_value(
                self.minicpm4.embed_tokens.embeddings.weight.transpose([1, 0]).astype(self.lm_head.linear.weight.dtype)
            )

    @classmethod
    def name(self):
        """ """
        return "MiniCPMForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.minicpm4.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        """ """
        # μP: scale hidden states before lm_head
        if self.lm_head_scale != 1.0:
            hidden_states = hidden_states / self.lm_head_scale
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def forward(
        self,
        inputs: Dict,
        forward_meta: ForwardMeta,
    ):
        ids_remove_padding = inputs["ids_remove_padding"]
        hidden_states = self.minicpm4(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.minicpm4.clear_grpah_opt_backend(fd_config=self.fd_config)


class MiniCPM4PretrainedModel(PretrainedModel):
    """
    MiniCPM4PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
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
                # Row Linear
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            if config.fuse_attention_qkv:
                base_actions["layers.0.self_attn.qkv_proj.weight"] = partial(fn, is_column=True)
            else:
                base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
                # MiniCPM4 has no QKV bias, only need weight splits
                if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
                    base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                    base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings
