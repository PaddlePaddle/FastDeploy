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

MiniMax-M1 Model for FastDeploy
Hybrid architecture: 70 linear attention layers + 10 full attention layers
MoE: 32 experts, top-2 routing per token
"""

from __future__ import annotations

import math
from typing import Dict

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
    ColumnParallelLinear,
    MergedColumnParallelLinear,
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
from fastdeploy.model_executor.ops.triton_ops.lightning_attn import lightning_attention


class MiniMaxM1MLP(nn.Layer):
    """MiniMax-M1 MLP Layer (Dense FFN)"""

    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate_up_proj",
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
            bias=getattr(self.gate_up_proj, "bias", None),
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x, forward_meta=None):
        gate_up_out = self.gate_up_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class MiniMaxM1MoE(nn.Layer):
    """MiniMax-M1 MoE Layer"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.norm_topk_prob = getattr(fd_config.model_config, "norm_topk_prob", False)

        weight_key_map = {
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.num_local_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        self.experts = FusedMoE(
            fd_config=fd_config,
            reduce_results=True,
            renormalize=self.norm_topk_prob,
            moe_intermediate_size=fd_config.model_config.intermediate_size,
            num_experts=fd_config.model_config.num_local_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        """Forward pass with router gating."""
        moe_out = self.experts(hidden_states, self.gate, forward_meta)
        if self.tp_size > 1:
            moe_out = tensor_model_parallel_all_reduce(moe_out)
        return moe_out


class MiniMaxM1Attention(nn.Layer):
    """MiniMax-M1 Full Attention (standard GQA)"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.head_dim = fd_config.model_config.head_dim
        self.num_key_value_heads = fd_config.model_config.num_key_value_heads

        self.qkv_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            input_size=self.hidden_size,
            output_size=(self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            with_bias=False,
        )

        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            with_bias=False,
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
        """Full attention forward."""
        q, k, v = self._compute_qkv(hidden_states)
        attn_output = self.attn(q, k, v, forward_meta=forward_meta)
        output = self.o_proj(attn_output)
        return output

    def _compute_qkv(self, hidden_states):
        """Project hidden states to Q, K, V."""
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], axis=-1)
        return q, k, v


class MiniMaxM1LinearAttention(nn.Layer):
    """MiniMax-M1 Linear Attention (Lightning Attention)"""

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        linear_layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = fd_config.model_config.hidden_size
        self.head_dim = fd_config.model_config.head_dim
        self.num_attention_heads = fd_config.model_config.num_attention_heads

        # QKV projection
        self.qkv_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            input_size=self.hidden_size,
            output_size=self.num_attention_heads * self.head_dim * 3,
            with_bias=False,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )

        # Build slope tensor for exponential decay
        slope_tensor = self._build_slope_tensor(self.num_attention_heads)
        if fd_config.model_config.num_hidden_layers <= 1:
            slope_tensor = slope_tensor * (1 + 1e-5)
        else:
            slope_tensor = slope_tensor * (1 - layer_id / (fd_config.model_config.num_hidden_layers - 1) + 1e-5)
        # Register as buffer (not trainable)
        self.register_buffer("slope_rate", slope_tensor)

        # KV cache shape: [heads, head_dim, head_dim]
        self.kv_cache_shape = (self.num_attention_heads, self.head_dim, self.head_dim)

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

    @staticmethod
    def _build_slope_tensor(n_heads: int):
        """Build ALiBi-style slope tensor for exponential decay."""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** (-(math.log2(n) - 3))))
            return [start * (start**i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power)
            slopes += get_slopes_power_of_2(2 * closest_power)[0::2][: n_heads - closest_power]

        return paddle.to_tensor(slopes, dtype=paddle.float32).reshape([n_heads, 1, 1])

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        """Linear attention forward."""
        # Project QKV
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [
                self.num_attention_heads * self.head_dim,
                self.num_attention_heads * self.head_dim,
                self.num_attention_heads * self.head_dim,
            ],
            axis=-1,
        )

        # Reshape for lightning attention
        batch_size = q.shape[0]
        q = q.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])
        k = k.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])
        v = v.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])

        # Transpose to [batch, heads, seq_len, dim]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        # Initialize KV history if needed
        kv_history = paddle.zeros(
            [batch_size, self.num_attention_heads, self.head_dim, self.head_dim],
            dtype=q.dtype,
        )

        # Apply lightning attention
        attn_output, _ = lightning_attention(
            q, k, v, self.slope_rate.squeeze(-1), block_size=256, kv_history=kv_history
        )

        # Reshape back
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, -1, self.num_attention_heads * self.head_dim])

        # Output projection
        output = self.o_proj(attn_output)
        return output


class MiniMaxM1DecoderLayer(nn.Layer):
    """MiniMax-M1 Decoder Layer with Hybrid Attention Dispatch"""

    @staticmethod
    def _build_attn_type_list(num_layers: int):
        """Build attention type list: 70 linear + 10 full (at indices 7,15,23,...)."""
        attn_type_list = [0] * num_layers  # Default: all linear
        # Full attention every 8 layers starting at layer 7
        full_attn_indices = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]
        for idx in full_attn_indices:
            if idx < num_layers:
                attn_type_list[idx] = 1
        return attn_type_list

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = fd_config.model_config.hidden_size
        self.layer_id = layer_id
        self.postnorm = getattr(fd_config.model_config, "postnorm", False)

        # Determine attention type for this layer
        # attn_type_list: 70 linear (0) + 10 full (1) at specific indices
        attn_type_list = getattr(
            fd_config.model_config,
            "attn_type_list",
            self._build_attn_type_list(fd_config.model_config.num_hidden_layers),
        )
        self.attention_type = attn_type_list[layer_id] if layer_id < len(attn_type_list) else 1

        # Attention layer (dispatch based on type)
        if self.attention_type == 0:  # Linear attention
            linear_layer_id = sum(1 for i in range(layer_id) if attn_type_list[i] == 0)
            self.self_attn = MiniMaxM1LinearAttention(
                fd_config,
                layer_id=layer_id,
                linear_layer_id=linear_layer_id,
                prefix=f"{prefix}.self_attn",
            )
        else:  # Full attention
            self.self_attn = MiniMaxM1Attention(
                fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
            )

        # Input layernorm (pre-norm)
        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
        )

        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
        )

        # DeepNorm alpha/beta scaling
        self.layernorm_attention_alpha = getattr(
            fd_config.model_config, "layernorm_full_attention_alpha", 3.5565588200778455
        )
        self.layernorm_attention_beta = getattr(fd_config.model_config, "layernorm_full_attention_beta", 1.0)
        self.layernorm_mlp_alpha = getattr(fd_config.model_config, "layernorm_mlp_alpha", 3.5565588200778455)
        self.layernorm_mlp_beta = getattr(fd_config.model_config, "layernorm_mlp_beta", 1.0)

        # FFN (MLP or MoE)
        if fd_config.model_config.num_local_experts > 1:
            self.mlp = MiniMaxM1MoE(
                fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiniMaxM1MLP(
                fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}.mlp",
                reduce_results=True,
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
        """Decoder layer forward with DeepNorm."""
        # Pre-norm
        hidden_states, residual = self.input_layernorm(
            hidden_states,
            residual_input=residual,
            forward_meta=forward_meta,
        )

        # Attention (dispatch based on type)
        if self.attention_type == 1:  # Full attention
            attn_output = self.self_attn(forward_meta=forward_meta, hidden_states=hidden_states)
        else:  # Linear attention
            attn_output = self.self_attn(forward_meta=forward_meta, hidden_states=hidden_states)

        # DeepNorm alpha/beta scaling
        residual = residual * self.layernorm_attention_alpha
        attn_output = attn_output * self.layernorm_attention_beta

        # Post-attention
        hidden_states, residual = self.post_attention_layernorm(attn_output, residual)

        # FFN
        mlp_output = self.mlp(hidden_states, forward_meta)

        # DeepNorm MLPalpha/beta
        residual = residual * self.layernorm_mlp_alpha
        mlp_output = mlp_output * self.layernorm_mlp_beta

        hidden_states = residual + mlp_output

        return hidden_states, residual


@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    """MiniMax-M1 Transformer Model"""

    def __init__(self, fd_config: FDConfig = None):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        self.hidden_size = fd_config.model_config.hidden_size
        fd_config.model_config.pretrained_config.prefix_name = "model"

        # Embedding
        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )

        # Decoder layers
        self.layers = nn.LayerList(
            [
                MiniMaxM1DecoderLayer(
                    fd_config,
                    layer_id=i,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        # Final layernorm
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """Load model parameters."""
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
        """Model forward pass."""
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None

        # Pass through decoder layers
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta=forward_meta,
                hidden_states=hidden_states,
                residual=residual,
            )

        # Final layernorm
        hidden_states = self.norm(hidden_states, residual)[0]

        return hidden_states


@ModelRegistry.register_model_class(
    architecture="MiniMaxText01ForCausalLM",
    module_name="minimax_m1",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """MiniMax-M1 Causal LM Model"""

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        """Model name."""
        return "MiniMaxText01ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict):
        """Load model parameters."""
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        """Compute logits."""
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        return logits

    def forward(
        self,
        inputs: Dict,
        forward_meta: ForwardMeta,
    ):
        """Forward pass."""
        ids_remove_padding = inputs["ids_remove_padding"]

        hidden_states = self.model(ids_remove_padding, forward_meta)
        return hidden_states


class MiniMaxM1PretrainedModel(PretrainedModel):
    """MiniMax-M1 Pretrained Model"""

    config_class = FDConfig

    @classmethod
    def arch_name(cls):
        """Architecture name."""
        return "MiniMaxText01ForCausalLM"

    @classmethod
    def name(cls):
        """Model name."""
        return "MiniMaxText01ForCausalLM"
