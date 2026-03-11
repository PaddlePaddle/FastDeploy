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
MiniMax-M1 Model Support for FastDeploy

MiniMax-M1 is the world's first open-source large-scale hybrid attention reasoning model.
Key features:
- Hybrid Attention: Lightning Attention + Standard Softmax Attention
- MoE Architecture: 32 experts, 2 experts activated per token
- 1M context length (10M max_position_embeddings)
"""

from __future__ import annotations

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.layers.rotary_embedding import (
    LlamaRotaryEmbedding,
)


class MiniMaxM1Config:
    """MiniMax-M1 model configuration"""

    def __init__(self, config_dict):
        self.model_type = config_dict.get("model_type", "minimax_m1")
        self.hidden_size = config_dict.get("hidden_size", 6144)
        self.intermediate_size = config_dict.get("intermediate_size", 9216)
        self.num_hidden_layers = config_dict.get("num_hidden_layers", 80)
        self.num_attention_heads = config_dict.get("num_attention_heads", 64)
        self.num_key_value_heads = config_dict.get("num_key_value_heads", 8)
        self.head_dim = config_dict.get("head_dim", 128)
        self.vocab_size = config_dict.get("vocab_size", 200064)
        self.rope_theta = config_dict.get("rope_theta", 10000000)
        self.max_position_embeddings = config_dict.get("max_position_embeddings", 10240000)
        self.rms_norm_eps = config_dict.get("rms_norm_eps", 1e-5)
        self.hidden_act = config_dict.get("hidden_act", "silu")
        
        # MoE configuration
        self.num_local_experts = config_dict.get("num_local_experts", 32)
        self.num_experts_per_tok = config_dict.get("num_experts_per_tok", 2)
        
        # Hybrid attention configuration
        self.attn_type_list = config_dict.get("attn_type_list", [])
        
        # Layer normalization parameters
        self.layernorm_full_attention_alpha = config_dict.get("layernorm_full_attention_alpha", 3.5565588200778455)
        self.layernorm_full_attention_beta = config_dict.get("layernorm_full_attention_beta", 1.0)
        self.layernorm_linear_attention_alpha = config_dict.get("layernorm_linear_attention_alpha", 3.5565588200778455)
        self.layernorm_linear_attention_beta = config_dict.get("layernorm_linear_attention_beta", 1.0)
        self.layernorm_mlp_alpha = config_dict.get("layernorm_mlp_alpha", 3.5565588200778455)
        self.layernorm_mlp_beta = config_dict.get("layernorm_mlp_beta", 1.0)


@ModelRegistry.register_model_class
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """
    MiniMax-M1 model for causal language modeling
    """

    model_prefix = "minimax_m1"
    model_category = ModelCategory.LLM

    def __init__(self, fd_config: FDConfig, prefix: str = "") -> None:
        super().__init__(fd_config, prefix)
        
        self.config = MiniMaxM1Config(fd_config.model_config.to_dict())
        
        logger.info(f"Initializing MiniMax-M1 model with config: {self.config.__dict__}")
        
        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            prefix=f"{prefix}.model.embed_tokens",
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
        )
        
        # Decoder layers
        self.layers = nn.LayerList([
            MiniMaxM1DecoderLayer(fd_config, layer_id, prefix=f"{prefix}.model.layers.{layer_id}")
            for layer_id in range(self.config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(
            fd_config=fd_config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            prefix=f"{prefix}.model.norm",
        )
        
        # LM Head
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            prefix=f"{prefix}.lm_head",
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
        )
        
        self._tie_weights()

    def _tie_weights(self):
        """Tie embeddings and lm_head weights if configured"""
        if self.fd_config.model_config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass"""
        hidden_states = self.embed_tokens(input_ids)
        
        # Rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                cos=cos,
                sin=sin,
            )
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

    def get_rotary_embedding(self):
        """Get rotary embedding layer"""
        return LlamaRotaryEmbedding(
            dim=self.config.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )


class MiniMaxM1DecoderLayer(nn.Layer):
    """MiniMax-M1 decoder layer with hybrid attention and MoE"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        
        self.layer_id = layer_id
        self.prefix = prefix
        
        # Hybrid attention
        self.self_attn = MiniMaxM1Attention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        
        # MoE MLP
        self.mlp = MiniMaxM1MoE(fd_config, layer_id, prefix=f"{prefix}.mlp")
        
        # Post-attention layernorm
        self.post_attention_layernorm = RMSNorm(
            fd_config=fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        
        # Post-MLP layernorm  
        self.post_mlp_layernorm = RMSNorm(
            fd_config=fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_mlp_layernorm",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        cos: paddle.Tensor | None = None,
        sin: paddle.Tensor | None = None,
    ):
        """Forward pass through decoder layer"""
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states
        
        # Post-attention layernorm
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MoE MLP with residual
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Post-MLP layernorm
        hidden_states = self.post_mlp_layernorm(hidden_states)
        
        return hidden_states


class MiniMaxM1Attention(nn.Layer):
    """MiniMax-M1 attention with hybrid Lightning/Softmax attention support"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        
        # TODO: Implement hybrid attention
        # MiniMax-M1 uses attn_type_list to determine attention type per layer
        # Currently using standard attention as placeholder
        
        from fastdeploy.model_executor.layers.attention.attention import Attention
        from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
        
        self.fd_config = fd_config
        self.layer_id = layer_id
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        
        num_kv_heads = fd_config.model_config.num_key_value_heads
        num_kv_heads_replicas = max(1, tp_size // num_kv_heads)
        
        self.q_size = fd_config.model_config.num_attention_heads * self.head_dim // tp_size
        self.kv_size = num_kv_heads * self.head_dim * num_kv_heads_replicas // tp_size
        
        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            with_bias=False,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
            layer_id=layer_id,
        )
        
        # Attention
        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        cos: paddle.Tensor | None = None,
        sin: paddle.Tensor | None = None,
    ):
        """Forward pass through attention"""
        qkv = self.qkv_proj(hidden_states)
        
        # TODO: Implement Lightning Attention for layers with attn_type=0
        
        # For now, use standard attention
        attn_output = self.attn(
            qkv=qkv,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
        )
        
        output = self.o_proj(attn_output)
        return output


class MiniMaxM1MoE(nn.Layer):
    """MiniMax-M1 MoE (Mixture of Experts) layer"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        
        # TODO: Implement MoE layer for MiniMax-M1
        # MiniMax-M1 uses shared_moe_mode='sigmoid'
        # 32 experts, 2 experts activated per token
        
        from fastdeploy.model_executor.layers.moe.moe import FusedMoE
        
        self.fd_config = fd_config
        self.layer_id = layer_id
        
        # Placeholder - need to implement MiniMax-specific MoE
        logger.warning(
            f"MiniMax-M1 MoE layer {layer_id} using placeholder implementation. "
            "Full MoE support needs custom implementation."
        )
        
        # Fallback to dense MLP for now
        from fastdeploy.model_executor.models.qwen3 import Qwen3MLP
        
        self.mlp = Qwen3MLP(fd_config, layer_id, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: paddle.Tensor):
        """Forward pass through MoE"""
        return self.mlp(hidden_states)


# Model registration metadata
MiniMaxM1ForCausalLM.arch_name = "MiniMax-M1"