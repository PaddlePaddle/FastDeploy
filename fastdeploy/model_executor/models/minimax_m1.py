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
Reference: https://github.com/MiniMax-AI/MiniMax-M1

Key features:
- Hybrid Attention: Lightning Attention (attn_type=0) + Standard Softmax Attention (attn_type=1)
- MoE Architecture: 32 experts, 2 experts activated per token
- 1M context length (10M max_position_embeddings)
- Shared experts with sigmoid routing
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.layers.rotary_embedding import (
    QwenRotaryEmbedding,
)
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.attention.attention import Attention


@ModelRegistry.register_model_class(
    architecture="MiniMaxM1ForCausalLM",
    module_name="minimax_m1",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """
    MiniMax-M1 model for causal language modeling
    
    MiniMax-M1 features:
    - Hybrid Attention: Lightning Attention + Standard Attention
    - MoE with 32 experts, 2 experts activated per token
    - 1M context length
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        
        # Parse config
        config = fd_config.model_config
        self.ori_vocab_size = getattr(config, "ori_vocab_size", getattr(config, "vocab_size", None))
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Model components
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=config.hidden_size,
            num_embeddings=config.vocab_size,
            prefix="lm_head",
        )
        
        # Tie weights if needed
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        pass

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        """Load model weights"""
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(
            dict(self.named_sublayers()), self.fd_config
        )

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            
            # Replace prefix
            loaded_weight_name = loaded_weight_name.replace("model.minimax_m1", "model")
            
            if loaded_weight_name in params_dict:
                weight = params_dict[loaded_weight_name]
                default_weight_loader(weight, loaded_weight)
            else:
                logger.warning(f"Weight {loaded_weight_name} not found in model")

    def compute_logits(self, hidden_states: paddle.Tensor, **logits_processor_kwargs):
        """Compute logits from hidden states"""
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        if hasattr(self, "ori_vocab_size") and getattr(self, "ori_vocab_size") is not None:
            logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass"""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def compute_logits(self, hidden_states: paddle.Tensor):
        """Compute logits from hidden states."""
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        return logits


class MiniMaxM1Model(nn.Layer):
    """MiniMax-M1 transformer model"""

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        
        self.config = fd_config.model_config
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        
        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            prefix="model.embed_tokens",
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
        )
        
        # Get attention type list (0=Lightning, 1=Standard)
        self.attn_type_list = getattr(self.config, "attn_type_list", [1] * self.config.num_hidden_layers)
        
        # Build decoder layers with hybrid attention
        self.layers = nn.LayerList()
        for layer_id in range(self.config.num_hidden_layers):
            attn_type = self.attn_type_list[layer_id] if layer_id < len(self.attn_type_list) else 1
            layer = MiniMaxM1DecoderLayer(
                fd_config=fd_config,
                layer_id=layer_id,
                attention_type=attn_type,
                prefix=f"model.layers.{layer_id}",
            )
            self.layers.append(layer)
        
        # Final layer norm
        self.norm = RMSNorm(
            fd_config=fd_config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            prefix="model.norm",
        )
        
        # Build slope tensor for Lightning Attention
        self.slopes = self._build_slope_tensor(self.config.num_attention_heads)

    def _build_slope_tensor(self, n_attention_heads: int):
        """Build slope tensor for linear attention"""
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                )
        
        slopes = paddle.to_tensor(
            get_slopes(n_attention_heads), 
            dtype="float32"
        ).reshape([n_attention_heads, 1, 1])
        return slopes

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
    ):
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare slope rates for Lightning Attention
        slope_rates = []
        for idx in range(len(self.layers)):
            slope = self.slopes.clone()
            slope = slope * (1 - idx / (len(self.layers) - 1) + 1e-5)
            slope_rates.append(slope)
        
        # Forward through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                slope_rate=slope_rates[idx],
            )
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class MiniMaxM1DecoderLayer(nn.Layer):
    """MiniMax-M1 decoder layer with hybrid attention"""

    def __init__(
        self, 
        fd_config: FDConfig, 
        layer_id: int, 
        attention_type: int = 1,
        prefix: str = ""
    ):
        super().__init__()
        
        self.layer_id = layer_id
        self.attention_type = attention_type  # 0=Lightning, 1=Standard
        
        # Attention - choose based on attention type
        if attention_type == 0:
            # Lightning Attention (not fully implemented, fallback to standard)
            logger.info(f"Layer {layer_id}: Using Lightning Attention")
            self.self_attn = MiniMaxM1LightningAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        else:
            # Standard Flash Attention
            logger.info(f"Layer {layer_id}: Using Standard Attention")
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        
        # MoE Layer
        self.block_sparse_moe = MiniMaxM1SparseMoEBlock(fd_config, layer_id, prefix=f"{prefix}.mlp")
        
        # Layer norms
        self.input_layernorm = RMSNorm(
            fd_config=fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
        )
        
        self.post_attention_layernorm = RMSNorm(
            fd_config=fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        
        # Post-norm configuration
        self.postnorm = getattr(fd_config.model_config, "postnorm", True)
        
        # Layer normalization scaling factors
        self.layernorm_attention_alpha = getattr(
            fd_config.model_config, "layernorm_linear_attention_alpha", 3.5565588200778455
        ) if attention_type == 0 else getattr(
            fd_config.model_config, "layernorm_full_attention_alpha", 3.5565588200778455
        )
        self.layernorm_attention_beta = getattr(
            fd_config.model_config, "layernorm_linear_attention_beta", 1.0
        ) if attention_type == 0 else getattr(
            fd_config.model_config, "layernorm_full_attention_beta", 1.0
        )
        self.layernorm_mlp_alpha = getattr(
            fd_config.model_config, "layernorm_mlp_alpha", 3.5565588200778455
        )
        self.layernorm_mlp_beta = getattr(
            fd_config.model_config, "layernorm_mlp_beta", 1.0
        )
        
        # Shared experts (if configured)
        shared_intermediate = getattr(fd_config.model_config, "shared_intermediate_size", 0)
        self.shared_moe = shared_intermediate > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(
                fd_config, 
                intermediate_size=shared_intermediate,
                prefix=f"{prefix}.shared_experts"
            )
            self.coefficient = nn.Linear(
                fd_config.model_config.hidden_size, 
                1, 
                bias=False
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        slope_rate: paddle.Tensor | None = None,
    ):
        """Forward pass through decoder layer"""
        
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states
        
        # Self attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            slope_rate=slope_rate,
        )
        
        # Residual connection with scaling
        hidden_states = residual * self.layernorm_attention_alpha + hidden_states * self.layernorm_attention_beta
        
        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.postnorm:
            residual = hidden_states
        
        # MoE
        moe_hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        
        # Shared experts (if configured)
        if self.shared_moe:
            output_mlp = self.shared_mlp(hidden_states)
            coef = paddle.nn.functional.sigmoid(
                self.coefficient.weight.astype(hidden_states.dtype) @ hidden_states.astype("float32").T
            ).T
            coef = coef.astype(hidden_states.dtype)
            hidden_states = moe_hidden_states * (1 - coef) + output_mlp * coef
        else:
            hidden_states = moe_hidden_states
        
        # Residual connection with scaling
        hidden_states = residual * self.layernorm_mlp_alpha + hidden_states * self.layernorm_mlp_beta
        
        return hidden_states


class MiniMaxM1StandardAttention(nn.Layer):
    """Standard attention for MiniMax-M1"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        
        self.fd_config = fd_config
        self.layer_id = layer_id
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        
        num_kv_heads = fd_config.model_config.num_key_value_heads
        num_kv_heads_replicas = max(1, tp_size // num_kv_heads)
        
        # QKV projection
        self.q_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.q_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.num_attention_heads * self.head_dim,
            with_bias=False,
        )
        
        self.k_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.k_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=num_kv_heads * self.head_dim,
            with_bias=False,
        )
        
        self.v_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.v_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=num_kv_heads * self.head_dim,
            with_bias=False,
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.num_attention_heads * self.head_dim,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )
        
        # RoPE
        self.rotary_dim = getattr(fd_config.model_config, "rotary_dim", self.head_dim)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=self.rotary_dim,
            max_position_embeddings=fd_config.model_config.max_position_embeddings,
            base=fd_config.model_config.rope_theta,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        slope_rate: paddle.Tensor | None = None,
    ):
        """Forward pass through standard attention"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape([batch_size, seq_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([batch_size, seq_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([batch_size, seq_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        
        # RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        
        # Apply RoPE
        # Note: Full RoPE implementation would go here
        
        # Attention (using FastDeploy's built-in attention)
        attn = Attention(
            self.fd_config,
            layer_id=self.layer_id,
            prefix=self.prefix if hasattr(self, 'prefix') else "",
            use_neox_rotary_style=True,
        )
        
        # Simplified attention computation
        qkv = paddle.concat([query_states, key_states, value_states], axis=-1)
        attn_output = attn(qkv=qkv, cos=cos, sin=sin, attention_mask=attention_mask)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, None, None


class MiniMaxM1LightningAttention(nn.Layer):
    """
    Lightning Attention for MiniMax-M1
    
    Reference: https://github.com/MiniMax-AI/MiniMax-M1
    
    This implements the block-wise Lightning Attention algorithm:
    - Processes sequences in blocks of 256 tokens
    - Uses linear attention with causal masking via decay factors
    - Maintains KV state across blocks for efficiency
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        
        self.fd_config = fd_config
        self.layer_id = layer_id
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_heads = fd_config.model_config.num_attention_heads
        self.head_dim = fd_config.model_config.head_dim
        
        # Activation function
        self.act = self._get_activation_fn(fd_config.model_config.hidden_act)
        
        # QKV projection
        self.qkv_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            input_size=self.hidden_size,
            output_size=3 * self.num_heads * self.head_dim,
            with_bias=False,
        )
        
        # Output gate
        self.output_gate = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.output_gate",
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            with_bias=False,
        )
        
        # Output projection
        self.out_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.out_proj",
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )
        
        # Norm layer for output
        self.norm = RMSNorm(
            fd_config=fd_config,
            hidden_size=self.num_heads * self.head_dim,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.norm",
        )
        
        self.prefix = prefix
        self.block_size = 256  # MiniMax-M1 uses block size of 256
        
        # Inference state
        self.kv_state = None
        self.offset = 0

    def _get_activation_fn(self, activation: str):
        """Get activation function"""
        if activation == "silu" or activation == "swish":
            return paddle.nn.functional.silu
        elif activation == "gelu":
            return paddle.nn.functional.gelu
        elif activation == "relu":
            return paddle.nn.functional.relu
        else:
            return paddle.nn.functional.silu

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        slope_rate: paddle.Tensor | None = None,
    ):
        """
        Lightning Attention forward pass
        
        This is a Paddle implementation of the block-wise Lightning Attention.
        For optimal performance, custom CUDA/Triton kernels are recommended.
        """
        # hidden_states: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear map
        qkv = self.act(self.qkv_proj(hidden_states))
        new_shape = qkv.shape[:-1] + (self.num_heads, -1)
        qkv = qkv.reshape(new_shape)
        q, k, v = paddle.split(qkv, [self.head_dim] * 3, axis=-1)
        
        # Transpose: [batch, num_heads, seq_len, head_dim]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])
        
        # Compute decay ratio
        if slope_rate is not None:
            ratio = paddle.exp(-slope_rate)
        else:
            ratio = paddle.ones([1], dtype="float32")
        
        # First time seeing this sequence
        if self.kv_state is None or self.offset == 0:
            self.offset = q.shape[-2]
            output = self._forward_first_time(q, k, v, attention_mask, ratio)
        else:
            # Continuing from previous sequence
            self.offset += 1
            output = self._forward_with_cache(q, k, v, ratio)
        
        # Reshape: [batch, seq_len, num_heads * head_dim]
        output = output.transpose([0, 2, 1, 3])
        output = output.reshape([batch_size, seq_len, -1])
        
        # Normalize
        output = self.norm(output)
        
        # Gate
        gate = paddle.nn.functional.sigmoid(self.output_gate(hidden_states))
        output = gate * output
        
        # Output projection
        output = self.out_proj(output)
        
        return output, None, None

    def _forward_first_time(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        ratio: paddle.Tensor,
    ):
        """
        Forward pass for the first time seeing a sequence
        Uses block-wise computation for efficiency
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Prepare decay arrays
        block_indices = paddle.arange(self.block_size).astype("float32")
        q_decay = paddle.exp(-ratio * block_indices.reshape([-1, 1]))
        k_decay = paddle.exp(-ratio * (self.block_size - block_indices.reshape([-1, 1])))
        
        # Create diagonal decay matrix
        indices = block_indices[:, None] - block_indices[None, :]
        s_indices = ratio * indices
        s_indices = paddle.where(indices >= 0, -s_indices, paddle.full_like(s_indices, float("-inf")))
        diag_decay = paddle.exp(s_indices)
        
        # Initialize KV state and output
        kv = paddle.zeros([batch_size, num_heads, head_dim, head_dim], dtype="float32")
        output = paddle.empty_like(q, dtype=q.dtype)
        
        for i in range(num_blocks):
            si = i * self.block_size
            ei = min(si + self.block_size, seq_len)
            m = ei - si
            
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
            
            # Non-diagonal part
            qkv_none_diag = paddle.matmul(qi * q_decay[:, :m], kv.astype(qi.dtype)).astype("float32")
            
            # Diagonal part
            qk = paddle.matmul(qi, ki.transpose([0, 1, 3, 2])).astype("float32") * diag_decay[:, :, :m, :m]
            qkv_diag = paddle.matmul(qk, vi.astype("float32"))
            
            # Block decay
            block_decay = paddle.exp(-ratio * m)
            output[:, :, si:ei] = (qkv_none_diag + qkv_diag).astype(q.dtype)
            kv = block_decay * kv + paddle.matmul(
                (ki * k_decay[:, -m:]).transpose([0, 1, 3, 2]).astype(vi.dtype), vi
            )
        
        self.kv_state = kv
        return output

    def _forward_with_cache(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        ratio: paddle.Tensor,
    ):
        """
        Forward pass with KV cache (for continues generation)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Get KV state
        kv = self.kv_state
        
        output_list = []
        for i in range(seq_len):
            # Update KV state: kv = ratio * kv + k_i * v_i^T
            kv = ratio * kv + paddle.einsum(
                "... n d, ... n e -> ... d e",
                k[:, :, i:i + 1],
                v[:, :, i:i + 1],
            )
            # Compute output: q_i @ kv
            qkv = paddle.einsum(
                "... n e, ... e d -> ... n d",
                q[:, :, i:i + 1],
                kv.astype(q.dtype)
            )
            output_list.append(qkv)
        
        output = paddle.concat(output_list, axis=-2)
        return output


class MiniMaxM1MLP(nn.Layer):
    """MiniMax-M1 MLP (shared expert)"""

    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = ""):
        super().__init__()
        
        self.gate_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
        )
        
        self.up_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size,
            with_bias=False,
        )
        
        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )
        
        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=None,
            act_method=fd_config.model_config.hidden_act,
        )

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x), self.up_proj(x)))


class MiniMaxM1SparseMoEBlock(nn.Layer):
    """MiniMax-M1 Sparse MoE Block"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        
        self.hidden_dim = fd_config.model_config.hidden_size
        self.ffn_dim = fd_config.model_config.intermediate_size
        self.num_experts = fd_config.model_config.num_local_experts
        self.top_k = fd_config.model_config.num_experts_per_tok
        
        # Gate
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=self.hidden_dim,
            output_size=self.num_experts,
            with_bias=False,
            skip_quant=True,
        )
        
        # Experts (using FusedMoE from FastDeploy)
        self.experts = FusedMoE(
            fd_config=fd_config,
            reduce_results=True,
            renormalize=False,
            moe_intermediate_size=self.ffn_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            topk_method=getattr(fd_config.model_config, "topk_method", "noaux_tc"),
            topk_group=getattr(fd_config.model_config, "topk_group", 1),
            n_group=getattr(fd_config.model_config, "n_group", 1),
            routed_scaling_factor=getattr(fd_config.model_config, "routed_scaling_factor", 1.0),
            layer_idx=layer_id,
            weight_key_map={},
        )
        
        # Jitter noise
        self.jitter_noise = getattr(fd_config.model_config, "router_jitter_noise", 0.0)

    def forward(self, hidden_states: paddle.Tensor):
        """Forward pass through MoE"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])
        
        # Router
        router_logits = self.gate(hidden_states)[0]
        
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * paddle.uniform(
                hidden_states.shape, min=1.0 - self.jitter_noise, max=1.0 + self.jitter_noise
            )
        
        # Compute routing weights
        routing_weights = paddle.nn.functional.softmax(router_logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(routing_weights, self.top_k, axis=-1)
        routing_weights = routing_weights / routing_weights.sum(axis=-1, keepdim=True)
        
        # Process experts
        final_hidden_states = self.experts(hidden_states, self.gate, None)
        
        final_hidden_states = final_hidden_states.reshape([batch_size, seq_len, hidden_dim])
        
        return final_hidden_states, router_logits


# Model registration metadata
MiniMaxM1ForCausalLM.arch_name = "MiniMax-M1"