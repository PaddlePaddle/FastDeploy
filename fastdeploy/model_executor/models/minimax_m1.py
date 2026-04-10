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
import re
from typing import Dict, Union

import numpy as np
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
    """MiniMax-M1 MoE Layer with low-bit quantization support."""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.norm_topk_prob = getattr(fd_config.model_config, "norm_topk_prob", False)

        # Build quantization-aware weight key map (mirrors Ernie4_5_MoE pattern)
        moe_quant_type = ""
        quant_config = getattr(fd_config, "quant_config", None)
        if quant_config and hasattr(quant_config, "moe_quant_type"):
            moe_quant_type = quant_config.moe_quant_type or ""

        is_quantized = getattr(fd_config.model_config, "is_quantized", False)
        moe_dynamic_quant = getattr(quant_config, "moe_dynamic_quant", False) if quant_config else False

        if moe_quant_type in ("w4a8", "tensor_wise_fp8", "block_wise_fp8") or (
            moe_quant_type == "w4afp8" and is_quantized and not moe_dynamic_quant
        ):
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4afp8" and is_quantized:
            # Dynamic w4afp8: no activation scales
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
            }
        else:
            # Default: unquantized
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
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
        linear_layer_id: int,  # Reserved for per-linear-layer indexing in future extensions
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = fd_config.model_config.hidden_size
        self.head_dim = fd_config.model_config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads // tp_size
        # Full (unsharded) inner dim for parallel linear layer declarations;
        # ColumnParallelLinear divides output and RowParallelLinear divides input
        # by tp_size internally.
        hidden_inner = fd_config.model_config.num_attention_heads * self.head_dim

        # QKV projection
        self.qkv_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
            input_size=self.hidden_size,
            output_size=hidden_inner * 3,
            with_bias=False,
        )

        # Output gate (sigmoid gating on attention output)
        self.output_gate = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.output_gate",
            input_size=self.hidden_size,
            output_size=hidden_inner,
            with_bias=False,
        )

        # Output projection (HF name: out_proj)
        self.out_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.out_proj",
            input_size=hidden_inner,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )

        # RMSNorm on attention output before gating (per-TP-rank dimension)
        self.norm = RMSNorm(
            fd_config,
            hidden_size=self.num_attention_heads * self.head_dim,
            eps=1e-5,
            prefix=f"{prefix}.norm",
        )

        # Build slope tensor for exponential decay; select this TP rank's subset
        slope_tensor = self._build_slope_tensor(fd_config.model_config.num_attention_heads)
        tp_rank = fd_config.parallel_config.tensor_parallel_rank
        slope_tensor = slope_tensor[tp_rank * self.num_attention_heads : (tp_rank + 1) * self.num_attention_heads]
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
        self.output_gate.load_state_dict(state_dict)
        self.out_proj.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)

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
        """Linear attention forward with output gating."""
        # Project QKV
        qkv = self.qkv_proj(hidden_states)
        hidden_inner = self.num_attention_heads * self.head_dim
        q, k, v = qkv.split([hidden_inner, hidden_inner, hidden_inner], axis=-1)

        # Apply SiLU activation (matches HF MiniMax convention)
        q = paddle.nn.functional.silu(q.astype("float32"))
        k = paddle.nn.functional.silu(k.astype("float32"))
        v = paddle.nn.functional.silu(v.astype("float32"))

        # Reshape for lightning attention
        batch_size = q.shape[0]
        q = q.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])
        k = k.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])
        v = v.reshape([batch_size, -1, self.num_attention_heads, self.head_dim])

        # Transpose to [batch, heads, seq_len, dim]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        # Retrieve or initialize KV history for recurrent state persistence.
        # TODO: Migrate to ForwardMeta.caches / slot-based cache management for
        #       proper multi-request isolation in production serving scenarios.
        if not hasattr(self, "_kv_history") or self._kv_history is None or self._kv_history.shape[0] != batch_size:
            self._kv_history = paddle.zeros(
                [batch_size, self.num_attention_heads, self.head_dim, self.head_dim],
                dtype=q.dtype,
            )

        # Apply lightning attention (returns 4D kv_history, not 5D concat)
        attn_output, new_kv_history = lightning_attention(
            q, k, v, self.slope_rate.squeeze(-1), block_size=256, kv_history=self._kv_history
        )
        # Update persisted KV state for next token generation
        self._kv_history = new_kv_history

        # Reshape back to [batch, seq, hidden_inner]
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, -1, self.num_attention_heads * self.head_dim])

        # Norm → gate → output projection (matches vLLM/HF forward)
        attn_output = self.norm(attn_output)[0]
        gate = self.output_gate(hidden_states)
        attn_output = paddle.nn.functional.sigmoid(gate) * attn_output.astype(hidden_states.dtype)
        output = self.out_proj(attn_output)
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

        # DeepNorm alpha/beta scaling — separate coefficients for linear vs full attention
        if self.attention_type == 0:  # Linear attention
            self.layernorm_attention_alpha = getattr(
                fd_config.model_config, "layernorm_linear_attention_alpha", 3.5565588200778455
            )
            self.layernorm_attention_beta = getattr(fd_config.model_config, "layernorm_linear_attention_beta", 1.0)
        else:  # Full attention
            self.layernorm_attention_alpha = getattr(
                fd_config.model_config, "layernorm_full_attention_alpha", 3.5565588200778455
            )
            self.layernorm_attention_beta = getattr(fd_config.model_config, "layernorm_full_attention_beta", 1.0)
        self.layernorm_mlp_alpha = getattr(fd_config.model_config, "layernorm_mlp_alpha", 3.5565588200778455)
        self.layernorm_mlp_beta = getattr(fd_config.model_config, "layernorm_mlp_beta", 1.0)

        # FFN (MLP or MoE)
        if fd_config.model_config.num_local_experts > 1:
            self.block_sparse_moe = MiniMaxM1MoE(
                fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.block_sparse_moe = MiniMaxM1MLP(
                fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}.mlp",
                reduce_results=True,
            )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.block_sparse_moe.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        """Decoder layer forward with DeepNorm.

        When postnorm=True (MiniMax-M1 default), the residual stream carries the
        *normed* activations rather than the pre-norm sum.  This follows the
        vLLM reference: ``residual = layernorm_output if postnorm else layernorm_input``.
        """
        # Input layernorm  (fused: x + residual → norm)
        hidden_states, residual = self.input_layernorm(
            hidden_states,
            residual_input=residual,
            forward_meta=forward_meta,
        )
        # hidden_states = norm(input + prev_residual)
        # residual      = input + prev_residual  (pre-norm)
        if self.postnorm:
            residual = hidden_states  # postnorm: residual = normed output

        # Attention (dispatch based on type)
        attn_output = self.self_attn(forward_meta=forward_meta, hidden_states=hidden_states)

        # DeepNorm alpha/beta scaling
        residual = residual * self.layernorm_attention_alpha
        attn_output = attn_output * self.layernorm_attention_beta

        # Post-attention layernorm
        if self.postnorm:
            layernorm_input = residual + attn_output
            hidden_states, residual = self.post_attention_layernorm(
                layernorm_input,
                forward_meta=forward_meta,
            )
            residual = hidden_states  # postnorm: residual = normed output
        else:
            hidden_states, residual = self.post_attention_layernorm(
                attn_output,
                residual_input=residual,
                forward_meta=forward_meta,
            )

        # FFN
        mlp_output = self.block_sparse_moe(hidden_states, forward_meta)

        # DeepNorm MLP alpha/beta
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
    architecture="MiniMaxM1ForCausalLM",
    module_name="minimax_m1",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
@ModelRegistry.register_model_class(
    architecture="MiniMaxText01ForCausalLM",
    module_name="minimax_m1",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """MiniMax-M1 Causal LM Model"""

    # Mapping HF checkpoint names → FD merged parameter names.
    # For full attention layers: separate q/k/v → merged qkv_proj
    # For MoE: gate_proj/up_proj → merged gate_up_proj (dense MLP fallback)
    _STACKED_PARAMS_MAPPING = [
        # (fd_param_name, hf_weight_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

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
        return "MiniMaxM1ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """Load model parameters from a given state dictionary.

        Pre-processes HF weight keys to match FD naming conventions, then
        delegates to sub-layer ``load_state_dict`` calls.
        """
        renamed: Dict[str, Union[np.ndarray, paddle.Tensor]] = {}
        # Collect full-attention q/k/v weights for merging into qkv_proj
        qkv_buffers: Dict[str, Dict[str, Union[np.ndarray, paddle.Tensor]]] = {}

        for name, weight in list(state_dict.items()):
            # Expert weights: w1→gate_proj, w3→up_proj, w2→down_proj
            # Handles both .weight (FP) and .quant_weight / .weight_scale / .activation_scale (quantized)
            if "block_sparse_moe.experts." in name:
                name = re.sub(r"\.w1\.", ".gate_proj.", name)
                name = re.sub(r"\.w3\.", ".up_proj.", name)
                name = re.sub(r"\.w2\.", ".down_proj.", name)
                renamed[name] = weight
            # Full attention: merge separate q/k/v into qkv_proj
            elif ".self_attn.q_proj." in name or ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
                # Extract layer prefix: e.g. "model.layers.7.self_attn"
                prefix_match = re.match(
                    r"(.*\.self_attn)\.(q|k|v)_proj\.(weight|quant_weight|weight_scale|activation_scale)$", name
                )
                if prefix_match:
                    attn_prefix = prefix_match.group(1)
                    proj_type = prefix_match.group(2)
                    suffix = prefix_match.group(3)
                    buf_key = f"{attn_prefix}|{suffix}"
                    if buf_key not in qkv_buffers:
                        qkv_buffers[buf_key] = {}
                    qkv_buffers[buf_key][proj_type] = weight
                else:
                    renamed[name] = weight
            else:
                renamed[name] = weight

        # Merge q/k/v into qkv_proj for full attention layers
        for buf_key, projections in qkv_buffers.items():
            if "q" in projections and "k" in projections and "v" in projections:
                attn_prefix, suffix = buf_key.split("|", 1)
                q_w = projections["q"]
                k_w = projections["k"]
                v_w = projections["v"]
                if isinstance(q_w, np.ndarray):
                    merged = np.concatenate([q_w, k_w, v_w], axis=0)
                else:
                    merged = paddle.concat([q_w, k_w, v_w], axis=0)
                renamed[f"{attn_prefix}.qkv_proj.{suffix}"] = merged

        self.model.load_state_dict(renamed)
        self.lm_head.load_state_dict(renamed)

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """Load model parameters from a weights iterator (v1 loader path).

        Handles HF→FD name mapping for:
        - Full attention: q_proj/k_proj/v_proj → qkv_proj (stacked)
        - MoE experts: w1/w3 → up_gate_proj, w2 → down_proj
        """
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = list(self._STACKED_PARAMS_MAPPING)

        # Expert weight mapping: HF w1/w2/w3 → FD up_gate_proj/down_proj
        n_experts = getattr(self.fd_config.model_config, "num_local_experts", 1)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=n_experts,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")

            # Stacked params (q/k/v → qkv_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                # Skip expert weights — handled separately
                if "block_sparse_moe.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Expert params (w1/w2/w3 → up_gate_proj/down_proj)
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    # Direct loading (norm, embed, lm_head, output_gate, out_proj, etc.)
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            # Note: model_param_name and param are guaranteed to be set here.
            # All three branches (stacked, expert, direct) set them before break;
            # when no branch matches, direct loading's `continue` skips to the
            # next outer iteration, so this line is never reached without them.
            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

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
        return "MiniMaxM1ForCausalLM"

    @classmethod
    def name(cls):
        """Model name."""
        return "MiniMaxM1ForCausalLM"
