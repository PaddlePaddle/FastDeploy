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
from functools import partial
from typing import Dict, List

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention import GDNAttention
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVGateParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.normalization import QKRMSNorm, RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.models.qwen2 import Qwen2MLP
from fastdeploy.model_executor.models.qwen3 import Qwen3Attention
from fastdeploy.transformer_utils.config import get_pooling_config

# ==============================================================================
# GDN Helper Classes
# ==============================================================================


class RMSNormGated(nn.Layer):
    """
    Gated RMSNorm for Qwen3.5 GDN output normalization.

    This implements the gated normalization pattern from transformers:
    1. Apply RMSNorm to hidden_states
    2. Apply SiLU to gate
    3. Multiply: output = RMSNorm(hidden_states) * SiLU(gate)

    Reference: transformers/models/qwen3_5/modeling_qwen3_5.py Qwen3_5RMSNormGated
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: str = "float32",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(value=1.0),
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        gate: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Args:
            hidden_states: [*, hidden_size] - input to normalize
            gate: [*, hidden_size] - gate tensor (z projection output)

        Returns:
            output: [*, hidden_size] - normalized and gated output
        """
        input_dtype = hidden_states.dtype

        # Compute RMSNorm
        hidden_states_fp32 = hidden_states.cast(paddle.float32)
        variance = hidden_states_fp32.pow(2).mean(axis=-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * paddle.rsqrt(variance + self.eps)

        # Apply weight and cast back
        hidden_states_normed = self.weight * hidden_states_normed.cast(input_dtype)

        # Apply SiLU gate
        gate_fp32 = gate.cast(paddle.float32)
        output = hidden_states_normed * paddle.nn.functional.silu(gate_fp32)

        return output.cast(input_dtype)

    def weight_loader(self, param, loaded_weight):
        """Load weight from checkpoint."""
        param.set_value(loaded_weight.cast(param.dtype))


class Qwen3_5MLP(Qwen2MLP):
    """Qwen3.5 MLP Layer, same as Qwen2 MLP."""

    pass


class Qwen3_5GatedDeltaNet(nn.Layer):
    """
    Gated Delta Network (GDN) - Linear Attention Layer for Qwen3.5.

    This implements a linear complexity attention mechanism using:
    - QKVZ projection (Query, Key, Value, Z gate)
    - BA projection (Beta, Alpha gates)
    - RMSNorm for normalization
    - Causal Conv1d for sequence transformation (Triton kernel)
    - A_log and dt_bias for GDN gating
    - FLA Triton kernels for core attention (chunk / fused recurrent)
    - Pre-allocated GPU state pool for conv and SSM states

    Reference: transformers/models/qwen3_5/modeling_qwen3_5.py Qwen3_5GatedDeltaNet
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.fd_config = fd_config
        config = fd_config.model_config

        # Get GDN-specific parameters
        self.head_k_dim = getattr(config, "linear_key_head_dim", 128)
        self.head_v_dim = getattr(config, "linear_value_head_dim", 128)
        self.num_k_heads = getattr(config, "linear_num_key_heads", 16)
        self.num_v_heads = getattr(config, "linear_num_value_heads", 16)
        self.conv_kernel_size = getattr(config, "linear_conv_kernel_dim", 4)

        self.hidden_size = config.hidden_size
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        # For MoE models with different QKV structure, detect from config
        if self.num_v_heads != self.num_k_heads:
            self.qkv_size = 4 * self.hidden_size
        else:
            self.qkv_size = self.key_dim * 3

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank
        self.layer_idx = layer_id
        self.rms_norm_eps = config.rms_norm_eps

        # For GQA: how many value heads per key head
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads

        # TP-local dimensions
        self.num_k_heads_local = self.num_k_heads // self.tp_size
        self.num_v_heads_local = self.num_v_heads // self.tp_size
        self.conv_dim = (self.key_dim * 2 + self.value_dim) // self.tp_size

        # GDN layer index (among linear_attention layers only, set by model init)
        self.gdn_layer_idx = 0

        # QKV projection
        self.in_proj_qkv = ColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.in_proj_qkv",
            input_size=self.hidden_size,
            output_size=self.qkv_size,
            with_bias=False,
        )

        # Z projection
        self.in_proj_z = ColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.in_proj_z",
            input_size=self.hidden_size,
            output_size=self.value_dim,
            with_bias=False,
        )

        # Beta projection (small, no TP — replicated across all ranks)
        self.in_proj_b = ReplicatedLinear(
            fd_config,
            prefix=f"{prefix}.in_proj_b",
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            with_bias=False,
        )

        # Alpha projection (small, no TP — replicated across all ranks)
        self.in_proj_a = ReplicatedLinear(
            fd_config,
            prefix=f"{prefix}.in_proj_a",
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            with_bias=False,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.out_proj",
            input_size=self.value_dim,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )

        # RMSNormGated for output normalization
        self.norm = RMSNormGated(
            hidden_size=self.head_v_dim,
            eps=self.rms_norm_eps,
            dtype="float32",
        )

        # Conv1d weight: [conv_dim_global, conv_kernel_size]
        # After TP split by ColumnParallelLinear on in_proj_qkv/z, the actual
        # conv_dim used in forward is conv_dim_local. The weight is loaded
        # globally and sliced at runtime (or loaded per-rank if checkpoint is sharded).
        self.conv1d_weight = nn.Parameter(paddle.zeros([self.key_dim * 2 + self.value_dim, self.conv_kernel_size]))

        # A_log parameter for GDN gating [num_v_heads]
        self.A_log = nn.Parameter(paddle.zeros([self.num_v_heads]))

        # dt_bias parameter [num_v_heads]
        self.dt_bias = nn.Parameter(paddle.ones([self.num_v_heads]))

        # GDN attention trampoline (delegates to forward_meta.gdn_attn_backend)
        self.gdn_attn = GDNAttention(fd_config, layer_id)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ) -> paddle.Tensor:
        """GDN forward pass — delegates conv1d + gating + SSM to GDNAttentionBackend.

        Args:
            forward_meta: Forward metadata (contains GDN backend, state pool, slot IDs)
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # Part 1: Input Projection
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape([num_tokens, -1, self.head_v_dim])
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Part 2: Conv1d + Split + Gating + SSM (via GDNAttention trampoline → backend)
        core_attn_out = self.gdn_attn(mixed_qkv, a, b, self, forward_meta)

        # Part 3: Output Projection
        core_attn_out = core_attn_out.reshape([-1, self.head_v_dim])
        z = z.reshape([-1, self.head_v_dim])

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape([num_tokens, -1])

        core_attn_out = core_attn_out.cast(hidden_states.dtype)
        output = self.out_proj(core_attn_out)

        return output

    def load_state_dict(self, state_dict):
        """Load state dict for GDN layer."""
        self.in_proj_qkv.load_state_dict(state_dict)
        self.in_proj_z.load_state_dict(state_dict)
        self.in_proj_b.load_state_dict(state_dict)
        self.in_proj_a.load_state_dict(state_dict)
        self.out_proj.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)


class Qwen3_5Attention(Qwen3Attention):
    """Qwen3.5 Full Attention Layer with Gated Attention.

    Qwen3.5 uses a gated attention mechanism where:
    1. q_proj outputs query + gate (num_heads * head_dim * 2) when attn_output_gate=True
    2. After attention, output is multiplied by sigmoid(gate)

    Reference: transformers/models/qwen3_5/modeling_qwen3_5.py Qwen3_5Attention
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        # Don't call parent __init__, we need to customize for gated attention
        super(Qwen3Attention, self).__init__()

        self.fd_config = fd_config
        config = fd_config.model_config
        self.head_dim = config.head_dim
        tp_size = fd_config.parallel_config.tensor_parallel_size
        self.layer_id = layer_id

        # Check if gated attention is enabled
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        num_kv_heads_replicas = max(1, tp_size // self.num_kv_heads)
        self.q_size = self.num_heads * self.head_dim // tp_size
        self.kv_size = self.num_kv_heads * self.head_dim * num_kv_heads_replicas // tp_size

        # For gated attention, use QKVGateParallelLinear
        # qkv_out_size = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_out_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim // tp_size

        if self.attn_output_gate:
            self.qkvg_proj = QKVGateParallelLinear(
                fd_config=fd_config,
                prefix=f"{prefix}.qkvg_proj",
                with_bias=getattr(config, "attention_bias", False),
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                fd_config,
                prefix=f"{prefix}.qkv_proj",
                with_bias=getattr(config, "attention_bias", False),
                num_heads=self.num_heads,
                kv_num_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_heads * self.head_dim,  # output size is num_heads * head_dim (not * 2)
            output_size=config.hidden_size,
            layer_id=layer_id,
        )

        self.attn = Attention(
            fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=True,
        )

        # QKRMSNorm: q_size is for query only (not including gate)
        # Gate is handled separately after attention
        self.qk_norm = QKRMSNorm(
            fd_config,
            head_dim=self.head_dim,
            q_size=self.q_size,
            kv_size=self.kv_size,
            eps=config.rms_norm_eps,
            prefix=prefix,
            begin_norm_axis=2,
        )

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor):
        """Forward pass with gated attention.

        When attn_output_gate=True:
        - qkvg_proj outputs [q, k, v, gate]
        - After attention, output is multiplied by sigmoid(gate)
        """
        if self.attn_output_gate:
            # Gated attention using QKVGateParallelLinear
            qkvg_out = self.qkvg_proj(hidden_states)
            # Split into qkv and gate
            # qkvg_out shape: [num_tokens, q_size + kv_size + kv_size + gate_size]
            # = [num_tokens, (num_heads + 2*num_kv_heads + num_heads) * head_dim / tp_size]
            qkv_out = qkvg_out[:, : self.qkv_out_size].contiguous()
            gate_out = qkvg_out[:, self.qkv_out_size :].contiguous()

            # Apply qk_norm to qkv_out (normalizes Q and K per-head)
            qkv_normalized = self.qk_norm(qkv_out, forward_meta)
            # Attention
            attn_out = self.attn(qkv=qkv_normalized, forward_meta=forward_meta)
            # Apply gate: output = attn_out * sigmoid(gate)
            attn_out = attn_out * paddle.nn.functional.sigmoid(gate_out)
        else:
            # Standard attention without gate
            qkv_out = self.qkv_proj(hidden_states)
            qkv_normalized = self.qk_norm(qkv_out, forward_meta)
            attn_out = self.attn(qkv=qkv_normalized, forward_meta=forward_meta)

        output = self.o_proj(attn_out)
        return output


class Qwen3_5DecoderLayer(nn.Layer):
    """
    Qwen3.5 Decoder Layer with mixed attention types.

    Supports both full_attention and linear_attention (GDN) layers
    based on layer_types configuration.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_type: str = "full_attention",
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_type = layer_type
        layer_id = int(prefix.split(sep=".")[-1])

        # Select attention type based on layer_type
        # Use explicit naming to match checkpoint structure:
        # - full_attention uses "self_attn" prefix
        # - linear_attention uses "linear_attn" prefix (GDN layers)
        if layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
            )
        elif layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        # MLP layer
        self.mlp = Qwen3_5MLP(
            fd_config=fd_config,
            prefix=f"{prefix}.mlp",
        )

        # Layer norms
        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )
        self.layer_id = layer_id

    def load_state_dict(self, state_dict):
        """Load state dict."""
        if self.layer_type == "full_attention":
            self.self_attn.load_state_dict(state_dict)
        elif self.layer_type == "linear_attention":
            self.linear_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        """Forward pass."""
        # Self Attention
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )
        # Select attention layer based on layer_type
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                forward_meta=forward_meta,
                hidden_states=hidden_states,
            )
        elif self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                forward_meta=forward_meta,
                hidden_states=hidden_states,
            )
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_meta)

        return hidden_states, residual


def _get_layer_types(fd_config: FDConfig) -> List[str]:
    """Get layer types for each layer."""
    config = fd_config.model_config

    # Check if layer_types is explicitly set
    if hasattr(config, "layer_types") and config.layer_types:
        return config.layer_types

    # Default: generate based on full_attention_interval
    num_layers = config.num_hidden_layers
    interval = getattr(config, "full_attention_interval", 4)

    layer_types = ["linear_attention" if (i + 1) % interval != 0 else "full_attention" for i in range(num_layers)]
    return layer_types


@support_graph_optimization
class Qwen3_5Model(nn.Layer):
    """Qwen3.5 Model."""

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        # Get layer types
        self.layer_types = _get_layer_types(fd_config)

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Qwen3_5DecoderLayer(
                    fd_config=fd_config,
                    layer_type=self.layer_types[i],
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        # Assign gdn_layer_idx to each GDN layer (for state pool indexing)
        gdn_layer_count = 0
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                self.layers[i].linear_attn.gdn_layer_idx = gdn_layer_count
                gdn_layer_count += 1

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
        """Load model parameters from a given state dictionary."""
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
        """Forward pass."""
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None

        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual)[0]

        return out


@ModelRegistry.register_model_class(
    architecture="Qwen3_5ForConditionalGeneration",
    module_name="qwen3_5",
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Qwen3_5ForCausalLM(ModelForCasualLM):
    """
    Qwen3.5 For Causal Language Model.
    Note: Architecture name is Qwen3_5ForConditionalGeneration for compatibility with HF config.
    """

    def __init__(self, fd_config: FDConfig):
        # Fix for Qwen3.5: The checkpoint uses different head_dim for queries vs output
        # Config: head_dim=256, num_attention_heads=16, hidden_size=1024
        # Checkpoint: q_proj uses 256 (4096 / 16), o_proj uses 128 (2048 / 16)
        # We detect this by checking if (num_attention_heads * head_dim) is 2x (hidden_size * 2)
        model_config = fd_config.model_config
        if hasattr(model_config, "num_attention_heads") and hasattr(model_config, "head_dim"):
            num_heads = model_config.num_attention_heads
            head_dim = model_config.head_dim
            # Check if query projection size is twice the expected output size
            # For Qwen3.5: 16 * 256 = 4096 (queries), but output expects 16 * 128 = 2048
            query_size = num_heads * head_dim
            expected_output_size = model_config.hidden_size * 2  # Typical for models with hidden_size=1024
            if query_size == expected_output_size * 2:
                # Set output_head_dim to half of head_dim for output projections
                model_config.output_head_dim = head_dim // 2
            else:
                model_config.output_head_dim = head_dim

        super(Qwen3_5ForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.model = Qwen3_5Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(self):
        return "Qwen3_5ForConditionalGeneration"

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """Load model parameters from a given weights_iterator object."""

        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        is_pooling_model = hasattr(self, "is_pooling_model") and self.is_pooling_model

        # For Qwen3.5 gated attention:
        # - checkpoint q_proj contains [query, gate] -> use "split_q_gate" to split and load separately
        # - k_proj, v_proj -> load to qkvg_proj with shard_id "k", "v"
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # Gated attention projections (self_attn)
            ("qkvg_proj", "q_proj", "split_q_gate"),
            ("qkvg_proj", "k_proj", "k"),
            ("qkvg_proj", "v_proj", "v"),
            # MLP
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            # Embeddings and LM head
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            # QK norms
            ("qk_norm.q_norm", "q_norm", None),
            ("qk_norm.k_norm", "k_norm", None),
        ]

        params_dict = dict(self.named_parameters())
        model_path = self.fd_config.model_config.model
        revision = self.fd_config.model_config.revision
        if is_pooling_model and get_pooling_config(model_path, revision):
            params_dict = {
                param_name[6:] if param_name.startswith("model.") else param_name: param
                for param_name, param in params_dict.items()
            }

        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")

            # Handle model.language_model prefix for Qwen3.5 models
            # Qwen3.5 checkpoint uses "model.language_model" but we need "model"
            # Replace "language_model." with "" to get "model.xxx"
            if "language_model." in loaded_weight_name:
                loaded_weight_name = loaded_weight_name.replace("language_model.", "")

            # Handle text_config prefix for Qwen3.5 models
            if loaded_weight_name.startswith("text_config."):
                loaded_weight_name = loaded_weight_name[len("text_config.") :]

            # Special handling for GDN components in linear_attn layers
            # Checkpoint uses "conv1d.weight" but model parameter is "conv1d_weight"
            # Also handle A_log and dt_bias (no .weight suffix in either)
            if ".linear_attn.conv1d.weight" in loaded_weight_name:
                loaded_weight_name = loaded_weight_name.replace(".conv1d.weight", ".conv1d_weight")
                # Checkpoint stores conv1d.weight as [conv_dim, 1, kernel_size]
                # Model expects [conv_dim, kernel_size]
                if len(loaded_weight.shape) == 3:
                    loaded_weight = loaded_weight.squeeze(1)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))

                # Qwen3.5 uses a different RMSNorm formula: output = (1.0 + weight) * normalized_x
                # Standard formula: output = weight * normalized_x
                # So we need to convert: FastDeploy weight = 1.0 + checkpoint weight
                # This applies to standard RMSNorm layers in Qwen3.5
                # NOTE: RMSNormGated (linear_attn.norm) uses weight * hidden_states (SAME as FastDeploy)
                # So we do NOT convert linear_attn.norm.weight
                if "linear_attn.norm" not in model_param_name and any(
                    name in model_param_name
                    for name in [
                        "qk_norm.q_norm",
                        "qk_norm.k_norm",
                        "input_layernorm",
                        "post_attention_layernorm",
                        "model.norm",
                    ]
                ):
                    loaded_weight = 1.0 + loaded_weight
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                # Qwen3.5 uses a different RMSNorm formula: output = (1.0 + weight) * normalized_x
                # Standard formula: output = weight * normalized_x
                # So we need to convert: FastDeploy weight = 1.0 + checkpoint weight
                # This applies to standard RMSNorm layers in Qwen3.5
                # NOTE: RMSNormGated (linear_attn.norm) uses weight * hidden_states (SAME as FastDeploy)
                # So we do NOT convert linear_attn.norm.weight
                if "linear_attn.norm" not in model_param_name and any(
                    name in model_param_name
                    for name in [
                        "qk_norm.q_norm",
                        "qk_norm.k_norm",
                        "input_layernorm",
                        "post_attention_layernorm",
                        "model.norm",
                    ]
                ):
                    loaded_weight = 1.0 + loaded_weight
                weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings and not is_pooling_model:
            self.lm_head.linear.weight.set_value(
                self.model.embed_tokens.embeddings.weight.transpose([1, 0]).astype(self.lm_head.linear.weight.dtype)
            )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """Load model parameters from a given state dictionary."""
        self.model.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.model.embed_tokens.embeddings.weight})
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        """Compute logits."""
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def forward(
        self,
        inputs: Dict,
        forward_meta: ForwardMeta,
    ):
        """Forward pass."""
        ids_remove_padding = inputs["ids_remove_padding"]
        hidden_states = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend."""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class Qwen3_5PretrainedModel(PretrainedModel):
    """Qwen3.5 Pretrained Model."""

    config_class = FDConfig

    def _init_weight(self, layer):
        return None

    @classmethod
    def arch_name(self):
        return "Qwen3_5ForConditionalGeneration"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

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
                # Row Linear
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
            }

            # Column Linear
            base_actions["layers.0.self_attn.q_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_proj.bias"] = partial(fn, is_column=True)
            if config.num_key_value_heads % config.tensor_model_parallel_size == 0:
                base_actions["layers.0.self_attn.k_proj.weight"] = partial(fn, is_column=True)
                base_actions["layers.0.self_attn.v_proj.weight"] = partial(fn, is_column=True)

            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)

            # GDN projections for linear_attn layers
            base_actions["layers.0.linear_attn.in_proj_qkv.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.linear_attn.in_proj_z.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.linear_attn.out_proj.weight"] = partial(fn, is_column=False)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings
