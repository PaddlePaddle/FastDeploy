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
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.utils import (
    WeightsMapper,
    default_weight_loader,
)


@ModelRegistry.register_model_class(
    architecture="MiniMaxM1ForCausalLM",
    module_name="minimax_m1",
    category=[ModelCategory.TEXT_GENERATION],
    primary_use=ModelCategory.TEXT_GENERATION,
)
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """
    MiniMax-M1 model for causal language modeling
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        
        # Parse config
        config = fd_config.model_config
        self.ori_vocab_size = getattr(config, "ori_vocab_size", getattr(config, "vocab_size", None))
        
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
    def load_weights(self, weights_iterator):
        """Load model weights using WeightsMapper"""
        params_dict = dict(self.named_parameters())
        mapper = WeightsMapper()
        
        for loaded_weight_name, loaded_weight in weights_iterator:
            # Standard prefix replacement
            loaded_weight_name = loaded_weight_name.replace("model.minimax_m1", "model")
            
            mapped_name = mapper.map_name(loaded_weight_name)
            if mapped_name in params_dict:
                weight = params_dict[mapped_name]
                default_weight_loader(weight, loaded_weight)
            else:
                logger.warning(f"Weight {loaded_weight_name} not found in model")

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """Forward pass using standard FastDeploy signature"""
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
        )
        return hidden_states

    def compute_logits(self, hidden_states: paddle.Tensor, **logits_processor_kwargs):
        """Compute logits from hidden states"""
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        if hasattr(self, "ori_vocab_size") and getattr(self, "ori_vocab_size") is not None:
            if logits.shape[-1] > self.ori_vocab_size:
                logits[:, self.ori_vocab_size :] = -float("inf")
        return logits


class MiniMaxM1Model(nn.Layer):
    """MiniMax-M1 transformer model"""

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        
        self.config = fd_config.model_config
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
        
        # Build decoder layers
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
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """Forward pass"""
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        
        # Prepare slope rates
        slope_rates = []
        for idx in range(len(self.layers)):
            slope = self.slopes.clone()
            slope = slope * (1 - idx / (len(self.layers) - 1) + 1e-5)
            slope_rates.append(slope)
        
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(
                hidden_states=hidden_states,
                residual=residual,
                forward_meta=forward_meta,
                slope_rate=slope_rates[idx],
            )
        
        hidden_states, _ = self.norm(hidden_states, residual_input=residual, forward_meta=forward_meta)
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
        self.attention_type = attention_type
        
        if attention_type == 0:
            self.self_attn = MiniMaxM1LightningAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id, prefix=f"{prefix}.self_attn")
        
        self.block_sparse_moe = MiniMaxM1SparseMoEBlock(fd_config, layer_id, prefix=f"{prefix}.mlp")
        
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
        
        # Scaling factors for residual connections
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
        
        shared_intermediate = getattr(fd_config.model_config, "shared_intermediate_size", 0)
        self.shared_moe = shared_intermediate > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(
                fd_config, 
                intermediate_size=shared_intermediate,
                prefix=f"{prefix}.shared_experts"
            )
            self.coefficient = ReplicatedLinear(
                fd_config=fd_config,
                prefix=f"{prefix}.coefficient",
                input_size=fd_config.model_config.hidden_size,
                output_size=1,
                with_bias=False,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor | None,
        forward_meta: ForwardMeta,
        slope_rate: paddle.Tensor | None = None,
    ):
        # Self Attention block
        norm_hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )
        
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            forward_meta=forward_meta,
            slope_rate=slope_rate,
        )
        
        # Residual connection with scaling
        hidden_states = residual * self.layernorm_attention_alpha + attn_output * self.layernorm_attention_beta
        residual = None
        
        # MLP / MoE block
        norm_hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )
        
        moe_output, _ = self.block_sparse_moe(norm_hidden_states, forward_meta)
        
        if self.shared_moe:
            shared_output = self.shared_mlp(norm_hidden_states)
            # Sigmoid routing
            coef = paddle.nn.functional.sigmoid(self.coefficient(norm_hidden_states))
            moe_output = moe_output * (1.0 - coef) + shared_output * coef
        
        hidden_states = residual * self.layernorm_mlp_alpha + moe_output * self.layernorm_mlp_beta
        residual = None
        
        return hidden_states, residual


class MiniMaxM1StandardAttention(nn.Layer):
    """Standard attention for MiniMax-M1"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(fd_config=fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )
        self.attn = Attention(fd_config=fd_config, layer_id=layer_id, prefix=prefix, use_neox_rotary_style=True)

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta, slope_rate: paddle.Tensor | None = None):
        qkv_out = self.qkv_proj(hidden_states)
        attn_output = self.attn(qkv=qkv_out, forward_meta=forward_meta)
        return self.o_proj(attn_output)


class MiniMaxM1LightningAttention(nn.Layer):
    """Lightning Attention for MiniMax-M1"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.fd_config = fd_config
        self.num_heads = fd_config.model_config.num_attention_heads // fd_config.parallel_config.tensor_parallel_size
        self.head_dim = fd_config.model_config.head_dim
        
        self.act = getattr(paddle.nn.functional, fd_config.model_config.hidden_act, paddle.nn.functional.silu)
        
        self.qkv_proj = QKVParallelLinear(fd_config=fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.output_gate = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.output_gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=self.num_heads * self.head_dim,
            with_bias=False,
        )
        self.out_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.out_proj",
            input_size=self.num_heads * self.head_dim,
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )
        self.norm = RMSNorm(
            fd_config=fd_config,
            hidden_size=self.num_heads * self.head_dim,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.norm",
        )
        
        self.block_size = 256
        self.kv_states = {} # State per sequence index

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta, slope_rate: paddle.Tensor | None = None):
        qkv = self.act(self.qkv_proj(hidden_states))
        q, k, v = paddle.split(qkv, 3, axis=-1)
        
        q = q.reshape([-1, self.num_heads, self.head_dim])
        k = k.reshape([-1, self.num_heads, self.head_dim])
        v = v.reshape([-1, self.num_heads, self.head_dim])
        
        ratio = paddle.exp(-slope_rate) if slope_rate is not None else paddle.ones([1], dtype="float32")
        
        if not forward_meta.forward_mode.is_decode():
            output = self._forward_prefill(q, k, v, ratio, forward_meta)
        else:
            output = self._forward_decode(q, k, v, ratio, forward_meta)
        
        output = output.reshape([-1, self.num_heads * self.head_dim])
        output = self.norm(output)[0]
        gate = paddle.nn.functional.sigmoid(self.output_gate(hidden_states))
        return self.out_proj(gate * output)

    def _forward_prefill(self, q, k, v, ratio, forward_meta):
        output = paddle.empty_like(q)
        cu_seqlens = forward_meta.cu_seqlens_q
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i+1]
            qi = q[start:end].transpose([1, 0, 2]).unsqueeze(0)
            ki = k[start:end].transpose([1, 0, 2]).unsqueeze(0)
            vi = v[start:end].transpose([1, 0, 2]).unsqueeze(0)
            
            out_i, state_i = self._compute_lightning_attention_core(qi, ki, vi, ratio)
            output[start:end] = out_i.squeeze(0).transpose([1, 0, 2])
            self.kv_states[i] = state_i 
        return output

    def _forward_decode(self, q, k, v, ratio, forward_meta):
        batch_size = q.shape[0]
        output = paddle.empty_like(q)
        for i in range(batch_size):
            state = self.kv_states.get(i, paddle.zeros([self.num_heads, self.head_dim, self.head_dim]))
            qi = q[i].unsqueeze(0)
            ki = k[i].unsqueeze(0)
            vi = v[i].unsqueeze(0)
            state = ratio * state + paddle.matmul(ki.transpose([0, 2, 1]), vi)
            out_i = paddle.matmul(qi, state)
            output[i] = out_i.squeeze(0)
            self.kv_states[i] = state
        return output

    def _compute_lightning_attention_core(self, q, k, v, ratio):
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        block_indices = paddle.arange(self.block_size).astype("float32").reshape([1, 1, -1, 1])
        q_decay = paddle.exp(-ratio * block_indices)
        k_decay = paddle.exp(-ratio * (self.block_size - block_indices))
        indices = paddle.arange(self.block_size).astype("float32")
        mask = indices[:, None] - indices[None, :]
        diag_decay = paddle.exp(paddle.where(mask >= 0, -ratio * mask.unsqueeze(0), paddle.full_like(mask, float("-inf")).unsqueeze(0)))
        kv_state = paddle.zeros([batch_size, num_heads, head_dim, head_dim], dtype="float32")
        output = paddle.empty_like(q, dtype=q.dtype)
        for i in range(num_blocks):
            si, ei = i * self.block_size, min((i + 1) * self.block_size, seq_len)
            m = ei - si
            qi, ki, vi = q[:, :, si:ei], k[:, :, si:ei], v[:, :, si:ei]
            qkv_inter = paddle.matmul(qi * q_decay[:, :, :m], kv_state.astype(qi.dtype)).astype("float32")
            qk = paddle.matmul(qi, ki.transpose([0, 1, 3, 2])).astype("float32") * diag_decay[:, :, :m, :m]
            qkv_intra = paddle.matmul(qk, vi.astype("float32"))
            output[:, :, si:ei] = (qkv_inter + qkv_intra).astype(q.dtype)
            block_decay = paddle.exp(-ratio * m)
            kv_state = block_decay * kv_state + paddle.matmul((ki * k_decay[:, :, -m:]).transpose([0, 1, 3, 2]).astype(vi.dtype), vi)
        return output, kv_state


class MiniMaxM1MLP(nn.Layer):
    """Shared expert MLP"""

    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = ""):
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
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
        )
        self.act_fn = SiluAndMul(fd_config=fd_config, bias=None, act_method=fd_config.model_config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_gate_proj(x)))


class MiniMaxM1SparseMoEBlock(nn.Layer):
    """Sparse MoE Block"""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        self.num_experts = fd_config.model_config.num_local_experts
        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=self.num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )
        weight_key_map = {
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }
        self.experts = FusedMoE(
            fd_config=fd_config,
            reduce_results=True,
            renormalize=False,
            moe_intermediate_size=fd_config.model_config.intermediate_size,
            num_experts=self.num_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method=getattr(fd_config.model_config, "topk_method", "noaux_tc"),
            topk_group=getattr(fd_config.model_config, "topk_group", 1),
            n_group=getattr(fd_config.model_config, "n_group", 1),
            routed_scaling_factor=getattr(fd_config.model_config, "routed_scaling_factor", 1.0),
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, self.gate, forward_meta)
        return final_hidden_states, router_logits


MiniMaxM1ForCausalLM.arch_name = "MiniMax-M1"