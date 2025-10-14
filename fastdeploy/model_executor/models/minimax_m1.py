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

from __future__ import annotations

import math
import re
from typing import Optional

import paddle
import paddle.nn.functional as F
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
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
    lightning_attention,
    linear_decode_forward_triton,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
)


class RMSNormTP(nn.Layer):
    """
    RMSNorm with Tensor Parallel support.
    """

    def __init__(self, fd_config: FDConfig, hidden_size: int, prefix: str, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.prefix = prefix
        self.weight_key = f"{prefix}.weight"
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank
        shard_size = hidden_size // self.tp_size
        if hidden_size % self.tp_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by tp_size ({self.tp_size}) for RMSNormTP")

        self.weight = self.create_parameter(
            shape=[shard_size], default_initializer=nn.initializer.Constant(1.0), dtype="float32"
        )
        # Attach the instance method shard_weight_loader to the weight parameter.
        self.weight.weight_loader = self.shard_weight_loader

    def forward(self, x):
        """Forward pass for RMSNormTP."""
        orig_dtype = x.dtype
        x_float = x.cast("float32")
        variance = x_float.pow(2).mean(axis=-1, keepdim=True)
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(variance)
            variance = variance / self.tp_size
        inv_std = paddle.rsqrt(variance + self.eps)
        norm_out = (x_float * inv_std).cast(orig_dtype) * self.weight
        return norm_out

    def shard_weight_loader(self, param, loaded_weight):
        """Custom loader to shard the full weight."""
        full_weight = get_tensor(loaded_weight)
        shard_size = full_weight.shape[0] // self.tp_size
        my_shard = full_weight[self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size]
        param.set_value(my_shard.cast(param.dtype))


class MiniMaxM1LinearAttention(nn.Layer):
    """
    Linear Attention module for MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.layer_id = layer_id

        tp_size = fd_config.parallel_config.tensor_parallel_size
        tp_rank = fd_config.parallel_config.tensor_parallel_rank

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.tp_heads = self.num_heads // tp_size

        hidden_inner_size = self.head_dim * self.num_heads

        self.qkv_proj = ColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.qkv_proj",
            input_size=config.hidden_size,
            output_size=hidden_inner_size * 3,
            with_bias=False,
        )
        self.output_gate = ColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.output_gate",
            input_size=config.hidden_size,
            output_size=hidden_inner_size,
            with_bias=False,
        )
        self.out_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.out_proj",
            input_size=hidden_inner_size,
            output_size=config.hidden_size,
            with_bias=False,
        )

        # Use our newly defined RMSNormTP
        self.norm = RMSNormTP(
            fd_config, hidden_size=hidden_inner_size, prefix=f"{prefix}.norm", eps=1e-5
        )

        slope_rate = self._build_slope_tensor(self.num_heads)
        if config.num_hidden_layers > 1:
            self.slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        else:
            self.slope_rate = slope_rate * (1 + 1e-5)

        self.tp_slope = self.slope_rate[tp_rank * self.tp_heads : (tp_rank + 1) * self.tp_heads].contiguous()

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        """Builds the slope tensor for linear attention."""

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n_attention_heads).is_integer():
            slopes = get_slopes_power_of_2(n_attention_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_attention_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : n_attention_heads - closest_power_of_2
            ]

        return paddle.to_tensor(slopes, dtype="float32").reshape([n_attention_heads, 1, 1])

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        """Forward pass for Linear Attention."""
        model_dtype = self.out_proj.weight.dtype
        total_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        qkv_act = F.silu(qkv)

        q, k, v = qkv_act.split(3, axis=-1)

        # Reshape for attention computation
        q = q.reshape((total_tokens, self.tp_heads, self.head_dim))
        k = k.reshape((total_tokens, self.tp_heads, self.head_dim))
        v = v.reshape((total_tokens, self.tp_heads, self.head_dim))

        if forward_meta.forward_mode.is_prefill():
            q = q.transpose((1, 0, 2)).unsqueeze(0)
            k = k.transpose((1, 0, 2)).unsqueeze(0)
            v = v.transpose((1, 0, 2)).unsqueeze(0)

            state_cache = forward_meta.linear_attn_caches[:, self.layer_id, :, :, :]
            output, updated_state_cache = lightning_attention(q, k, v, self.tp_slope, kv_history=state_cache)
            forward_meta.linear_attn_caches[:, self.layer_id, :, :, :] = updated_state_cache
            output = output.squeeze(0).transpose((1, 0, 2)).reshape((total_tokens, -1))

        else:  # decode
            q = q.unsqueeze(2)
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)

            state_cache = forward_meta.linear_attn_caches[:, self.layer_id, :, :, :]
            slot_mapping = forward_meta.slot_mapping
            output = linear_decode_forward_triton(q, k, v, state_cache, self.tp_slope, slot_mapping)

        output = self.norm(output)

        gate = self.output_gate(hidden_states)
        output = F.sigmoid(gate) * output.cast(model_dtype)

        final_output = self.out_proj(output)

        return final_output


class MiniMaxM1MLP(nn.Layer):
    """
    MLP module for MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(
            fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
        )
        self.down_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=intermediate_size,
            output_size=config.hidden_size,
            with_bias=False,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """Forward pass for MLP."""
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class MiniMaxM1MoEBlock(nn.Layer):
    """
    Mixture of Experts block for MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(
            fd_config,
            prefix=f"{prefix}.gate",
            input_size=config.hidden_size,
            output_size=config.num_local_experts,
            with_bias=False,
            weight_dtype="float32",
        )
        self.experts = FusedMoE(
            fd_config,
            moe_intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            layer_idx=layer_id,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Forward pass for MoE block."""
        return self.experts(hidden_states, self.gate)


class MiniMaxM1DecoderLayer(nn.Layer):
    """
    Decoder layer for MiniMax-M1, supporting both GQA and Linear Attention.
    """

    def __init__(self, fd_config: FDConfig, original_layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.original_layer_id = original_layer_id
        self.attn_type = config.attn_type_list[original_layer_id]

        attn_prefix = f"{prefix}.self_attn"

        if self.attn_type == 1:  # GQA
            logger.info(f"Initializing DecoderLayer with prefix '{prefix}' (original layer {original_layer_id}) as GQA type.")
            self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
            self.o_proj = RowParallelLinear(
                fd_config,
                prefix=f"{prefix}.o_proj",
                input_size=config.num_attention_heads * config.head_dim,
                output_size=config.hidden_size,
                with_bias=False,
            )
            self.self_attn = Attention(
                fd_config, layer_id=original_layer_id, prefix=attn_prefix, use_neox_rotary_style=True
            )
        elif self.attn_type == 0:  # Linear Attention
            logger.info(
                f"Initializing DecoderLayer with prefix '{prefix}' (original layer {original_layer_id}) as Linear Attention type."
            )
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id=original_layer_id, prefix=attn_prefix)
            self.qkv_proj = None
            self.o_proj = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type} for layer {original_layer_id}")

        self.mlp = MiniMaxM1MoEBlock(fd_config, original_layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(
            fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm"
        )
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
        )

        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(
                fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp", reduce_results=False
            )
            self.coefficient = ReplicatedLinear(
                fd_config,
                prefix=f"{prefix}.coefficient",
                input_size=config.hidden_size,
                output_size=1,
                with_bias=False,
                weight_dtype="float32",
            )

        # Get alpha and beta scaling factors
        self.postnorm = config.postnorm
        if self.attn_type == 0:
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta

    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor]):
        """Forward pass for the decoder layer."""
        layernorm_output = self.input_layernorm(hidden_states)
        residual_attn = layernorm_output if self.postnorm else hidden_states
        if self.attn_type == 1:  # GQA
            qkv_out = self.qkv_proj(layernorm_output)
            attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
            attn_output = self.o_proj(attn_output)
        else:  # Linear Attention
            attn_output = self.self_attn(layernorm_output, forward_meta)

        hidden_states = (residual_attn * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)

        # MLP Block
        layernorm_output_mlp = self.post_attention_layernorm(hidden_states)
        residual_mlp = layernorm_output_mlp if self.postnorm else hidden_states

        mlp_output = self.mlp(layernorm_output_mlp)

        if self.shared_moe:
            shared_output = self.shared_mlp(layernorm_output_mlp)
            coef_logits = self.coefficient(layernorm_output_mlp.cast("float32"))
            coef = F.sigmoid(coef_logits)
            mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef

        # Final alpha/beta scaling and residual connection
        final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)

        return final_output, None


@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    """
    The core model of MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(
            fd_config, self.config.vocab_size, self.config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )

        # Use nn.LayerDict to build layers, with original layer index as key
        layers_to_build = {}
        for i in range(self.config.num_hidden_layers):
            layer_prefix = f"{prefix}.layers.{i}"
            layers_to_build[str(i)] = MiniMaxM1DecoderLayer(fd_config, original_layer_id=i, prefix=layer_prefix)

        self.layers = nn.LayerDict(layers_to_build)
        self.norm = RMSNorm(
            fd_config,
            self.config.hidden_size,  # positional arg
            eps=self.config.rms_norm_eps,
            prefix=f"{prefix}.norm",
        )

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        """Forward pass for the model."""
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        # Simplified loop, no residual handling
        for i in range(len(self.layers)):
            layer = self.layers[str(i)]
            hidden_states, _ = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=None)
        out = self.norm(hidden_states)
        return out


class MiniMaxM1ForCausalLM(ModelForCasualLM):
    """
    Causal LM model for MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        # Save the model config as a self.config attribute
        self.config = self.fd_config.model_config
        self.config.pretrained_config.prefix_name = "model"
        if hasattr(self.config, "num_local_experts") and not hasattr(self.config, "moe_num_experts"):
            self.config.moe_num_experts = self.config.num_local_experts
        if (
            hasattr(self.config, "rotary_dim")
            and hasattr(self.config, "head_dim")
            and self.config.rotary_dim < self.config.head_dim
        ):
            self.config.partial_rotary_factor = self.config.rotary_dim / self.config.head_dim
        if not hasattr(self.config, "first_k_dense_replace"):
            self.config.first_k_dense_replace = 0

        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=self.config.hidden_size,
            num_embeddings=self.config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        logits = self.lm_head(hidden_states)
        # Cast logits to float32 to ensure compatibility with sampling operators
        return logits.cast("float32")

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """Loads model weights with custom mapping logic for MiniMax-M1."""
        logger.info("Initializing robust multi-GPU weight loader for MiniMax-M1...")

        params_dict = dict(self.named_parameters())
        sublayers_dict = dict(self.named_sublayers())
        process_weights_after_loading_fn = process_weights_after_loading(sublayers_dict)

        loaded_checkpoint_keys = set()

        for loaded_weight_name, loaded_weight in weights_iterator:
            layer_match = re.search(r"\.layers\.(\d+)\.", loaded_weight_name)
            if layer_match and int(layer_match.group(1)) >= self.config.num_hidden_layers:
                continue

            param_to_load = None

            # Rule 1: MoE Expert Weights
            moe_match = re.search(
                r"(\.layers\.\d+\.)block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name
            )
            if moe_match:
                prefix_path, expert_id_str, weight_type = moe_match.groups()
                expert_id = int(expert_id_str)

                mlp_prefix = f"model{prefix_path.replace('block_sparse_moe', 'mlp')}"
                if weight_type in ["w1", "w3"]:
                    param_name = f"{mlp_prefix}mlp.experts.up_gate_proj_weight"
                    shard_id = "gate" if weight_type == "w1" else "up"
                else:  # w2
                    param_name = f"{mlp_prefix}mlp.experts.down_proj_weight"
                    shard_id = "down"

                if param_name in params_dict:
                    param = params_dict[param_name]
                    param.weight_loader(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
                    param_to_load = param

                loaded_checkpoint_keys.add(loaded_weight_name)
                continue

            # Rule 2: GQA Attention Weights
            if "self_attn" in loaded_weight_name:
                layer_idx = int(re.search(r"\.layers\.(\d+)\.", loaded_weight_name).group(1))
                if self.config.attn_type_list[layer_idx] == 1:  # Is GQA
                    param_name = loaded_weight_name
                    shard_id = None
                    if "q_proj.weight" in loaded_weight_name:
                        param_name = loaded_weight_name.replace("self_attn.q_proj.weight", "qkv_proj.weight")
                        shard_id = "q"
                    elif "k_proj.weight" in loaded_weight_name:
                        param_name = loaded_weight_name.replace("self_attn.k_proj.weight", "qkv_proj.weight")
                        shard_id = "k"
                    elif "v_proj.weight" in loaded_weight_name:
                        param_name = loaded_weight_name.replace("self_attn.v_proj.weight", "qkv_proj.weight")
                        shard_id = "v"
                    elif "o_proj.weight" in loaded_weight_name:
                        param_name = loaded_weight_name.replace("self_attn.o_proj.weight", "o_proj.weight")

                    if param_name in params_dict:
                        param = params_dict[param_name]
                        loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                        loader(param, loaded_weight, shard_id)
                        param_to_load = param

                    loaded_checkpoint_keys.add(loaded_weight_name)
                    continue

            # Rule 3: General name mapping and default loading
            param_name = loaded_weight_name
            simple_rename_map = {
                "block_sparse_moe.gate.weight": "mlp.gate.weight",
                "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
                "lm_head.weight": "lm_head.linear.weight",
            }
            for old, new in simple_rename_map.items():
                if old in param_name:
                    param_name = param_name.replace(old, new)
                    break

            if param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                loader(param, loaded_weight)
                param_to_load = param
                loaded_checkpoint_keys.add(loaded_weight_name)
            elif loaded_weight_name not in loaded_checkpoint_keys:
                logger.warning(f"Weight '{loaded_weight_name}' was not used (tried name '{param_name}').")

            if param_to_load is not None:
                sublayer_name = param.name.rsplit(".", 1)[0]
                process_weights_after_loading_fn(sublayer_name, param_to_load)

        logger.info("Weight loading process finished.")

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")


class MiniMaxM1PretrainedModel(PretrainedModel):
    """
    Pretrained model class for MiniMax-M1.
    """

    config_class = FDConfig

    @classmethod
    def arch_name(cls):
        return "MiniMaxM1ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.info("Bypassing automatic tensor parallel mappings for MiniMax-M1.")
        return {}