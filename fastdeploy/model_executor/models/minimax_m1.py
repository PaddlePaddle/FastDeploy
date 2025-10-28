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
)
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)

import pprint


def print_tensor_stats(tensor, name):
    """Print statistics of Paddle tensors"""

    if tensor is None:
        logger.info(f"DEBUG_FD: {name} is None")
        return
    with paddle.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        num_elements = tensor.numel()
        if num_elements > 0:
            tensor_float = tensor.astype('float32')
            tensor_cpu = tensor_float.cpu()
            stats["max"] = f"{tensor_cpu.max().item():.6f}"
            stats["min"] = f"{tensor_cpu.min().item():.6f}"
            stats["mean"] = f"{tensor_cpu.mean().item():.6f}"

            # Calculate std only if the number of elements is greater than 1
            if num_elements > 1:
                stats["std"] = f"{tensor_cpu.std().item():.6f}"
            else:
                stats["std"] = "0.000000" 
            flat_data = tensor_cpu.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        logger.info(f"\n--- [FD DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")

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
        model_dtype = fd_config.model_config.dtype
        self.weight = self.create_parameter(
            shape=[shard_size], default_initializer=nn.initializer.Constant(1.0), dtype=model_dtype
        )
        # Attach the instance method shard_weight_loader to the weight parameter.
        self.weight.weight_loader = self.shard_weight_loader

    
    def forward(self, x):
        output_dtype = self.weight.dtype
        x_float = x.astype("float32")
        variance = x_float.pow(2).mean(axis=-1, keepdim=True)
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(variance)
            variance = variance / self.tp_size
        inv_std = paddle.rsqrt(variance + self.eps)
        norm_out_f32 = x_float * inv_std
        norm_out = norm_out_f32.astype(output_dtype) * self.weight
        return norm_out
    
    
    def shard_weight_loader(self, param, loaded_weight, *args, **kwargs):
        """Custom loader to shard the full weight."""
        full_weight = get_tensor(loaded_weight)
        shard_size = full_weight.shape[0] // self.tp_size
        my_shard = full_weight[self.tp_rank * shard_size : (self.tp_rank + 1) * shard_size]
        param.set_value(my_shard.cast(param.dtype))


class MiniMaxM1LinearAttention(nn.Layer):
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
            with_bias=False
        )
        self.output_gate = ColumnParallelLinear(
            fd_config, 
            prefix=f"{prefix}.output_gate", 
            input_size=config.hidden_size, 
            output_size=hidden_inner_size, 
            with_bias=False
        )
        self.out_proj = RowParallelLinear(
            fd_config, 
            prefix=f"{prefix}.out_proj", 
            input_size=hidden_inner_size, 
            output_size=config.hidden_size, 
            with_bias=False
        )
        self.norm = RMSNormTP(
            fd_config, 
            hidden_size=hidden_inner_size, 
            prefix=f"{prefix}.norm", 
            eps=1e-5
        )
        
        slope_rate = self._build_slope_tensor(self.num_heads)
        if config.num_hidden_layers > 1:
            self.slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        else:
            self.slope_rate = slope_rate * (1 + 1e-5)
        self.tp_slope = self.slope_rate[tp_rank * self.tp_heads : (tp_rank + 1) * self.tp_heads].contiguous()
        
        def standard_qkv_slicing_loader(param, loaded_weight, *args, **kwargs):
            logger.warning(f"--- [FD DEBUG] Using SIMPLIFIED AND CORRECTED Loader for L{self.layer_id} ---")
            
            tp_rank = self.qkv_proj.fd_config.parallel_config.tensor_parallel_rank
            tp_size = self.qkv_proj.nranks
            full_weight_torch_layout = get_tensor(loaded_weight)

            # 1. Calculate the shard size for each rank
            output_size_per_partition = full_weight_torch_layout.shape[0] // tp_size

            # 2. Calculate the starting and ending positions of the current rank split
            start_row = tp_rank * output_size_per_partition
            end_row = (tp_rank + 1) * output_size_per_partition

            # 3. Directly split the current rank part from the complete weight
            my_shard_torch_layout = full_weight_torch_layout[start_row:end_row, :]
            
            # 4. Transpose to match Paddle's layout [in, out]
            my_shard_paddle_layout = my_shard_torch_layout.transpose([1, 0])

            # 5. Verify and set the value
            assert my_shard_paddle_layout.shape == param.shape, \
                f"Shape mismatch! Final shard shape {my_shard_paddle_layout.shape} vs param shape {param.shape}"
            param.set_value(my_shard_paddle_layout)

        self.qkv_proj.weight.weight_loader = standard_qkv_slicing_loader
        

    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]
        if math.log2(n_attention_heads).is_integer():
            slopes = get_slopes_power_of_2(n_attention_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_attention_heads))
            slopes = (get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_attention_heads - closest_power_of_2])
        return paddle.to_tensor(slopes, dtype='float32').reshape([n_attention_heads, 1, 1])
    

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        layer_id = self.layer_id
        is_profiling_or_warmup = forward_meta.step_use_cudagraph

        # ----- Profiling / CUDAGraph Capture Path -----
        if is_profiling_or_warmup:
            model_dtype = self.out_proj.weight.dtype 

            # 1. Simulate gate calculation
            gate_dummy = self.output_gate(hidden_states)
            
            # 2. Create a zero tensor of the correct shape
            attention_output_dummy = gate_dummy * 0

            # 3. Simulate subsequent operation procedures
            norm_output_dummy = self.norm(attention_output_dummy)
            gated_output_dummy = F.sigmoid(gate_dummy) * norm_output_dummy
            
            gated_output_dummy = gated_output_dummy.cast(model_dtype)
            final_output_dummy = self.out_proj(gated_output_dummy)
            
            return final_output_dummy


        model_dtype = self.out_proj.weight.dtype
        total_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        
        qkv_float32 = qkv.astype("float32")
        qkv_act = F.silu(qkv_float32)

        qkv_reshaped = qkv_act.reshape((total_tokens, self.tp_heads, 3 * self.head_dim))
        q, k, v = qkv_reshaped.split([self.head_dim, self.head_dim, self.head_dim], axis=-1)

        prefill_token_num = int(paddle.sum(forward_meta.seq_lens_encoder).item())
        decode_token_num = total_tokens - prefill_token_num
        has_prefill = prefill_token_num > 0
        has_decode = decode_token_num > 0

        output_prefill = None
        output_decode = None

        if has_prefill:
            q_prefill, k_prefill, v_prefill = q[:prefill_token_num], k[:prefill_token_num], v[:prefill_token_num]
            
            q_attn = q_prefill.transpose((1, 0, 2)).unsqueeze(0)
            k_attn = k_prefill.transpose((1, 0, 2)).unsqueeze(0)
            v_attn = v_prefill.transpose((1, 0, 2)).unsqueeze(0)
            
            # During real reasoning, the first prefilled kv_history is None
            state_cache_for_prefill = None

            output_prefill, updated_state_cache = lightning_attention(
                q_attn, k_attn, v_attn, self.tp_slope, 
                kv_history=state_cache_for_prefill,
            )
            
            # When actually inferring, it is necessary to write back to the cache
            if forward_meta.linear_attn_caches is not None:
                slot_indices = forward_meta.slot_mapping[:prefill_token_num].unique()
                if len(slot_indices) > 0:
                    current_slot = slot_indices[0].item()
                    forward_meta.linear_attn_caches[current_slot, layer_id, :, :, :] = updated_state_cache.squeeze(0)

            output_prefill = output_prefill.squeeze(0).transpose((1, 0, 2)).reshape((prefill_token_num, -1))

        if has_decode:
            q_decode, k_decode, v_decode = q[prefill_token_num:], k[prefill_token_num:], v[prefill_token_num:]
            
            q_decode = q_decode.unsqueeze(2)
            k_decode = k_decode.unsqueeze(2)
            v_decode = v_decode.unsqueeze(2)

            state_cache = forward_meta.linear_attn_caches[:, layer_id, :, :, :]
            slot_mapping_decode = forward_meta.slot_mapping[prefill_token_num:]
            
            output_decode = linear_decode_forward_triton(q_decode, k_decode, v_decode, state_cache, self.tp_slope, slot_mapping_decode)

        if output_prefill is not None and output_decode is not None:
            output = paddle.concat([output_prefill, output_decode], axis=0)
        elif output_prefill is not None:
            output = output_prefill
        elif output_decode is not None:
            output = output_decode
        else:
            return paddle.zeros_like(hidden_states)
        
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
        
        # GQA
        if self.attn_type == 1:  
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
         # Linear Attention
        elif self.attn_type == 0: 
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
        """Forward pass for the decoder layer with FORCED full debugging."""
        is_profile_run = forward_meta.step_use_cudagraph
        layer_id = self.original_layer_id
        
        # We print logs for ALL layers in INFERENCE mode
        if is_profile_run:
            # For profiling runs, just pass through to avoid log spam and errors
            if self.attn_type == 1:
                qkv_out = self.qkv_proj(self.input_layernorm(hidden_states))
                attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
                attn_output = self.o_proj(attn_output)
            else:
                attn_output = self.self_attn(self.input_layernorm(hidden_states), forward_meta)
            # A simplified path for profiling, may not be numerically identical but avoids errors
            hidden_states = hidden_states + attn_output 
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states, None


        layernorm_output = self.input_layernorm(hidden_states)

        residual_attn = layernorm_output if self.postnorm else hidden_states

        attn_output = None
        # GQA
        if self.attn_type == 1:  
            qkv_out = self.qkv_proj(layernorm_output)
            print_tensor_stats(qkv_out, f"FD_L{layer_id}:1b_After_QKV_Proj_Combined")

            q_size_tp = self.self_attn.num_heads * self.self_attn.head_dim
            k_size_tp = self.self_attn.kv_num_heads * self.self_attn.head_dim

            q_before_rope, k_before_rope, v_tensor = qkv_out.split([q_size_tp, k_size_tp, k_size_tp], axis=-1)
            print_tensor_stats(q_before_rope, f"FD_L{layer_id}:1c_Q_BeforeRoPE")
            print_tensor_stats(k_before_rope, f"FD_L{layer_id}:1d_K_BeforeRoPE")
            print_tensor_stats(v_tensor,      f"FD_L{layer_id}:1e_V_Tensor")
            
            
            attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
            attn_output = self.o_proj(attn_output)
        # Linear Attention
        else:  
            attn_output = self.self_attn(layernorm_output, forward_meta)

        hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
        layernorm_output_mlp = self.post_attention_layernorm(hidden_states_after_attn)
        residual_mlp = layernorm_output_mlp if self.postnorm else hidden_states_after_attn
        mlp_output = self.mlp(layernorm_output_mlp)
        
        if self.shared_moe:
            shared_output = self.shared_mlp(layernorm_output_mlp)
            coef_logits = self.coefficient(layernorm_output_mlp.cast("float32"))
            coef = F.sigmoid(coef_logits)
            mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef

        final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        
        return final_output, None

@support_graph_optimization
class MiniMaxM1Model(nn.Layer):
    """
    The core model of MiniMax-M1.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.fd_config = fd_config 
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
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        for i in range(len(self.layers)):
            layer = self.layers[str(i)]
            hidden_states, _ = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=None)

        out = self.norm(hidden_states)
        return out

@ModelRegistry.register_model_class(
    architecture="MiniMaxM1ForCausalLM",
    module_name="minimax_m1",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
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
        return logits.cast("float32")
    
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        import numpy as np
        import re
        from fastdeploy.model_executor.utils import get_tensor, process_weights_after_loading

        params_dict = dict(self.named_parameters())
        sublayers_dict = dict(self.named_sublayers())
        process_weights_after_loading_fn = process_weights_after_loading(sublayers_dict)

        def _get_bfloat16_tensor_from_slice(weight_slice):
            data_bytes = weight_slice[:]
            uint16_array = np.frombuffer(data_bytes, dtype=np.uint16)
            bfloat16_dtype_obj = paddle.to_tensor([0], dtype='bfloat16').numpy().dtype
            bf16_array = uint16_array.view(bfloat16_dtype_obj).reshape(weight_slice.shape)
            return paddle.to_tensor(bf16_array)

        for loaded_weight_name, loaded_weight_slice in weights_iterator:
            current_weight = None
            if "PySafeSlice" in str(type(loaded_weight_slice)):
                dtype_str = str(getattr(loaded_weight_slice, 'dtype', '')).lower()
                if 'bfloat16' in dtype_str or 'bf16' in dtype_str:
                    current_weight = _get_bfloat16_tensor_from_slice(loaded_weight_slice)
            
            if current_weight is None:
                current_weight = get_tensor(loaded_weight_slice)
            
            was_handled = False 

            # Rule 1: MoE Expert Weights
            moe_match = re.search(r"(\.layers\.\d+\.)block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if moe_match:
                prefix_path, expert_id_str, weight_type = moe_match.groups()
                expert_id = int(expert_id_str)
                mlp_prefix = f"model{prefix_path}mlp."
                if weight_type in ["w1", "w3"]:
                    param_name = f"{mlp_prefix}experts.up_gate_proj_weight" 
                    shard_id = "gate" if weight_type == "w1" else "up"
                else: # w2
                    param_name = f"{mlp_prefix}experts.down_proj_weight"
                    shard_id = "down"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if hasattr(param, 'weight_loader'):
                        param.weight_loader(param, current_weight, expert_id=expert_id, shard_id=shard_id)
                        process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                        was_handled = True
                
                if was_handled: continue

            # Rule 2: Attention Weights (GQA or Linear)
            if 'self_attn' in loaded_weight_name:
                layer_match = re.search(r'\.layers\.(\d+)\.', loaded_weight_name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx >= self.config.num_hidden_layers:
                        logger.warning(f"Skipping weight for out-of-bounds layer index {layer_idx}: {loaded_weight_name}")
                        continue
                    
                    if self.config.attn_type_list[layer_idx] == 1: # GQA
                        shard_id = None
                        target_param_name = None
                        if 'q_proj.weight' in loaded_weight_name: 
                            target_param_name = loaded_weight_name.replace('self_attn.q_proj.weight', 'qkv_proj.weight')
                            shard_id = 'q'
                        elif 'k_proj.weight' in loaded_weight_name: 
                            target_param_name = loaded_weight_name.replace('self_attn.k_proj.weight', 'qkv_proj.weight')
                            shard_id = 'k'
                        elif 'v_proj.weight' in loaded_weight_name: 
                            target_param_name = loaded_weight_name.replace('self_attn.v_proj.weight', 'qkv_proj.weight')
                            shard_id = 'v'
                        elif 'o_proj.weight' in loaded_weight_name: 
                            target_param_name = loaded_weight_name.replace('self_attn.o_proj.weight', 'o_proj.weight')

                        if target_param_name and target_param_name in params_dict:
                            param = params_dict[target_param_name]
                            loader = getattr(param, 'weight_loader', None)
                            if loader:
                                loader(param, current_weight, shard_id)
                                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                                was_handled = True

                    elif self.config.attn_type_list[layer_idx] == 0: # Linear Attention
                        target_param_name = loaded_weight_name
                        if target_param_name in params_dict:
                            param = params_dict[target_param_name]
                            loader = getattr(param, 'weight_loader', None)
                            if loader:
                                loader(param, current_weight)
                                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                                was_handled = True

                if was_handled: continue
            
            # Rule 3: Other Common Name Mappings
            param_name = loaded_weight_name
            simple_rename_map = {
                "block_sparse_moe.gate.weight": "mlp.gate.weight",
                "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
                "lm_head.weight": "lm_head.linear.weight",
            }
            # App Rename
            for old, new in simple_rename_map.items():
                if old in param_name:
                    param_name = param_name.replace(old, new)
            
            # Rule 4: Only process simple weights where the name can be directly matched
            if not was_handled and param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                loader(param, current_weight) 
                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                was_handled = True

            if not was_handled:
                logger.warning(f"[LOADER_V2_DEBUG] Weight '{loaded_weight_name}' was NOT handled by any rule (final tried name: {param_name}).")
        
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