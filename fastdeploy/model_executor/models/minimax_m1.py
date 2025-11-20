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
import types

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger
import pprint

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
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.ops.triton_ops.minimax_mamba_ops import (
    lightning_attention,
    linear_decode_forward_triton,
)
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    process_weights_after_loading,
    slice_fn, 
    h2d_copy
)

def print_tensor_stats(tensor, name):
    """Print statistics of Paddle tensors"""
    if tensor is None:
        logger.info(f"DEBUG_FD: {name} is None")
        return
    with paddle.no_grad():
        try:
            if isinstance(tensor, paddle.Tensor):
                t_cpu = tensor.astype('float32').cpu()
            else:
                t_cpu = paddle.to_tensor(tensor).astype('float32').cpu()
            
            stats = {"shape": list(t_cpu.shape), "dtype": str(t_cpu.dtype)}
            num_elements = t_cpu.numel().item()
            if num_elements > 0:
                stats["max"] = f"{t_cpu.max().item():.6f}"
                stats["min"] = f"{t_cpu.min().item():.6f}"
                stats["mean"] = f"{t_cpu.mean().item():.6f}"
                if num_elements > 1:
                    stats["std"] = f"{t_cpu.std().item():.6f}"
                else:
                    stats["std"] = "0.000000"
                flat_data = t_cpu.flatten().numpy()[:5]
                stats["first_5_values"] = flat_data
            logger.info(f"\n--- [FD DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")
        except Exception as e:
            logger.error(f"Error printing stats for {name}: {e}")

class RMSNormTP(nn.Layer):
    def __init__(self, fd_config: FDConfig, hidden_size: int, prefix: str, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.prefix = prefix
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank
        shard_size = hidden_size // self.tp_size
        if hidden_size % self.tp_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by tp_size ({self.tp_size}) for RMSNormTP")
        self.weight = self.create_parameter(
            shape=[shard_size], default_initializer=nn.initializer.Constant(1.0), dtype="float32"
        )
        self.weight.weight_loader = self.shard_weight_loader
    
    def forward(self, x):
        logger.info(f"--- [FD DEBUG] Entering RMSNormTP for '{self.prefix}' ---")
        print_tensor_stats(x, f"RMSNorm_Input_{self.prefix}")
        orig_dtype = x.dtype
        x_float = x.cast("float32")
        variance = x_float.pow(2).mean(axis=-1, keepdim=True)
        print_tensor_stats(variance, f"RMSNorm_Variance_Before_AllReduce_{self.prefix}")
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(variance)
            variance = variance / self.tp_size
            print_tensor_stats(variance, f"RMSNorm_Variance_After_AllReduce_{self.prefix}")
        print_tensor_stats(variance, f"RMSNorm_Variance_{self.prefix}")
        inv_std = paddle.rsqrt(variance + self.eps)
        print_tensor_stats(inv_std, f"RMSNorm_InvStd_{self.prefix}")
        norm_out = (x_float * inv_std).cast(orig_dtype) * self.weight
        logger.info(f"--- [FD DEBUG] Exiting RMSNormTP for '{self.prefix}' ---")
        return norm_out, None 

    def shard_weight_loader(self, param, loaded_weight, *args, **kwargs):
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
        self.qkv_proj = ColumnParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", input_size=config.hidden_size, output_size=hidden_inner_size * 3, with_bias=False)
        self.output_gate = ColumnParallelLinear(fd_config, prefix=f"{prefix}.output_gate", input_size=config.hidden_size, output_size=hidden_inner_size, with_bias=False)
        self.out_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.out_proj", input_size=hidden_inner_size, output_size=config.hidden_size, with_bias=False)
        self.norm = RMSNormTP(fd_config, hidden_size=hidden_inner_size, prefix=f"{prefix}.norm", eps=1e-5)
        slope_rate = self._build_slope_tensor(self.num_heads)
        if config.num_hidden_layers > 1:
            self.slope_rate = slope_rate * (1 - layer_id / (config.num_hidden_layers - 1) + 1e-5)
        else:
            self.slope_rate = slope_rate * (1 + 1e-5)
        self.tp_slope = self.slope_rate[tp_rank * self.tp_heads : (tp_rank + 1) * self.tp_heads].contiguous()

        def standard_qkv_slicing_loader(param, loaded_weight, *args, **kwargs):
            tp_rank = self.qkv_proj.fd_config.parallel_config.tensor_parallel_rank
            tp_size = self.qkv_proj.nranks
            full_weight_torch_layout = get_tensor(loaded_weight)
            output_size_per_partition = full_weight_torch_layout.shape[0] // tp_size
            start_row = tp_rank * output_size_per_partition
            end_row = (tp_rank + 1) * output_size_per_partition
            my_shard_torch_layout = full_weight_torch_layout[start_row:end_row, :]
            assert my_shard_torch_layout.shape == param.shape, f"Shape mismatch! Shard {my_shard_torch_layout.shape} vs param {param.shape}"
            param.set_value(my_shard_torch_layout)
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
        logger.info(f"\n{'='*20} [FD DEBUG] Entering LinearAttention Layer {layer_id} {'='*20}")
        print_tensor_stats(hidden_states, f"L{layer_id}:0_InputHiddenStates")

        is_profiling_or_warmup = forward_meta.step_use_cudagraph
        if is_profiling_or_warmup:
            model_dtype = self.out_proj.weight.dtype 
            gate_dummy = self.output_gate(hidden_states)
            attention_output_dummy = gate_dummy * 0
            norm_output_dummy_tuple = self.norm(attention_output_dummy)
            norm_output_dummy = norm_output_dummy_tuple[0] if isinstance(norm_output_dummy_tuple, tuple) else norm_output_dummy_tuple
            gated_output_dummy = F.sigmoid(gate_dummy) * norm_output_dummy
            gated_output_dummy = gated_output_dummy.cast(model_dtype)
            final_output_dummy = self.out_proj(gated_output_dummy)
            return final_output_dummy

        model_dtype = self.out_proj.weight.dtype
        total_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        print_tensor_stats(qkv, f"L{layer_id}:1_AfterQKVProj")
        
        qkv_float32 = qkv.astype("float32")
        qkv_act = F.silu(qkv_float32)
        print_tensor_stats(qkv_act, f"L{layer_id}:2_AfterSILU")

        qkv_reshaped = qkv_act.reshape((total_tokens, self.tp_heads, 3 * self.head_dim))
        q, k, v = qkv_reshaped.split([self.head_dim, self.head_dim, self.head_dim], axis=-1)
        print_tensor_stats(q, f"L{layer_id}:2a_Split_Q")
        print_tensor_stats(k, f"L{layer_id}:2b_Split_K")
        print_tensor_stats(v, f"L{layer_id}:2c_Split_V")

        prefill_token_num = int(paddle.sum(forward_meta.seq_lens_encoder).item())
        decode_token_num = total_tokens - prefill_token_num
        has_prefill = prefill_token_num > 0
        has_decode = decode_token_num > 0
        logger.info(f"--- [FD DEBUG] L{layer_id} | Total Tokens: {total_tokens}, Prefill: {prefill_token_num}, Decode: {decode_token_num} ---")

        output_prefill, output_decode = None, None
        if has_prefill:
            logger.info(f"--- [FD DEBUG] L{layer_id} | Running PREFILL path ---")
            q_prefill, k_prefill, v_prefill = q[:prefill_token_num], k[:prefill_token_num], v[:prefill_token_num]
            q_attn = q_prefill.transpose((1, 0, 2)).unsqueeze(0)
            k_attn = k_prefill.transpose((1, 0, 2)).unsqueeze(0)
            v_attn = v_prefill.transpose((1, 0, 2)).unsqueeze(0)
            state_cache_for_prefill = None
            output_prefill, updated_state_cache = lightning_attention(q_attn, k_attn, v_attn, self.tp_slope, kv_history=state_cache_for_prefill)
            if forward_meta.linear_attn_caches is not None:
                slot_indices = forward_meta.slot_mapping[:prefill_token_num].unique()
                if len(slot_indices) > 0:
                    current_slot = slot_indices[0].item()
                    forward_meta.linear_attn_caches[current_slot, layer_id, :, :, :] = updated_state_cache.squeeze(0)
            output_prefill = output_prefill.squeeze(0).transpose((1, 0, 2)).reshape((prefill_token_num, -1))

        if has_decode:
            logger.info(f"--- [FD DEBUG] L{layer_id} | Running DECODE path ---")
            q_decode, k_decode, v_decode = q[prefill_token_num:], k[prefill_token_num:], v[prefill_token_num:]
            q_decode, k_decode, v_decode = q_decode.unsqueeze(2), k_decode.unsqueeze(2), v_decode.unsqueeze(2)
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
            logger.warning(f"--- [FD DEBUG] L{layer_id} | Both prefill and decode paths were skipped! Returning zeros. ---")
            return paddle.zeros_like(hidden_states)
        
        print_tensor_stats(output, f"L{layer_id}:3_AfterAttentionKernel")
        norm_output_tuple = self.norm(output)
        norm_output = norm_output_tuple[0] if isinstance(norm_output_tuple, tuple) else norm_output_tuple
        print_tensor_stats(norm_output, f"L{layer_id}:4_AfterRMSNormTP")

        gate = self.output_gate(hidden_states)
        print_tensor_stats(gate, f"L{layer_id}:5_GateValue")
        
        output = F.sigmoid(gate) * norm_output.cast(model_dtype)
        print_tensor_stats(output, f"L{layer_id}:6_AfterGating")

        final_output = self.out_proj(output)
        print_tensor_stats(final_output, f"L{layer_id}:7_FinalOutput")
        logger.info(f"{'='*20} [FD DEBUG] Exiting LinearAttention Layer {layer_id} {'='*20}\n")
        return final_output

class MiniMaxM1MLP(nn.Layer):
    def __init__(self, fd_config: FDConfig, intermediate_size: int, prefix: str = "", reduce_results: bool = True):
        super().__init__()
        config = fd_config.model_config
        self.up_gate_proj = MergedColumnParallelLinear(fd_config, prefix=f"{prefix}.up_gate_proj", input_size=config.hidden_size, output_size=intermediate_size * 2, with_bias=False)
        self.down_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.down_proj", input_size=intermediate_size, output_size=config.hidden_size, with_bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul()
    def forward(self, x):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

class MiniMaxM1MoEBlock(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.gate = ReplicatedLinear(fd_config, prefix=f"{prefix}.gate", input_size=config.hidden_size, output_size=config.num_local_experts, with_bias=False, weight_dtype="float32")
        self.experts = FusedMoE(fd_config, moe_intermediate_size=config.intermediate_size, num_experts=config.num_local_experts, top_k=config.num_experts_per_tok, layer_idx=layer_id)
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.experts(hidden_states, self.gate)

class MiniMaxM1DecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, original_layer_id: int, prefix: str = ""):
        super().__init__()
        config = fd_config.model_config
        self.original_layer_id = original_layer_id
        self.attn_type = config.attn_type_list[original_layer_id]
        attn_prefix = f"{prefix}.self_attn"
        if self.attn_type == 1:
            self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
            self.o_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.o_proj", input_size=config.num_attention_heads * config.head_dim, output_size=config.hidden_size, with_bias=False)
            self.self_attn = Attention(fd_config, layer_id=original_layer_id, prefix=attn_prefix, use_neox_rotary_style=True)
        elif self.attn_type == 0: 
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id=original_layer_id, prefix=attn_prefix)
            self.qkv_proj = None
            self.o_proj = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type} for layer {original_layer_id}")
        self.mlp = MiniMaxM1MoEBlock(fd_config, original_layer_id, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.input_layernorm")
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm")
        self.shared_moe = config.shared_intermediate_size > 0
        if self.shared_moe:
            self.shared_mlp = MiniMaxM1MLP(fd_config, config.shared_intermediate_size, prefix=f"{prefix}.shared_mlp", reduce_results=False)
            self.coefficient = ReplicatedLinear(fd_config, prefix=f"{prefix}.coefficient", input_size=config.hidden_size, output_size=1, with_bias=False, weight_dtype="float32")
        self.postnorm = config.postnorm
        if self.attn_type == 0:
            self.layernorm_attention_alpha = config.layernorm_linear_attention_alpha
            self.layernorm_attention_beta = config.layernorm_linear_attention_beta
        else:
            self.layernorm_attention_alpha = config.layernorm_full_attention_alpha
            self.layernorm_attention_beta = config.layernorm_full_attention_beta
        self.layernorm_mlp_alpha = config.layernorm_mlp_alpha
        self.layernorm_mlp_beta = config.layernorm_mlp_beta
    
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor], run_mode="[UNKNOWN]"):
        layer_id = self.original_layer_id
        SHOULD_LOG = (layer_id == 0 or layer_id == 7)
        if SHOULD_LOG:
            logger.info(f"\n{'='*20} {run_mode} [FD DEBUG] Entering DecoderLayer {layer_id} {'='*20}")
            print_tensor_stats(hidden_states, f"{run_mode} L{layer_id}:0a_Input_HiddenStates")
            print_tensor_stats(residual, f"{run_mode} L{layer_id}:0b_Input_Residual")

        if forward_meta.step_use_cudagraph:
            # 简化 Profile Run
            layernorm_output_tuple = self.input_layernorm(hidden_states)
            layernorm_output = layernorm_output_tuple[0] if isinstance(layernorm_output_tuple, tuple) else layernorm_output_tuple
            if self.attn_type == 1:
                qkv_out = self.qkv_proj(layernorm_output)
                attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
                attn_output = self.o_proj(attn_output)
            else:
                attn_output = self.self_attn(layernorm_output, forward_meta)
            hidden_states = hidden_states + attn_output
            post_ln_output_tuple = self.post_attention_layernorm(hidden_states)
            post_ln_output = post_ln_output_tuple[0] if isinstance(post_ln_output_tuple, tuple) else post_ln_output_tuple
            hidden_states = hidden_states + self.mlp(post_ln_output)
            return hidden_states, None

        # --- 正常推理 ---
        layernorm_output_tuple = self.input_layernorm(hidden_states)
        layernorm_output = layernorm_output_tuple[0] if isinstance(layernorm_output_tuple, tuple) else layernorm_output_tuple
        if SHOULD_LOG: print_tensor_stats(layernorm_output, f"{run_mode} L{layer_id}:1_After_InputLayernorm")
        
        residual_attn = layernorm_output if self.postnorm else hidden_states
        attn_output = None

        if self.attn_type == 1: # GQA
            qkv_out = self.qkv_proj(layernorm_output)
            if SHOULD_LOG:
                print_tensor_stats(qkv_out, f"[NEW][CHK_D] L{layer_id}:1b_After_QKV_Proj_Combined")
            
            # Optional: Print Q/K/V split for GQA
            if SHOULD_LOG:
                q_size_tp = self.self_attn.num_heads * self.self_attn.head_dim
                k_size_tp = self.self_attn.kv_num_heads * self.self_attn.head_dim
                q_before_rope, k_before_rope, v_tensor = qkv_out.split([q_size_tp, k_size_tp, k_size_tp], axis=-1)
                print_tensor_stats(q_before_rope, f"FD_L{layer_id}:1c_Q_BeforeRoPE")
                print_tensor_stats(k_before_rope, f"FD_L{layer_id}:1d_K_BeforeRoPE")
                print_tensor_stats(v_tensor,      f"FD_L{layer_id}:1e_V_Tensor")

            attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
            attn_output = self.o_proj(attn_output)
        else: # Linear
            if SHOULD_LOG:
                print_tensor_stats(layernorm_output, f"[NEW][CHK_D_INPUT] L{layer_id} - Input to LinearAttention")
            attn_output = self.self_attn(layernorm_output, forward_meta)
        
        if SHOULD_LOG: print_tensor_stats(attn_output, f"{run_mode} L{layer_id}:2_After_Attention")

        hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
        if SHOULD_LOG: print_tensor_stats(hidden_states_after_attn, f"{run_mode} L{layer_id}:3_After_Attn_Residual")

        layernorm_output_mlp_tuple = self.post_attention_layernorm(hidden_states_after_attn)
        layernorm_output_mlp = layernorm_output_mlp_tuple[0] if isinstance(layernorm_output_mlp_tuple, tuple) else layernorm_output_mlp_tuple
        if SHOULD_LOG: print_tensor_stats(layernorm_output_mlp, f"{run_mode} L{layer_id}:4_After_PostAttnLayernorm")

        residual_mlp = layernorm_output_mlp if self.postnorm else hidden_states_after_attn
        mlp_output = self.mlp(layernorm_output_mlp)
        if SHOULD_LOG: print_tensor_stats(mlp_output, f"{run_mode} L{layer_id}:5a_After_MoE_MLP")

        if self.shared_moe:
            shared_output = self.shared_mlp(layernorm_output_mlp)
            coef_logits = self.coefficient(layernorm_output_mlp.cast("float32"))
            coef = F.sigmoid(coef_logits)
            mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef
            if SHOULD_LOG: print_tensor_stats(mlp_output, f"{run_mode} L{layer_id}:5b_After_Shared_MLP_Merge")

        final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        if SHOULD_LOG: 
            print_tensor_stats(final_output, f"{run_mode} L{layer_id}:6_FinalOutput")
            logger.info(f"{'='*20} [FD DEBUG] Exiting DecoderLayer {layer_id} {'='*20}\n")

        return final_output, None

class MiniMaxM1Model(nn.Layer):
    def __init__(self, fd_config: FDConfig):
        super().__init__()
        self.fd_config = fd_config 
        self.config = fd_config.model_config
        prefix = "model"
        self.embed_tokens = VocabParallelEmbedding(fd_config, self.config.vocab_size, self.config.hidden_size, prefix=f"{prefix}.embed_tokens")
        layers_to_build = {}
        for i in range(self.config.num_hidden_layers):
            layers_to_build[str(i)] = MiniMaxM1DecoderLayer(fd_config, original_layer_id=i, prefix=f"{prefix}.layers.{i}")
        self.layers = nn.LayerDict(layers_to_build)
        self.norm = RMSNorm(fd_config, self.config.hidden_size, eps=self.config.rms_norm_eps, prefix=f"{prefix}.norm")

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        is_profile_run = forward_meta.step_use_cudagraph
        run_mode = "[PROFILE]" if is_profile_run else "[INFERENCE]"
        if self.fd_config.parallel_config.tensor_parallel_rank == 0:
            print(f"\n{'#'*20} FastDeploy RUN MODE: {run_mode} {'#'*20}\n")
        print_tensor_stats(ids_remove_padding, f"{run_mode} TOP:0_InputIDs")
        
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        print_tensor_stats(hidden_states, f"{run_mode} TOP:1_AfterEmbedding")
        
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[str(i)]
            hidden_states, residual = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=residual, run_mode=run_mode)
        
        out_tuple, residual = self.norm(hidden_states, residual)
        out = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
        print_tensor_stats(out, f"{run_mode} TOP:3_FinalOutput")
        return out, residual

@ModelRegistry.register_model_class(architecture="MiniMaxM1ForCausalLM", module_name="minimax_m1", category=ModelCategory.TEXT_GENERATION)
class MiniMaxM1ForCausalLM(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        self.config = self.fd_config.model_config
        self.config.pretrained_config.prefix_name = "model"
        if hasattr(self.config, "num_local_experts") and not hasattr(self.config, "moe_num_experts"):
            self.config.moe_num_experts = self.config.num_local_experts
        if not hasattr(self.config, "first_k_dense_replace"):
            self.config.first_k_dense_replace = 0
        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(fd_config, embedding_dim=self.config.hidden_size, num_embeddings=self.config.vocab_size, prefix="lm_head")

    @classmethod
    def name(cls):
        return "MiniMaxM1ForCausalLM"

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        model_output = self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        return model_output[0]

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs):
        logits = self.lm_head(hidden_states)
        return logits.cast("float32")
    
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        logger.info("Initializing framework-compliant loader with DEBUG VIEW LOGS")
        params_dict = dict(self.named_parameters())
        
        def _get_bfloat16_tensor_from_slice(weight_slice):
            data_bytes = weight_slice[:]
            uint16_array = np.frombuffer(data_bytes, dtype=np.uint16)
            bfloat16_dtype_obj = paddle.to_tensor([0], dtype='bfloat16').numpy().dtype
            bf16_array = uint16_array.view(bfloat16_dtype_obj).reshape(weight_slice.shape)
            return paddle.to_tensor(bf16_array)

        def print_simulated_old_view(tensor, name, tp_size=4):
            try:
                t_trans = tensor.transpose([1, 0])
                col_size = t_trans.shape[1] // tp_size
                t_shard = t_trans[:, 0:col_size]
                logger.info(f"   >>> [SIMULATED OLD VIEW] {name} (Rank 0 Slice) <<<")
                print_tensor_stats(t_shard, f"Simulated Old View of {name}")
            except Exception as e:
                logger.warning(f"Failed to print simulated view: {e}")

        for loaded_weight_name, loaded_weight_slice in weights_iterator:
            layer_match = re.search(r'\.layers\.(\d+)\.', loaded_weight_name)
            layer_id = int(layer_match.group(1)) if layer_match else -1
            SHOULD_LOG = (layer_id == 0 or layer_id == 7)

            if layer_match and int(layer_match.group(1)) >= self.config.num_hidden_layers: continue

            current_weight = _get_bfloat16_tensor_from_slice(loaded_weight_slice) if "PySafeSlice" in str(type(loaded_weight_slice)) and 'bfloat16' in str(getattr(loaded_weight_slice, 'dtype', '')).lower() else get_tensor(loaded_weight_slice)
            
            # --- Checkpoint A ---
            is_target_weight = ("q_proj.weight" in loaded_weight_name or "self_attn.qkv_proj.weight" in loaded_weight_name)
            if is_target_weight and SHOULD_LOG:
                print_tensor_stats(current_weight, f"[NEW][CHK_A] L{layer_id} - Raw weight from file for '{loaded_weight_name}'")
                print_simulated_old_view(current_weight, loaded_weight_name, self.fd_config.parallel_config.tensor_parallel_size)

            was_handled = False
            moe_match = re.search(r"\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if moe_match:
                layer_idx, expert_id_str, weight_type = moe_match.groups()
                expert_id = int(expert_id_str)
                param_name_tpl, shard_id = (None, None)
                if weight_type in ["w1", "w3"]:
                    param_name_tpl = "model.layers.{}.mlp.experts.up_gate_proj_weight"
                    shard_id = "gate" if weight_type == "w1" else "up"
                else:
                    param_name_tpl = "model.layers.{}.mlp.experts.down_proj_weight"
                    shard_id = "down"
                model_param_name = param_name_tpl.format(layer_idx)
                if model_param_name in params_dict:
                    param = params_dict[model_param_name]
                    loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                    loader(param, current_weight, expert_id=expert_id, shard_id=shard_id)
                    was_handled = True
                if was_handled: continue

            param_name, shard_id = loaded_weight_name, None
            if 'self_attn' in param_name and layer_match:
                if self.config.attn_type_list[layer_id] == 1:
                    if 'q_proj.weight' in param_name: param_name, shard_id = param_name.replace('self_attn.q_proj.weight', 'qkv_proj.weight'), 'q'
                    elif 'k_proj.weight' in param_name: param_name, shard_id = param_name.replace('self_attn.k_proj.weight', 'qkv_proj.weight'), 'k'
                    elif 'v_proj.weight' in param_name: param_name, shard_id = param_name.replace('self_attn.v_proj.weight', 'qkv_proj.weight'), 'v'
                    elif 'o_proj.weight' in param_name: param_name = param_name.replace('self_attn.o_proj.weight', 'o_proj.weight')
            
            simple_rename_map = { "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight", "lm_head.weight": "lm_head.linear.weight", "block_sparse_moe.gate.weight": "mlp.gate.weight" }
            for old, new in simple_rename_map.items():
                if old in param_name: param_name = param_name.replace(old, new)
            
            if param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                if shard_id:
                    loader(param, current_weight, shard_id)
                else:
                    loader(param, current_weight)
        
        logger.warning(f"\n{'!'*20} [NEW][CHK_C] FINAL WEIGHT CHECK (PRE-TRANSPOSE) {'!'*20}")
        try:
            layer0_weight = self.model.layers['0'].self_attn.qkv_proj.weight
            print_tensor_stats(layer0_weight, "  - L0 LinearAttn qkv_proj.weight")
            layer7_weight = self.model.layers['7'].qkv_proj.weight
            print_tensor_stats(layer7_weight, "  - L7 GQA qkv_proj.weight")
        except Exception as e:
            logger.error(f"Failed to access weights for debugging: {e}")
        logger.warning("!"*70 + "\n")

    def set_state_dict(self, state_dict):
        raise NotImplementedError("MiniMax-M1 uses the `load_weights` method.")


class MiniMaxM1PretrainedModel(PretrainedModel):
    config_class = FDConfig
    @classmethod
    def arch_name(cls): return "MiniMaxM1ForCausalLM"
    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        logger.info("Bypassing automatic tensor parallel mappings for MiniMax-M1.")
        return {}