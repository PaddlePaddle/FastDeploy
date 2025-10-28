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
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
import pprint
import numpy as np

def print_tensor_stats(tensor, name,  is_capturing=False):
    """打印Paddle张量的统计信息 (强制 float32)"""
    # --- [最终解决方案] ---
    # 如果正在捕获图，则直接返回，不执行任何操作
    if is_capturing:
        return
    # --- [结束解决方案] ---
    
    if tensor is None:
        logger.info(f"DEBUG_FD: {name} is None")
        return
    with paddle.no_grad():
        stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        # ========== [修改这里] ==========
        num_elements = tensor.numel()
        if num_elements > 0:
            tensor_float = tensor.astype('float32')
            tensor_cpu = tensor_float.cpu()
            stats["max"] = f"{tensor_cpu.max().item():.6f}"
            stats["min"] = f"{tensor_cpu.min().item():.6f}"
            stats["mean"] = f"{tensor_cpu.mean().item():.6f}"
            
            # 只有当元素数量大于1时才计算std
            if num_elements > 1:
                stats["std"] = f"{tensor_cpu.std().item():.6f}"
            else:
                stats["std"] = "0.000000" # 单个元素的std为0
        # ========== [结束修改] ==========
            flat_data = tensor_cpu.flatten().numpy()[:5]
            stats["first_5_values"] = flat_data
        logger.info(f"\n--- [FD DEBUG] {name} ---\n{pprint.pformat(stats, indent=2)}\n--------------------------\n")
# ============================================================================


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

    
    def forward(self, x, forward_meta: Optional[ForwardMeta] = None):
        is_capturing = forward_meta.step_use_cudagraph if forward_meta and hasattr(forward_meta, 'step_use_cudagraph') else False
        logger.info(f"--- [FD DEBUG] Entering RMSNormTP for '{self.prefix}' ---")
        print_tensor_stats(x, f"RMSNorm_Input_{self.prefix}", is_capturing=is_capturing)

        # 目标输出类型由 weight 的类型决定
        output_dtype = self.weight.dtype
        
        # 输入 x 不论是什么类型，我们都统一转到 float32 计算
        x_float = x.astype("float32")

        variance = x_float.pow(2).mean(axis=-1, keepdim=True)
        print_tensor_stats(variance, f"RMSNorm_Variance_Before_AllReduce_{self.prefix}", is_capturing=is_capturing)

        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(variance)
            variance = variance / self.tp_size
            print_tensor_stats(variance, f"RMSNorm_Variance_After_AllReduce_{self.prefix}", is_capturing=is_capturing)

        print_tensor_stats(variance, f"RMSNorm_Variance_{self.prefix}", is_capturing=is_capturing)

        inv_std = paddle.rsqrt(variance + self.eps)
        print_tensor_stats(inv_std, f"RMSNorm_InvStd_{self.prefix}", is_capturing=is_capturing)

        # 在 float32 下完成归一化
        norm_out_f32 = x_float * inv_std

        # 先将归一化结果转为目标类型，再乘以同类型的 weight
        norm_out = norm_out_f32.astype(output_dtype) * self.weight

        logger.info(f"--- [FD DEBUG] Exiting RMSNormTP for '{self.prefix}' ---")
        print_tensor_stats(norm_out, f"RMSNorm_Output_{self.prefix}", is_capturing=is_capturing) # 加一个输出打印
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
            full_weight_torch_layout = get_tensor(loaded_weight) # Shape: [out_full, in], e.g., [49152, 6144]

            # 1. 计算每个 rank 的分片大小
            # full_weight_torch_layout.shape[0] 是完整的输出维度
            output_size_per_partition = full_weight_torch_layout.shape[0] // tp_size

            # 2. 计算当前 rank 的切分起始和结束位置
            start_row = tp_rank * output_size_per_partition
            end_row = (tp_rank + 1) * output_size_per_partition

            # 3. 直接从完整权重中切分出当前 rank 的部分 (仍然是 Torch/vLLM 布局)
            my_shard_torch_layout = full_weight_torch_layout[start_row:end_row, :]
            
            # 4. 转置以匹配 Paddle 的布局 [in, out]
            my_shard_paddle_layout = my_shard_torch_layout.transpose([1, 0])

            # 5. 验证并设置值
            assert my_shard_paddle_layout.shape == param.shape, \
                f"Shape mismatch! Final shard shape {my_shard_paddle_layout.shape} vs param shape {param.shape}"
            param.set_value(my_shard_paddle_layout)

            if tp_rank == 0:
                print_tensor_stats(param, f"FD_L{self.layer_id}_QKV_WEIGHT_SHARD_FINAL (PADDLE LAYOUT)")
                        
        self.qkv_proj.weight.weight_loader = standard_qkv_slicing_loader
        # # --- [结束] ---
        

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
    
    # --- [核心修改] ---
    # forward 方法被重构为两条路径：profiling 和 inference
    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        from paddleformers.utils.log import logger
        is_capturing = forward_meta.step_use_cudagraph if forward_meta else False
        if self.layer_id == 0: # 只打印第一层
            print_tensor_stats(self.qkv_proj.weight, "FD_L0_QKV_PROJ_WEIGHT", is_capturing)

        layer_id = self.layer_id
        is_profiling_or_warmup = forward_meta.step_use_cudagraph

        # ----- Profiling / CUDAGraph Capture Path -----
        if is_profiling_or_warmup:
            logger.warning(f"--- [FD DEBUG] L{layer_id} | Bypassing LinearAttention with safe graph operations. ---")

            model_dtype = self.out_proj.weight.dtype # 获取模型期望的数据类型，通常是 bfloat16

            # 1. 模拟 gate 的计算
            gate_dummy = self.output_gate(hidden_states)
            
            # 2. 创建形状正确的零张量
            attention_output_dummy = gate_dummy * 0

            # 3. 模拟后续的操作流程
            norm_output_dummy = self.norm(attention_output_dummy, forward_meta=forward_meta)
            gated_output_dummy = F.sigmoid(gate_dummy) * norm_output_dummy
            
            # --- [核心修复] ---
            # 在进入 out_proj 之前，将 gated_output_dummy 的类型转换回模型默认的 dtype
            gated_output_dummy = gated_output_dummy.cast(model_dtype)
            # --- [结束修复] ---

            final_output_dummy = self.out_proj(gated_output_dummy)
            
            return final_output_dummy

        # ----- Real Inference Path -----
        # (以下是你之前对齐精度的完整代码，基本保持不变)
        logger.info(f"\n{'='*20} [FD DEBUG] Entering LinearAttention Layer {layer_id} {'='*20}")
        print_tensor_stats(hidden_states, f"L{layer_id}:0_InputHiddenStates")

        model_dtype = self.out_proj.weight.dtype
        total_tokens = hidden_states.shape[0]
        
        # ========== [新增调试代码] ==========
        if self.layer_id == 0: # 只打印第一层就够了
            print_tensor_stats(hidden_states, f"FD_MATMUL_INPUT_L{self.layer_id}")
            # 注意：Paddle的Linear层权重是 [in, out]，而 matmul 需要 [batch, in] @ [in, out]，所以不需要转置
            print_tensor_stats(self.qkv_proj.weight, f"FD_MATMUL_WEIGHT_L{self.layer_id}")
        # ========== [结束新增] ==========
        
        # ========== [修改这里的调试代码] ==========
        # 同样使用 flag 防止重复保存
        if self.layer_id == 0 and not hasattr(self, '_weight_dumped'):
            
            # 打印信息，确认正在保存
            if self.qkv_proj.fd_config.parallel_config.tensor_parallel_rank == 0:
                print(f"\n--- [FD DEBUG] Dumping L{self.layer_id} QKV weight for rank 0... ---\n")
            
            # 获取最终加载到GPU上的权重参数
            weight_shard = self.qkv_proj.weight 

            # 保存到文件
            # 注意：Paddle的权重布局是 [input_features, output_features]
            if self.qkv_proj.fd_config.parallel_config.tensor_parallel_rank == 0:
                paddle.save(weight_shard.cpu().astype("float32"), "fd_qkv_weight_shard_rank0.pdparams")
                print(f"\n--- [FD DEBUG] Saved rank 0 weight shard to fd_qkv_weight_shard_rank0.pdparams ---\n")

            # 设置 flag
            # 在分布式环境中，需要确保所有rank都有这个属性，即使它们不保存文件
            self._weight_dumped = True
        # ========== [结束修改] ==========

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

        # 这里的 prefill/decode 判断逻辑保持不变
        prefill_token_num = int(paddle.sum(forward_meta.seq_lens_encoder).item())
        decode_token_num = total_tokens - prefill_token_num
        has_prefill = prefill_token_num > 0
        has_decode = decode_token_num > 0
        logger.info(f"--- [FD DEBUG] L{layer_id} | Total Tokens: {total_tokens}, Prefill: {prefill_token_num}, Decode: {decode_token_num} ---")

        output_prefill = None
        output_decode = None

        if has_prefill:
            logger.info(f"--- [FD DEBUG] L{layer_id} | Running PREFILL path ---")
            q_prefill, k_prefill, v_prefill = q[:prefill_token_num], k[:prefill_token_num], v[:prefill_token_num]
            
            q_attn = q_prefill.transpose((1, 0, 2)).unsqueeze(0)
            k_attn = k_prefill.transpose((1, 0, 2)).unsqueeze(0)
            v_attn = v_prefill.transpose((1, 0, 2)).unsqueeze(0)
            
            # 真实推理时，第一次 prefill 的 kv_history 就是 None
            state_cache_for_prefill = None

            output_prefill, updated_state_cache = lightning_attention(
                q_attn, k_attn, v_attn, self.tp_slope, 
                kv_history=state_cache_for_prefill,
            )
            
            # 真实推理时，需要写回 cache
            if forward_meta.linear_attn_caches is not None:
                slot_indices = forward_meta.slot_mapping[:prefill_token_num].unique()
                if len(slot_indices) > 0:
                    current_slot = slot_indices[0].item()
                    forward_meta.linear_attn_caches[current_slot, layer_id, :, :, :] = updated_state_cache.squeeze(0)

            output_prefill = output_prefill.squeeze(0).transpose((1, 0, 2)).reshape((prefill_token_num, -1))

        if has_decode:
            logger.info(f"--- [FD DEBUG] L{layer_id} | Running DECODE path ---")
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
            logger.warning(f"--- [FD DEBUG] L{layer_id} | Both prefill and decode paths were skipped! Returning zeros. ---")
            return paddle.zeros_like(hidden_states) # 这种情况下返回零是安全的

        print_tensor_stats(output, f"L{layer_id}:3_AfterAttentionKernel")
        
        output = self.norm(output, forward_meta=forward_meta)
        print_tensor_stats(output, f"L{layer_id}:4_AfterRMSNormTP")
        
        gate = self.output_gate(hidden_states)
        print_tensor_stats(gate, f"L{layer_id}:5_GateValue")
        
        output = F.sigmoid(gate) * output.cast(model_dtype)
        print_tensor_stats(output, f"L{layer_id}:6_AfterGating")
        
        final_output = self.out_proj(output)
        print_tensor_stats(final_output, f"L{layer_id}:7_FinalOutput")
        
        logger.info(f"{'='*20} [FD DEBUG] Exiting LinearAttention Layer {layer_id} {'='*20}\n")
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
        

    # def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor], run_mode: str = "[UNKNOWN]"):
    #     """Forward pass for the decoder layer."""
    #     is_profile_run = forward_meta.step_use_cudagraph
    #     layer_id = self.original_layer_id
    #     # ========== [开始新的 GQA 权重 Dump 代码] ==========
    #     if self.attn_type == 1 and not hasattr(self, '_gqa_weight_dumped'):
            
    #         if self.qkv_proj.fd_config.parallel_config.tensor_parallel_rank == 0:
    #             print(f"\n--- [FD DEBUG] Dumping GQA L{layer_id} QKV weight for rank 0... ---\n")
                
    #             weight_shard = self.qkv_proj.weight 

    #             # ========== [核心修改在这里 - Numpy 中介法] ==========
    #             # 1. 将 GPU Parameter 转换为 CPU numpy 数组
    #             weight_shard_numpy = weight_shard.numpy()

    #             # 2. 从 numpy 数组创建一个全新的、干净的 Paddle Tensor
    #             clean_weight_shard = paddle.to_tensor(weight_shard_numpy)
                
    #             # 3. 保存这个干净的 Tensor
    #             paddle.save(clean_weight_shard.astype("float32"), f"fd_gqa_l{layer_id}_qkv_weight_shard_rank0.pdparams")
    #             # ========== [结束修改] ==========
                
    #             print(f"\n--- [FD DEBUG] Saved rank 0 weight shard to fd_gqa_l{layer_id}_qkv_weight_shard_rank0.pdparams ---\n")
            
    #         type(self)._gqa_weight_dumped = True
    #     # ========== [结束 GQA 权重 Dump 代码] ==========

    #     logger.info(f"\n{'='*20} {run_mode} [FD DEBUG] Entering DecoderLayer {layer_id} {'='*20}")
    #     print_tensor_stats(hidden_states, f"{run_mode} L{layer_id}:0a_Input_HiddenStates", is_profile_run)
    #     print_tensor_stats(residual, f"{run_mode} L{layer_id}:0b_Input_Residual", is_profile_run)

    #     # --- Attention Block ---
    #     layernorm_output = self.input_layernorm(hidden_states)
    #     print_tensor_stats(layernorm_output, f"{run_mode} L{layer_id}:1_After_InputLayernorm", is_profile_run)

    #     residual_attn = layernorm_output if self.postnorm else hidden_states

    #     attn_output = None
    #     if self.attn_type == 1:  # GQA
    #         qkv_out = self.qkv_proj(layernorm_output)
    #         attn_output = self.self_attn(qkv=qkv_out, forward_meta=forward_meta)
    #         attn_output = self.o_proj(attn_output)
    #     else:  # Linear Attention
    #         attn_output = self.self_attn(layernorm_output, forward_meta)
    #     print_tensor_stats(attn_output, f"{run_mode} L{layer_id}:2_After_Attention", is_profile_run)

    #     # --- Residual Connection 1 ---
    #     hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
    #     print_tensor_stats(hidden_states_after_attn, f"{run_mode} L{layer_id}:3_After_Attn_Residual(alpha={self.layernorm_attention_alpha}, beta={self.layernorm_attention_beta})", is_profile_run)

    #     # --- MLP Block ---
    #     layernorm_output_mlp = self.post_attention_layernorm(hidden_states_after_attn)
    #     print_tensor_stats(layernorm_output_mlp, f"{run_mode} L{layer_id}:4_After_PostAttnLayernorm", is_profile_run)

    #     residual_mlp = layernorm_output_mlp if self.postnorm else hidden_states_after_attn

    #     mlp_output = self.mlp(layernorm_output_mlp)
    #     print_tensor_stats(mlp_output, f"{run_mode} L{layer_id}:5a_After_MoE_MLP", is_profile_run)

    #     if self.shared_moe:
    #         shared_output = self.shared_mlp(layernorm_output_mlp)
    #         coef_logits = self.coefficient(layernorm_output_mlp.cast("float32"))
    #         coef = F.sigmoid(coef_logits)
    #         mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef
    #         print_tensor_stats(mlp_output, f"{run_mode} L{layer_id}:5b_After_Shared_MLP_Merge", is_profile_run)

    #     # --- Residual Connection 2 ---
    #     final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
    #     print_tensor_stats(final_output, f"{run_mode} L{layer_id}:6_FinalOutput(alpha={self.layernorm_mlp_alpha}, beta={self.layernorm_mlp_beta})", is_profile_run)

    #     logger.info(f"{'='*20} [FD DEBUG] Exiting DecoderLayer {layer_id} {'='*20}\n")
    #     return final_output, None # residual is managed internally now
    
    
    def forward(self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, residual: Optional[paddle.Tensor], run_mode: str = "[UNKNOWN]"):
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

        # --- Start of FORCED INFERENCE DEBUGGING ---
        attn_type_str = "GQA" if self.attn_type == 1 else "LinearAttn"
        logger.info(f"\n{'='*20} [FD FORCE DEBUG] Entering DecoderLayer {layer_id} ({attn_type_str}) {'='*20}")
        print_tensor_stats(hidden_states, f"FD_L{layer_id}:0a_Input_HiddenStates")

        layernorm_output = self.input_layernorm(hidden_states)
        print_tensor_stats(layernorm_output, f"FD_L{layer_id}:1_After_InputLayernorm")

        residual_attn = layernorm_output if self.postnorm else hidden_states

        attn_output = None
        if self.attn_type == 1:  # GQA
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

        else:  # Linear Attention
            attn_output = self.self_attn(layernorm_output, forward_meta)

        print_tensor_stats(attn_output, f"FD_L{layer_id}:2_After_Attention")

        hidden_states_after_attn = (residual_attn * self.layernorm_attention_alpha) + (attn_output * self.layernorm_attention_beta)
        print_tensor_stats(hidden_states_after_attn, f"FD_L{layer_id}:3_After_Attn_Residual")

        layernorm_output_mlp = self.post_attention_layernorm(hidden_states_after_attn)
        print_tensor_stats(layernorm_output_mlp, f"FD_L{layer_id}:4_After_PostAttnLayernorm")
            
        residual_mlp = layernorm_output_mlp if self.postnorm else hidden_states_after_attn
        mlp_output = self.mlp(layernorm_output_mlp)
        
        print_tensor_stats(mlp_output, f"FD_L{layer_id}:5a_After_MoE_MLP")

        if self.shared_moe:
            shared_output = self.shared_mlp(layernorm_output_mlp)
            coef_logits = self.coefficient(layernorm_output_mlp.cast("float32"))
            coef = F.sigmoid(coef_logits)
            mlp_output = mlp_output.cast(coef.dtype) * (1 - coef) + shared_output.cast(coef.dtype) * coef
            print_tensor_stats(mlp_output, f"FD_L{layer_id}:5b_After_Shared_MLP_Merge")

        final_output = (residual_mlp * self.layernorm_mlp_alpha) + (mlp_output * self.layernorm_mlp_beta)
        print_tensor_stats(final_output, f"FD_L{layer_id}:6_FinalOutput")
        
        logger.info(f"{'='*20} [FD FORCE DEBUG] Exiting DecoderLayer {layer_id} ({attn_type_str}) {'='*20}\n")
            
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

        # 在所有打印前都加上 run_mode 前缀

        # ==================== 区分 Profile 和 正式推理 ====================
        # 在 profile_run 期间，forward_meta.seq_lens_encoder 通常会被设置为一个非零长度的 dummy tensor
        # 在真实推理时，它会反映真实的 prefill token 数量。
        # 一个更可靠的标志是 `step_use_cudagraph`，它在 profile 时为 True，推理时通常为 False。
        is_profile_run = forward_meta.step_use_cudagraph
        run_mode = "[PROFILE]" if is_profile_run else "[INFERENCE]"

        if self.fd_config.parallel_config.tensor_parallel_rank == 0:
            print(f"\n{'#'*20} FastDeploy RUN MODE: {run_mode} {'#'*20}\n")
        # =================================================================
        print_tensor_stats(ids_remove_padding, f"{run_mode} TOP:0_InputIDs", is_profile_run)
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
        print_tensor_stats(hidden_states, f"{run_mode} TOP:1_AfterEmbedding", is_profile_run)

        for i in range(len(self.layers)):
            layer = self.layers[str(i)]
            # 将 run_mode 传递下去
            hidden_states, _ = layer(forward_meta=forward_meta, hidden_states=hidden_states, residual=None, run_mode=run_mode)

        out = self.norm(hidden_states)
        print_tensor_stats(out, f"{run_mode} TOP:3_FinalOutput", is_profile_run)
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
        # ================== [金丝雀测试] ==================
        print("\n\n>>>>>> [DEBUG] EXECUTING THE NEWEST VERSION OF MINIMAX_M1_FOR_CAUSALLM INIT <<<<<<\n\n")
        # =================================================

        super().__init__(fd_config)

        # Save the model config as a self.config attribute
        self.config = self.fd_config.model_config
        self.config.pretrained_config.prefix_name = "model"
        print(f"self.config.rotary_dim {self.config.rotary_dim}")
        print(f"self.config.head_dim {self.config.head_dim}")
        
        if hasattr(self.config, "num_local_experts") and not hasattr(self.config, "moe_num_experts"):
            self.config.moe_num_experts = self.config.num_local_experts
        if (
            hasattr(self.config, "rotary_dim")
            and hasattr(self.config, "head_dim")
            and self.config.rotary_dim < self.config.head_dim
        ):
            
            self.config.partial_rotary_factor = self.config.rotary_dim / self.config.head_dim
            print(f"self.config.partial_rotary_factor {self.config.partial_rotary_factor}")
        if not hasattr(self.config, "first_k_dense_replace"):
            self.config.first_k_dense_replace = 0

        self.model = MiniMaxM1Model(fd_config)
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=self.config.hidden_size,
            num_embeddings=self.config.vocab_size,
            prefix="lm_head",
        )
        # ========== [在这里加入 DEBUG 代码] ==========
        if fd_config.parallel_config.tensor_parallel_rank == 0:
            print("\n" + "="*20 + " ALL MODEL PARAMETERS " + "="*20)
            for name, param in self.model.named_parameters():
                # 我们只关心 MoE expert 的权重名
                if "mlp.experts" in name:
                    print(f"Found MoE Param: {name} | Shape: {param.shape}")
            print("="*50 + "\n")
        # ========== [结束 DEBUG 代码] ==========

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
        logger.info("Initializing robust multi-GPU weight loader for MiniMax-M1...")
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

        # ==================== 开始重构的加载循环 V2 ====================
        for loaded_weight_name, loaded_weight_slice in weights_iterator:
            
            # --- 统一的权重预处理 ---
            current_weight = None
            if "PySafeSlice" in str(type(loaded_weight_slice)):
                dtype_str = str(getattr(loaded_weight_slice, 'dtype', '')).lower()
                if 'bfloat16' in dtype_str or 'bf16' in dtype_str:
                    current_weight = _get_bfloat16_tensor_from_slice(loaded_weight_slice)
            
            if current_weight is None:
                current_weight = get_tensor(loaded_weight_slice)
            
            # --- 逻辑分发开始 ---
            was_handled = False 

            if self.fd_config.parallel_config.tensor_parallel_rank == 0:
                 logger.info(f"[LOADER_V2_DEBUG] Processing raw weight: '{loaded_weight_name}'")

            # 规则 1: MoE Expert Weights
            moe_match = re.search(r"(\.layers\.\d+\.)block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
            if moe_match:
                prefix_path, expert_id_str, weight_type = moe_match.groups()
                expert_id = int(expert_id_str)
                
                # ========== [ 这里是修正的部分 ] ==========
                # 从 prefix_path (e.g., '.layers.2.') 构造出 mlp_prefix (e.g., 'model.layers.2.mlp.')
                mlp_prefix = f"model{prefix_path}mlp."
                # ========== [ 修正结束 ] ==========

                # 现在可以正确地构造 FusedMoE 内部参数名
                if weight_type in ["w1", "w3"]:
                    # 假设 FusedMoE 内部参数名是 experts.up_gate_proj_weight
                    param_name = f"{mlp_prefix}experts.up_gate_proj_weight" 
                    shard_id = "gate" if weight_type == "w1" else "up"
                else: # w2
                    # 假设 FusedMoE 内部参数名是 experts.down_proj_weight
                    param_name = f"{mlp_prefix}experts.down_proj_weight"
                    shard_id = "down"

                if param_name in params_dict:
                    param = params_dict[param_name]
                    if hasattr(param, 'weight_loader'):
                        logger.info(f"[LOADER_V2_DEBUG] Handled by [MoE Expert Loader] -> {param_name} (expert: {expert_id}, shard: {shard_id})")
                        param.weight_loader(param, current_weight, expert_id=expert_id, shard_id=shard_id)
                        process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                        was_handled = True
                
                if was_handled: continue

            # 规则 2: Attention Weights (GQA 或 Linear)
            if 'self_attn' in loaded_weight_name:
                layer_match = re.search(r'\.layers\.(\d+)\.', loaded_weight_name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    # ========== [ 添加这部分边界检查 ] ==========
                    if layer_idx >= self.config.num_hidden_layers:
                        # 如果层号超出了模型定义的范围，就跳过这个权重
                        logger.warning(f"[LOADER_V2_DEBUG] Skipping weight for out-of-bounds layer index {layer_idx}: {loaded_weight_name}")
                        continue
                    # ========== [ 结束检查 ] ==========

                    
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
                                logger.info(f"[LOADER_V2_DEBUG] Handled by [GQA Loader] -> {target_param_name} (shard: {shard_id})")
                                loader(param, current_weight, shard_id)
                                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                                was_handled = True

                    elif self.config.attn_type_list[layer_idx] == 0: # Linear Attention
                        # Linear Attention 的权重名在 checkpoint 和模型中应该能直接对应
                        # 例如 `model.layers.3.self_attn.qkv_proj.weight`
                        target_param_name = loaded_weight_name
                        if target_param_name in params_dict:
                            param = params_dict[target_param_name]
                            loader = getattr(param, 'weight_loader', None)
                            if loader:
                                logger.info(f"[LOADER_V2_DEBUG] Handled by [Linear Attn Loader] -> {target_param_name}")
                                # Linear Attention 的 loader 可能不需要 shard_id
                                loader(param, current_weight)
                                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                                was_handled = True

                if was_handled: continue
            
            # 规则 3: 其他通用名称映射 (不包括 Attention 和 MoE expert)
            param_name = loaded_weight_name
            simple_rename_map = {
                "block_sparse_moe.gate.weight": "mlp.gate.weight",
                "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
                "lm_head.weight": "lm_head.linear.weight",
            }
            # 应用重命名
            for old, new in simple_rename_map.items():
                if old in param_name:
                    param_name = param_name.replace(old, new)
            
            # 规则 4 (最终 fallback): 只处理名字能直接匹配的简单权重
            if not was_handled and param_name in params_dict:
                param = params_dict[param_name]
                loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                logger.info(f"[LOADER_V2_DEBUG] Handled by [Fallback Loader] -> {param_name}")
                loader(param, current_weight) # Fallback 不应该有 shard_id
                process_weights_after_loading_fn(param.name.rsplit(".", 1)[0], param)
                was_handled = True

            # 如果所有规则都未处理，则发出警告
            if not was_handled:
                logger.warning(f"[LOADER_V2_DEBUG] Weight '{loaded_weight_name}' was NOT handled by any rule (final tried name: {param_name}).")
        
        logger.info("Weight loading process finished.")
    

    # @paddle.no_grad()
    # def load_weights(self, weights_iterator) -> None:
    #     logger.info("Initializing robust multi-GPU weight loader for MiniMax-M1...")
    #     import numpy as np
    #     import re
    #     from fastdeploy.model_executor.utils import (
    #         get_tensor,
    #         default_weight_loader,
    #         process_weights_after_loading,
    #     )

    #     params_dict = dict(self.named_parameters())
    #     sublayers_dict = dict(self.named_sublayers())
    #     process_weights_after_loading_fn = process_weights_after_loading(sublayers_dict)

    #     def _get_bfloat16_tensor_from_slice(weight_slice):
    #         data_bytes = weight_slice[:]
    #         uint16_array = np.frombuffer(data_bytes, dtype=np.uint16)
    #         bfloat16_dtype_obj = paddle.to_tensor([0], dtype='bfloat16').numpy().dtype
    #         bf16_array = uint16_array.view(bfloat16_dtype_obj)
    #         bf16_array_reshaped = bf16_array.reshape(weight_slice.shape)
    #         return paddle.to_tensor(bf16_array_reshaped)

    #     for loaded_weight_name, loaded_weight_slice in weights_iterator:
            
    #         # Step 1: Uniformly convert the loaded slice to a Paddle tensor
    #         current_weight = None
    #         if "PySafeSlice" in str(type(loaded_weight_slice)):
    #             dtype_str = str(getattr(loaded_weight_slice, 'dtype', '')).lower()
    #             if 'bfloat16' in dtype_str or 'bf16' in dtype_str:
    #                 current_weight = _get_bfloat16_tensor_from_slice(loaded_weight_slice)
            
    #         if current_weight is None:
    #             current_weight = get_tensor(loaded_weight_slice)
                
    #         if "layers.0.self_attn.qkv_proj.weight" in loaded_weight_name:
    #             # 只有 rank 0 打印
    #             if self.fd_config.parallel_config.tensor_parallel_rank == 0:
    #                 print("\n" + "="*20 + " FD RAW WEIGHT DEBUG " + "="*20)
                    
    #                 # 1. 打印原始加载的全量权重
    #                 full_weight_flat = current_weight.detach().cpu().astype('float32').flatten().numpy()
    #                 print("FD RAW FULL weight, first 20:", full_weight_flat[:20])
                    
    #                 print("="*20 + " END FD DEBUG " + "="*20 + "\n")


    #         # Step 2: Dispatch the loading task
            
    #         layer_match = re.search(r'\.layers\.(\d+)\.', loaded_weight_name)
    #         if layer_match and int(layer_match.group(1)) >= self.config.num_hidden_layers:
    #             continue

    #         param_to_load = None

    #         # Rule 1: MoE Expert Weights
    #         moe_match = re.search(r"(\.layers\.\d+\.)block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight", loaded_weight_name)
    #         if moe_match:
    #             prefix_path, expert_id_str, weight_type = moe_match.groups()
    #             expert_id = int(expert_id_str)
    #             mlp_prefix = f"model{prefix_path.replace('block_sparse_moe', 'mlp')}"
    #             if weight_type in ["w1", "w3"]:
    #                 param_name = f"{mlp_prefix}mlp.experts.up_gate_proj_weight"
    #                 shard_id = "gate" if weight_type == "w1" else "up"
    #             else: # w2
    #                 param_name = f"{mlp_prefix}mlp.experts.down_proj_weight"
    #                 shard_id = "down"

    #             if param_name in params_dict:
    #                 param = params_dict[param_name]
    #                 if hasattr(param, 'weight_loader'):
    #                     param.weight_loader(param, current_weight, expert_id=expert_id, shard_id=shard_id)
    #                     param_to_load = param
    #             continue
            
    #         # Rule 2: General name mapping and default loading
    #         param_name = loaded_weight_name
    #         shard_id = None # Used for GQA weights

    #         # Handle GQA weight name mapping to qkv_proj and identify shard_id
    #         if 'self_attn' in loaded_weight_name and '.layers.' in loaded_weight_name:
    #             layer_idx = int(re.search(r'\.layers\.(\d+)\.', loaded_weight_name).group(1))
    #             if self.config.attn_type_list[layer_idx] == 1: # GQA
    #                 if 'q_proj.weight' in loaded_weight_name: 
    #                     param_name = loaded_weight_name.replace('self_attn.q_proj.weight', 'qkv_proj.weight')
    #                     shard_id = 'q'
    #                 elif 'k_proj.weight' in loaded_weight_name: 
    #                     param_name = loaded_weight_name.replace('self_attn.k_proj.weight', 'qkv_proj.weight')
    #                     shard_id = 'k'
    #                 elif 'v_proj.weight' in loaded_weight_name: 
    #                     param_name = loaded_weight_name.replace('self_attn.v_proj.weight', 'qkv_proj.weight')
    #                     shard_id = 'v'
    #                 elif 'o_proj.weight' in loaded_weight_name: 
    #                     param_name = loaded_weight_name.replace('self_attn.o_proj.weight', 'o_proj.weight')

    #         # Handle other general name mappings
    #         simple_rename_map = {
    #             "block_sparse_moe.gate.weight": "mlp.gate.weight",
    #             "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
    #             "lm_head.weight": "lm_head.linear.weight",
    #         }
    #         for old, new in simple_rename_map.items():
    #             if old in param_name:
    #                 param_name = param_name.replace(old, new)
            
    #         if param_name in params_dict:
    #             param = params_dict[param_name]
    #             loader = getattr(param, 'weight_loader', default_weight_loader(self.fd_config))
                
    #             # The loader for QKVParallelLinear expects a shard_id, others may ignore it.
    #             # Our custom loader for LinearAttention will correctly ignore it.
    #             loader(param, current_weight, shard_id)
    #             param_to_load = param
    #         else:
    #             if not any(rule in loaded_weight_name for rule in ['q_proj', 'k_proj', 'v_proj']):
    #                 logger.warning(f"Weight '{loaded_weight_name}' was not used (tried name '{param_name}').")

    #         if param_to_load is not None:
    #             # 在调用 process_weights_after_loading_fn 之前，我们检查一下
    #             # if "layers.0.self_attn.qkv_proj.weight" in param_to_load.name:
    #             #     logger.warning("--- [FD DEBUG] AFTER DEFAULT LOADER ---")
    #             #     print_tensor_stats(param_to_load, "L0_QKV_WEIGHT_AFTER_DEFAULT_LOAD")
    #             if "layers.0.self_attn.qkv_proj.weight" in param_to_load.name:
    #                 if self.fd_config.parallel_config.tensor_parallel_rank == 0:
    #                     logger.warning("--- [FD DEBUG] AFTER CUSTOM LOADER ---")
                        
    #                     # 2. 打印最终加载到参数里的值
    #                     final_shard_flat = param_to_load.detach().cpu().astype('float32').flatten().numpy()
    #                     print("FD FINAL SHARD (rank 0), first 20:", final_shard_flat[:20])
                        
    #                     # 顺便把完整的统计信息也打了
    #                     print_tensor_stats(param_to_load, "L0_QKV_WEIGHT_AFTER_CUSTOM_LOAD")
                    
                
    #             sublayer_name = param_to_load.name.rsplit(".", 1)[0]
    #             process_weights_after_loading_fn(sublayer_name, param_to_load)

    #     logger.info("Weight loading process finished.")
        
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