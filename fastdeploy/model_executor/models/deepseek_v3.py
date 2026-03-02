"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Tuple

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
    KVBatchLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
)
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import (
        get_position_ids_and_mask_encoder_batch,
    )

from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    per_token_group_quant_fp8,
)
from fastdeploy.model_executor.ops.gpu import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
    radix_topk_ragged_transform,
)

paddle.enable_compat(scope={"flash_mla", "deep_gemm"})  # Enable torch proxy before importing flash_mla
import deep_gemm


class DeepSeekV3MLP(nn.Layer):
    """
    DeepSeekV3MLP, for Dense FFN and Shared Experts Layer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
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
            reduce_results=reduce_results,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=None,
            act_method=fd_config.model_config.hidden_act,
        )

    # def load_state_dict(self, state_dict):
    #     """ """
    #     self.up_gate_proj.load_state_dict(state_dict)
    #     self.down_proj.load_state_dict(state_dict)

    def forward(self, x, forward_meta=None):
        """ """
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class DeepSeekV3MoE(nn.Layer):
    """
    DeepSeekV3MoE, for MoE Layer.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.norm_topk_prob = fd_config.model_config.norm_topk_prob

        weight_key_map = {
            "gate_correction_bias_key": f"{prefix}.gate.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.n_routed_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        if fd_config.model_config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = self.create_parameter(
                shape=[1, fd_config.model_config.n_routed_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            fd_config=fd_config,
            reduce_results=False,
            renormalize=self.norm_topk_prob,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.n_routed_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method=fd_config.model_config.topk_method,
            topk_group=fd_config.model_config.topk_group,
            n_group=fd_config.model_config.n_group,
            routed_scaling_factor=fd_config.model_config.routed_scaling_factor,
            layer_idx=layer_id,
            gate_correction_bias=self.gate.e_score_correction_bias,
            weight_key_map=weight_key_map,
        )

        self.num_shared_experts = fd_config.model_config.n_shared_experts
        shared_experts_intermediate_size = self.num_shared_experts * fd_config.model_config.moe_intermediate_size

        self.shared_experts = DeepSeekV3MLP(
            fd_config=fd_config,
            intermediate_size=shared_experts_intermediate_size,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )

    # def load_state_dict(self, state_dict):
    #     """ """
    #     if self.experts.gate_correction_bias is not None:
    #         gate_correction_bias_tensor = state_dict.pop(self.experts.gate_correction_bias_key)
    #         if self.experts.gate_correction_bias.shape != gate_correction_bias_tensor.shape:
    #             gate_correction_bias_tensor = gate_correction_bias_tensor.reshape(
    #                 self.experts.gate_correction_bias.shape
    #             )
    #         self.experts.gate_correction_bias.set_value(gate_correction_bias_tensor)
    #     self.gate.load_state_dict(state_dict)
    #     self.experts.load_state_dict(state_dict)
    #     self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        """ """
        shared_experts_out = self.shared_experts(hidden_states)
        moe_out = self.experts(hidden_states, self.gate, forward_meta)
        moe_out = moe_out + shared_experts_out
        # We do to TP all reduce after the sum of experts.
        if self.tp_size > 1:
            moe_out = tensor_model_parallel_all_reduce(moe_out)
        return moe_out


class DeepseekV3MLAAttention(nn.Layer):
    """
    DeepseekV3MLAAttention
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size

        # MLA
        self.qk_nope_head_dim = fd_config.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = fd_config.model_config.v_head_dim
        self.q_lora_rank = fd_config.model_config.q_lora_rank
        self.kv_lora_rank = fd_config.model_config.kv_lora_rank

        self.attn_softmax_scale = self.qk_head_dim**-0.5
        self.rope_theta = fd_config.model_config.rope_theta
        self.rms_norm_eps = fd_config.model_config.rms_norm_eps

        assert self.q_lora_rank is not None, "self.q_lora_rank is None, Please Check your config."
        # NOTE: (changwenbin) qkv_a_proj horizontal fusion
        self.qkv_a_proj_with_mqa = MergedReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_a_proj_with_mqa",
            input_size=self.hidden_size,
            output_sizes=[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            with_bias=False,
        )

        self.q_a_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.q_lora_rank,
            eps=self.rms_norm_eps,
            prefix=f"{prefix}.q_a_layernorm",
        )

        self.q_b_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.q_b_proj",
            input_size=self.q_lora_rank,
            output_size=self.num_attention_heads * self.qk_head_dim,
            with_bias=False,
        )

        self.kv_a_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.kv_lora_rank,
            eps=self.rms_norm_eps,
            prefix=f"{prefix}.kv_a_layernorm",
        )

        self.kv_b_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.kv_b_proj",
            input_size=self.kv_lora_rank,
            output_size=self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            with_bias=False,
        )

        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_attention_heads * self.v_head_dim,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )

        self.kv_b_proj_bmm = KVBatchLinear(
            fd_config=fd_config,
            kv_b_proj=self.kv_b_proj,
            prefix=f"{prefix}.kv_b_proj",
            kv_lora_rank=self.kv_lora_rank,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        self.rope_scaling = fd_config.model_config.rope_scaling
        if self.rope_scaling:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", False)
            scaling_factor = self.rope_scaling["factor"]
            mscale = self.yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        rope_scaling_kwargs = {
            key: self.rope_scaling[key]
            for key in [
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.rope_scaling
        }
        self.rope_scaling_factor = self.rope_scaling["factor"]
        self.rope_scaling_original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling_original_max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
            **rope_scaling_kwargs,
        )

        self.mla_attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

        self.prefix = prefix

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        """ """
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """ """

        # NOTE: (changwenbin) Bring out the public calculation in PD MIX to avoid repeated calculation.
        fmha_out = None

        # NOTE: (changwenbin) qkv_a_proj horizontal fusion
        qkv_a_out = self.qkv_a_proj_with_mqa(hidden_states)

        query, compressed_kv, key_pe = qkv_a_out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], axis=-1
        )

        query = self.q_a_layernorm(query)[0]
        query = self.q_b_proj(query)
        query.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
        query_nope, query_pe = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        key_pe.reshape_([-1, 1, self.qk_rope_head_dim])
        query_pe, key_pe = self.rotary_emb(position_ids, query_pe, key_pe)

        compressed_kv = self.kv_a_layernorm(compressed_kv)[0]

        if forward_meta.max_len_tensor_cpu[1]:  # max_enc_len_this_time
            key_value = self.kv_b_proj(compressed_kv)
            key_value.reshape_(
                [
                    -1,
                    self.num_attention_heads_tp,
                    self.qk_nope_head_dim + self.v_head_dim,
                ]
            )
            key_nope, value = key_value.split([self.qk_nope_head_dim, self.v_head_dim], axis=-1)

            query[..., self.qk_nope_head_dim :] = query_pe
            key = paddle.empty_like(query)
            key[..., : self.qk_nope_head_dim] = key_nope
            key[..., self.qk_nope_head_dim :] = key_pe
            value = paddle.nn.functional.pad(value, [0, self.qk_head_dim - self.v_head_dim], value=0)

            fmha_out_prefill = self.mla_attn(
                q=query,
                k=key,
                v=value,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmha_out_prefill.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, : self.v_head_dim]
            fmha_out_prefill.reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            fmha_out_prefill = fmha_out_prefill * mask_encoder_batch.cast(fmha_out_prefill.dtype)

            fmha_out = fmha_out_prefill

        if forward_meta.max_len_tensor_cpu[2]:  # max_dec_len_this_time
            q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]), proj_type="k").transpose([1, 0, 2])

            q_input = paddle.concat([q_nope_out, query_pe], axis=-1)
            q_input.reshape_(
                [
                    -1,
                    self.num_attention_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )

            fmha_out_decode = self.mla_attn(
                q=q_input,
                k=None,
                v=None,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmha_out_decode = fmha_out_decode.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose(
                [1, 0, 2]
            )

            fmha_out_decode = (
                self.kv_b_proj_bmm(fmha_out_decode, proj_type="v")
                .transpose([1, 0, 2])
                .reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            )

            if fmha_out is None:
                fmha_out = fmha_out_decode
            else:
                fmha_out = fmha_out + fmha_out_decode
        # breakpoint()
        output = self.o_proj(fmha_out)
        return output

    # def load_state_dict(self, state_dict):
    #     """ """
    #     self.q_a_layernorm.load_state_dict(state_dict)
    #     self.qkv_a_proj_with_mqa.load_state_dict(state_dict)
    #     self.kv_a_layernorm.load_state_dict(state_dict)
    #     self.q_b_proj.load_state_dict(state_dict)
    #     self.kv_b_proj_bmm.load_state_dict(state_dict)
    #     self.kv_b_proj.load_state_dict(state_dict)
    #     # NOTE(Ryan):Make sure kv_b_proj_bmm loaded before kv_b_proj,
    #     # The same weight key will be poped after kv_b_proj.
    #     self.o_proj.load_state_dict(state_dict)
    #     self.mla_attn.load_state_dict(state_dict)


def compute_slot_mapping(
    block_tables: paddle.Tensor,  # [num_reqs, max_blocks_per_req]
    positions: paddle.Tensor,  # [num_tokens] 每个token的位置
    batch_id_per_token: paddle.Tensor,  # [num_tokens] 每个token属于哪个请求
    block_size: int,
) -> paddle.Tensor:
    """
    计算 slot_mapping

    公式: slot = block_id * block_size + offset_in_block
    """
    # 1. 计算每个 token 对应的 block 索引
    block_idx = positions // block_size  # [num_tokens]

    # 2. 从 block_tables 中查表获取 block_id
    # block_tables[batch_id_per_token, block_idx]
    block_ids = block_tables[batch_id_per_token, block_idx]  # [num_tokens]

    # 3. 计算在 block 内的偏移
    block_offset = positions % block_size  # [num_tokens]

    # 4. 计算 slot_mapping
    slot_mapping = block_ids * block_size + block_offset

    return slot_mapping.cast(paddle.int64)


class Indexer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str = "",
    ):
        super().__init__()
        self.config = fd_config
        self.layer_id = layer_id
        self.max_model_len = fd_config.model_config.max_model_len

        self.index_head_dim = fd_config.model_config.index_head_dim
        self.index_n_heads = fd_config.model_config.index_n_heads
        self.index_topk = fd_config.model_config.index_topk

        self.rope_dim = fd_config.model_config.qk_rope_head_dim  # 64
        self.q_lora_rank = fd_config.model_config.q_lora_rank  # 1536
        self.hidden_size = fd_config.model_config.hidden_size
        # no tensor parallel, just replicated
        # breakpoint()
        self.wq_b = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.wq_b",
            input_size=self.q_lora_rank,
            output_size=self.index_head_dim * self.index_n_heads,
            with_bias=False,
            skip_quant=True,
            weight_dtype="bfloat16",
        )
        self.wk = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.wk",
            input_size=self.hidden_size,
            output_size=self.index_head_dim,
            with_bias=False,
            skip_quant=True,
            weight_dtype="bfloat16",
        )
        self.k_norm = RMSNorm(fd_config, self.index_head_dim, eps=1e-6, prefix=f"{prefix}.k_norm")
        # self.k_norm = LayerNorm(self.head_dim, eps=1e-6)

        self.weights_proj = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.weights_proj",
            input_size=self.hidden_size,
            output_size=self.index_n_heads,
            with_bias=False,
            skip_quant=True,
            weight_dtype="bfloat16",
        )

        self.softmax_scale = self.index_head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config

        self.offsets = paddle.zeros([self.max_model_len], dtype="int32")
        # self.buffer = paddle.zeros([2048 * 2048], dtype=paddle.uint8)
        self.lengths = paddle.zeros([self.max_model_len], dtype="int32")

    def forward(
        self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, qr: paddle.Tensor, positions, rotary_emb
    ) -> paddle.Tensor:
        self.indexer_cache = forward_meta.caches[2 * self.layer_id + 1]

        # breakpoint()
        q = self.wq_b(qr)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = paddle.split(q, [self.rope_dim, self.index_head_dim - self.rope_dim], axis=-1)

        k = self.wk(hidden_states)
        k, _ = self.k_norm(k)
        k_pe, k_nope = paddle.split(k, [self.rope_dim, self.index_head_dim - self.rope_dim], axis=-1)

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
        # so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.index_n_heads, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = paddle.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = paddle.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.index_head_dim)
        # breakpoint()
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.index_n_heads, self.index_head_dim)
        q_scale = q_scale.view(-1, self.index_n_heads, 1)

        weights = self.weights_proj(hidden_states)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.index_n_heads**-0.5
        weights = weights.squeeze(-1)

        slot_mapping = compute_slot_mapping(
            forward_meta.block_tables,
            forward_meta.position_ids,
            forward_meta.batch_id_per_token,
            64,
        )

        if forward_meta.max_len_tensor_cpu[1]:

            def ceil_to_ue8m0(x: paddle.Tensor):
                return paddle.pow(paddle.full([1], 2.0, device=x.place), paddle.ceil(paddle.log2(x.abs())))

            def per_custom_dims_cast_to_fp8(
                x: paddle.Tensor, dims: Tuple, use_ue8m0: bool
            ) -> Tuple[paddle.Tensor, paddle.Tensor]:
                excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
                x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
                sf = x_amax / 448.0
                sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
                x_scaled = (x * (1.0 / sf)).to(paddle.float8_e4m3fn)
                return x_scaled, sf.squeeze()

            kv_fp8 = per_custom_dims_cast_to_fp8(k, (0,), False)

            print("forward_meta.seq_lens_encoder", forward_meta.seq_lens_encoder)

            # ===================================== cache =============================================

            indexer_k_quant_and_cache(k, self.indexer_cache, slot_mapping, self.quant_block_size, self.scale_fmt)
            # paddle.set_printoptions(precision=4, threshold=160, edgeitems=40, sci_mode=None, linewidth=80)
            # self.indexer_cache.view(paddle.float8_e4m3fn)
            # self.indexer_cache.view(paddle.float32)
            # # self.indexer_cache.reshape(-1,132)[:8190]
            # (self.indexer_cache.view(paddle.float32)>0).sum()

            # breakpoint()
            k_fp8_cache = paddle.zeros_like(k, dtype=paddle.uint8)
            k_scale_cache = paddle.zeros([k.shape[0], 4], dtype=paddle.float32)
            cp_gather_indexer_k_quant_cache(
                self.indexer_cache, k_fp8_cache, k_scale_cache, forward_meta.block_tables, forward_meta.cu_seqlens_k
            )
            k_scale_cache = k_scale_cache.flatten()[: k.shape[0]]
            k_cache = k_fp8_cache.view(paddle.float8_e4m3fn), k_scale_cache
            # ===================================== cache =============================================

            # ks,ke = forward_meta.attn_mask_offsets[::2].contiguous(),forward_meta.attn_mask_offsets[1::2].contiguous()
            ks = paddle.zeros(forward_meta.seq_lens_encoder[0], dtype=paddle.int32)
            ke = paddle.arange(forward_meta.seq_lens_encoder[0], dtype=paddle.int32) + 1  # + (seq_len_kv - seq_len)
            max_seqlen_k = (ke - ks).max().item()

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8, k_cache, weights, ks, ke, max_seqlen_k=max_seqlen_k, clean_logits=False
            )
            assert logits.size() == (forward_meta.seq_lens_encoder[0], max_seqlen_k)
            tmp = paddle.full(
                (forward_meta.seq_lens_encoder[0], forward_meta.seq_lens_encoder[0]),
                float("-inf"),
            )
            for i in range(forward_meta.seq_lens_encoder[0]):
                tmp[i, ks[i] : ke[i]] = logits[i, : ke[i] - ks[i]]
            logits = tmp

            # breakpoint()
            indexer_top_k = paddle.full([logits.shape[0], self.index_topk], -1, dtype="int32")

            radix_topk_ragged_transform(
                logits.contiguous(),
                indexer_top_k,
                ks,  # self.offsets,
                ke,  # mask.contiguous(),#self.lengths,
                None,  # forward_meta.seq_lens_decoder,
                None,  # forward_meta.batch_id_per_token,
                None,  # self.buffer
                self.index_topk,
                1,
            )
        if forward_meta.max_len_tensor_cpu[2]:

            indexer_k_quant_and_cache(k, self.indexer_cache, slot_mapping, self.quant_block_size, self.scale_fmt)

            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                forward_meta.seq_lens_decoder.squeeze(1) + 1, 64, deep_gemm.get_num_sms()
            )

            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8.unsqueeze(1),
                self.indexer_cache.unsqueeze(2),
                weights,
                forward_meta.seq_lens_decoder.squeeze(1) + 1,
                forward_meta.block_tables,
                schedule_metadata,
                self.max_model_len,
                clean_logits=True,
            )

            indexer_top_k = paddle.full([logits.shape[0], self.index_topk], -1, dtype="int32")

            radix_topk_ragged_transform(
                logits.contiguous(),
                indexer_top_k,
                self.offsets,
                self.lengths,
                forward_meta.seq_lens_decoder + 1,
                forward_meta.batch_id_per_token,
                None,  # self.buffer
                self.index_topk,
                1,
            )

        return indexer_top_k


class DeepseekV32DSAAttention(nn.Layer):
    """
    DeepseekV32DSAAttention
    """

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size

        # MLA
        self.qk_nope_head_dim = fd_config.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = fd_config.model_config.v_head_dim
        self.q_lora_rank = fd_config.model_config.q_lora_rank
        self.kv_lora_rank = fd_config.model_config.kv_lora_rank

        # Indexer
        self.index_head_dim = fd_config.model_config.index_head_dim
        self.index_n_heads = fd_config.model_config.index_n_heads
        self.index_topk = fd_config.model_config.index_topk

        self.attn_softmax_scale = self.qk_head_dim**-0.5
        self.rope_theta = fd_config.model_config.rope_theta
        self.rms_norm_eps = fd_config.model_config.rms_norm_eps

        assert self.q_lora_rank is not None, "self.q_lora_rank is None, Please Check your config."
        # NOTE: (changwenbin) qkv_a_proj horizontal fusion
        self.qkv_a_proj_with_mqa = MergedReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_a_proj_with_mqa",
            input_size=self.hidden_size,
            output_sizes=[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            with_bias=False,
        )

        self.q_a_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.q_lora_rank,
            eps=self.rms_norm_eps,
            prefix=f"{prefix}.q_a_layernorm",
        )

        self.q_b_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.q_b_proj",
            input_size=self.q_lora_rank,
            output_size=self.num_attention_heads * self.qk_head_dim,
            with_bias=False,
        )

        self.kv_a_layernorm = RMSNorm(
            fd_config,
            hidden_size=self.kv_lora_rank,
            eps=self.rms_norm_eps,
            prefix=f"{prefix}.kv_a_layernorm",
        )

        self.kv_b_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.kv_b_proj",
            input_size=self.kv_lora_rank,
            output_size=self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            with_bias=False,
        )

        self.o_proj = RowParallelLinear(
            fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.num_attention_heads * self.v_head_dim,
            output_size=self.hidden_size,
            with_bias=False,
            layer_id=layer_id,
        )

        self.kv_b_proj_bmm = KVBatchLinear(
            fd_config=fd_config,
            kv_b_proj=self.kv_b_proj,
            prefix=f"{prefix}.kv_b_proj",
            kv_lora_rank=self.kv_lora_rank,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        self.rope_scaling = fd_config.model_config.rope_scaling
        if self.rope_scaling:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", False)
            scaling_factor = self.rope_scaling["factor"]
            mscale = self.yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        rope_scaling_kwargs = {
            key: self.rope_scaling[key]
            for key in [
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.rope_scaling
        }
        self.rope_scaling_factor = self.rope_scaling["factor"]
        self.rope_scaling_original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling_original_max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
            **rope_scaling_kwargs,
        )
        self.indexer_rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling_original_max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
            **rope_scaling_kwargs,
        )

        self.indexer = Indexer(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
        )
        self.dsa_attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

        self.prefix = prefix

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        """ """
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """ """
        forward_meta.position_ids = position_ids
        # NOTE: (changwenbin) Bring out the public calculation in PD MIX to avoid repeated calculation.
        fmha_out = None

        # NOTE: (changwenbin) qkv_a_proj horizontal fusion
        qkv_a_out = self.qkv_a_proj_with_mqa(hidden_states)

        query, compressed_kv, key_pe = qkv_a_out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], axis=-1
        )

        query = self.q_a_layernorm(query)[0]
        qr = query
        query = self.q_b_proj(query)
        query.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
        query_nope, query_pe = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        key_pe.reshape_([-1, 1, self.qk_rope_head_dim])
        query_pe, key_pe = self.rotary_emb(position_ids, query_pe, key_pe)

        compressed_kv = self.kv_a_layernorm(compressed_kv)[0]
        kv = paddle.concat([compressed_kv, key_pe.squeeze(1)], axis=-1)

        q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]).contiguous(), proj_type="k")

        q_input = paddle.concat([q_nope_out.transpose([1, 0, 2]).contiguous(), query_pe], axis=-1)

        indexer_top_k = self.indexer(forward_meta, hidden_states, qr, position_ids, rotary_emb=self.indexer_rotary_emb)

        if forward_meta.max_len_tensor_cpu[1]:  # max_enc_len_this_time
            print("this is prefill port")
            # q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]).contiguous(), proj_type="k")

            # q_input = paddle.concat([q_nope_out.transpose([1,0,2]).contiguous(), query_pe], axis=-1)

            # indexer_top_k = self.indexer(forward_meta, hidden_states, qr, position_ids, rotary_emb = self.indexer_rotary_emb )

            # fmha_out_prefill,_,__ = flash_mla.flash_mla_sparse_fwd(
            #     q_input.contiguous(),
            #     kv.unsqueeze(1),
            #     indexer_top_k.unsqueeze(1),
            #     sm_scale= self.attn_softmax_scale,
            # )

            # if paddle.isnan(q_input.contiguous()).sum().item() > 0:
            #     print("q_input is NAN")
            #     breakpoint()
            # if paddle.isnan(kv.unsqueeze(1).contiguous()).sum().item() > 0:
            #     print("kv is NAN")
            #     breakpoint()
            # if paddle.isnan(indexer_top_k.unsqueeze(1).contiguous()).sum().item() > 0:
            #     print("indexer_top_k is NAN")
            #     breakpoint()

            fmha_out = self.dsa_attn(
                q=q_input.contiguous(),
                k=kv.unsqueeze(1).contiguous(),
                v=indexer_top_k.unsqueeze(1).contiguous(),
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )
            # if paddle.isnan(fmha_out_prefill).sum().item() > 0:
            #     print("fmha_out_prefill is NAN")
            #     breakpoint()
            # print("fmha_out_prefill",fmha_out_prefill)

            # fmha_out_prefill = fmha_out_prefill.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose(
            #     [1, 0, 2]
            # )

            # fmha_out = self.kv_b_proj_bmm(
            #     fmha_out_prefill,
            #     proj_type="v",
            # ).transpose([1, 0, 2]).reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])

        if forward_meta.max_len_tensor_cpu[2]:  # max_dec_len_this_time
            print("this is decode port")

            # q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]).contiguous(), proj_type="k")

            # q_input = paddle.concat([q_nope_out.transpose([1,0,2]).contiguous(), query_pe], axis=-1)

            # indexer_top_k = self.indexer(forward_meta, hidden_states, qr, position_ids, rotary_emb = self.indexer_rotary_emb )

            fmha_out = self.dsa_attn(
                q=q_input,
                k=None,
                v=indexer_top_k.unsqueeze(1).contiguous(),
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            # fmha_out_decode = fmha_out_decode.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose(
            #     [1, 0, 2]
            # )

            # fmha_out_decode = self.kv_b_proj_bmm(
            #     fmha_out_decode,
            #     proj_type="v",
            # ).transpose([1, 0, 2]).reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])

            # if fmha_out is None:
            #     fmha_out = fmha_out_decode
            # else:
            #     fmha_out = fmha_out + fmha_out_decode

            # ================decode end

        fmha_out = fmha_out.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose(
            [1, 0, 2]
        )
        fmha_out = (
            self.kv_b_proj_bmm(
                fmha_out,
                proj_type="v",
            )
            .transpose([1, 0, 2])
            .reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
        )

        output = self.o_proj(fmha_out)

        # if paddle.isnan(output).sum().item() > 0:
        #     print("fmha_out is NAN")
        #     breakpoint()
        # print("output = ",output)
        # # breakpoint()

        return output


class DeepSeekV3DecoderLayer(nn.Layer):
    """
    DeepSeekV3DecoderLayer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        if fd_config.model_config.model_type == "deepseek_v32":
            self.self_attn = DeepseekV32DSAAttention(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = DeepseekV3MLAAttention(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.self_attn",
            )

        if (
            fd_config.model_config.n_routed_experts is not None
            and layer_id >= fd_config.model_config.first_k_dense_replace
        ):
            self.mlp = DeepSeekV3MoE(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepSeekV3MLP(
                fd_config=fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                prefix=f"{prefix}.mlp",
            )

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

    # def load_state_dict(self, state_dict):
    #     """ """
    #     self.self_attn.load_state_dict(state_dict)
    #     self.mlp.load_state_dict(state_dict)
    #     self.input_layernorm.load_state_dict(state_dict)
    #     self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """ """
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta
        )

        hidden_states = self.self_attn(forward_meta, hidden_states, position_ids, mask_encoder_batch)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_meta)
        return hidden_states, residual


@support_graph_optimization
class DeepSeekV3Model(nn.Layer):
    """
    DeepSeekV3Model
    """

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the DeepSeekV3Model class.
        """
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "deepseek_v3"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix="deepseek_v3.embed_tokens",
        )

        self.layers = nn.LayerList(
            [
                DeepSeekV3DecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix="deepseek_v3.norm",
        )

    # def load_state_dict(self, state_dict):
    #     """
    #     Load model parameters from a given state dictionary.
    #     """
    #     self.embed_tokens.load_state_dict(state_dict)
    #     self.norm.load_state_dict(state_dict)
    #     for i in range(self.num_layers):
    #         logger.info(f"Start load layer {i}")
    #         self.layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """ """
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
                position_ids,
                mask_encoder_batch,
            )
        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


@ModelRegistry.register_model_class(
    architecture="DeepseekV3ForCausalLM",
    module_name="deepseek_v3",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class DeepseekV3ForCausalLM(ModelForCasualLM):
    """
    DeepseekV3ForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super().__init__(fd_config)
        self.model = DeepSeekV3Model(fd_config)
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.position_ids_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens], dtype=paddle.int32
        )
        self.mask_encoder_batch_buffer = paddle.empty(
            [fd_config.scheduler_config.max_num_batched_tokens, 1], dtype=paddle.int32
        )

    @classmethod
    def name(cls):
        """ """
        return "DeepseekV3ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.
        """
        # self.model.load_state_dict(state_dict)
        # self.lm_head.load_state_dict(state_dict)

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        """
        Load model parameters from a given weights_iterator object.
        Args:
            weights_iterator (Iterator): An iterator yielding (name, weight) pairs.
        """
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("experts.gate_correction_bias", "gate.e_score_correction_bias", None),
            ("qkv_a_proj_with_mqa", "q_a_proj", "q_a"),
            ("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", "kv_a"),
        ]
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            loaded_weight_name = loaded_weight_name.replace("deepseek_v3", "model")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)

                if model_param_name not in params_dict:
                    continue

                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            if "kv_b_proj" in model_sublayer_name:
                kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                process_weights_after_loading_fn(kv_model_sublayer_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """ """
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def pre_process(self, forward_meta):
        """ """
        seq_lens_encoder = forward_meta.seq_lens_encoder
        seq_lens_decoder = forward_meta.seq_lens_decoder
        seq_lens_this_time = forward_meta.seq_lens_this_time

        current_total_tokens = forward_meta.ids_remove_padding.shape[0]
        position_ids = self.position_ids_buffer[:current_total_tokens]
        mask_encoder_batch = self.mask_encoder_batch_buffer[:current_total_tokens]

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            position_ids,
            mask_encoder_batch,
        )
        return position_ids, mask_encoder_batch

    def empty_input_forward(self, forward_meta):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[1, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.first_k_dense_replace,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.model.layers[i].mlp.experts(fake_hidden_states, self.model.layers[i].mlp.gate, forward_meta)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        position_ids, mask_encoder_batch = self.pre_process(forward_meta)
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            position_ids=position_ids,
            mask_encoder_batch=mask_encoder_batch,
        )
        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)


class DeepSeekV3PretrainedModel(PretrainedModel):
    """
    DeepSeekV3PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "DeepseekV3ForCausalLM"

    # @classmethod
    # def _get_tensor_parallel_mappings(cls, config, is_split=True):

    #     logger.info("DeepseekV3 inference model _get_tensor_parallel_mappings")

    #     from paddleformers.transformers.conversion_utils import split_or_merge_func

    #     fn = split_or_merge_func(
    #         is_split=is_split,
    #         tensor_model_parallel_size=config.tensor_model_parallel_size,
    #         tensor_parallel_rank=config.tensor_parallel_rank,
    #         num_attention_heads=config.num_attention_heads,
    #     )

    #     def get_tensor_parallel_split_mappings(num_layers):
    #         final_actions = {}

    #         base_actions = {
    #             "lm_head.weight": partial(fn, is_column=True),
    #             "embed_tokens.weight": partial(fn, is_column=False),
    #             "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
    #         }

    #         # Self Attention Layer which are need TP.
    #         base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.self_attn.q_b_proj.weight_scale_inv"] = partial(fn, is_column=True)
    #         base_actions["layers.0.self_attn.kv_b_proj.weight_scale_inv"] = partial(fn, is_column=True)

    #         # MLP Layer
    #         base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

    #         # Moe Layer
    #         for expert_idx in range(config.n_routed_experts):
    #             base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
    #             base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
    #             base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)

    #         # Shared Expert Layer
    #         base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

    #         # MTP parts
    #         base_actions["layers.61.embed_tokens.weight"] = partial(fn, is_column=False)
    #         base_actions["layers.61.eh_proj.weight"] = partial(fn, is_column=True)
    #         base_actions["layers.61.shared_head.head.weight"] = partial(fn, is_column=True)

    #         for key, action in base_actions.items():
    #             if "layers.0." in key:
    #                 for i in range(num_layers):
    #                     final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
    #             final_actions[key] = action

    #         return final_actions

    #     mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
    #     return mappings


@ModelRegistry.register_model_class(
    architecture="DeepseekV32ForCausalLM",
    module_name="deepseek_v32",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """
    DeepseekV32ForCausalLM
    """

    @classmethod
    def name(cls):
        return "DeepseekV32ForCausalLM"


class DeepSeekV32PretrainedModel(DeepSeekV3PretrainedModel):
    """
    DeepSeekV32PretrainedModel
    """

    @classmethod
    def arch_name(self):
        return "DeepseekV32ForCausalLM"
