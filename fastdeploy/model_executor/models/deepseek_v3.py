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
from typing import Dict

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
    KVBatchLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import LayerNorm, RMSNorm
from fastdeploy.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
)
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import (
        get_position_ids_and_mask_encoder_batch,
    )

from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    per_token_group_quant_fp8,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        cp_gather_indexer_k_quant_cache,
        indexer_k_quant_and_cache,
        merge_prefill_decode_output,
        radix_topk_ragged_transform,
    )


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
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.attn_tp_size = fd_config.parallel_config.tensor_parallel_size
        if self.ep_size > 1:
            self.tp_size = 1
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
            hidden_size=fd_config.model_config.hidden_size,
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

    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        """ """
        shared_experts_out = self.shared_experts(hidden_states)

        if self.attn_tp_size > 1 and self.ep_size > 1:
            shared_experts_out = tensor_model_parallel_all_reduce(shared_experts_out)

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

        self.fd_config = fd_config
        self.use_gated_attn = getattr(self.fd_config.model_config, "use_gated_attn", False)
        self.use_bias = getattr(self.fd_config.model_config, "use_bias", False)
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
        if self.use_gated_attn:
            self.gate = ReplicatedLinear(
                fd_config=fd_config,
                prefix=f"{prefix}.gate",
                input_size=self.hidden_size,
                output_size=self.num_attention_heads * self.v_head_dim,
                with_bias=self.use_bias,
            )
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
            with_bias=self.use_bias,
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
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling and "factor" in self.rope_scaling:
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
        else:
            # Default rope without scaling
            # The current `max_model_len` can cover the maximum context length.
            max_position_embeddings = getattr(fd_config.model_config, "max_model_len", 8192)
            self.rotary_emb = DeepseekScalingRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=self.rope_theta,
                scaling_factor=1.0,
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
    ):
        """ """

        attn_out = None
        if self.use_gated_attn:
            gate_out = self.gate(hidden_states)

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
        query_pe, key_pe = self.rotary_emb(forward_meta.position_ids, query_pe, key_pe)

        compressed_kv = self.kv_a_layernorm(compressed_kv)[0]

        need_do_prefill = forward_meta.max_len_tensor_cpu[1] > 0
        need_do_decode = forward_meta.max_len_tensor_cpu[2] > 0

        if need_do_prefill:  # max_enc_len_this_time
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

            fmha_out = self.mla_attn(
                q=query,
                k=key,
                v=value,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmha_out.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
            fmha_out = fmha_out[:, :, : self.v_head_dim]
            fmha_out.reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            attn_out = fmha_out

        if need_do_decode:  # max_dec_len_this_time
            q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]), proj_type="k").transpose([1, 0, 2])

            q_input = paddle.concat([q_nope_out, query_pe], axis=-1)
            q_input.reshape_(
                [
                    -1,
                    self.num_attention_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )

            fmqa_out = self.mla_attn(
                q=q_input,
                k=None,
                v=None,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmqa_out = fmqa_out.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose([1, 0, 2])

            fmqa_out = (
                self.kv_b_proj_bmm(fmqa_out, proj_type="v")
                .transpose([1, 0, 2])
                .reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            )

            if need_do_prefill:
                merge_prefill_decode_output(
                    attn_out,
                    fmqa_out,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.cu_seqlens_q,
                    self.num_attention_heads_tp,
                    self.v_head_dim,
                    1,
                )
            else:
                attn_out = fmqa_out
        if self.use_gated_attn:
            gated_attn_act = getattr(self.fd_config.model_config, "gated_attn_act", "sigmoid")
            if gated_attn_act == "sigmoid":
                attn_out = attn_out * F.sigmoid(gate_out)
            elif gated_attn_act == "scaled_softsign":
                attn_out = attn_out * ((F.softsign(gate_out) + 1.0) / 2.0)
            else:
                raise NotImplementedError(f"{gated_attn_act} not implemented")
        output = self.o_proj(attn_out)
        return output


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


import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit()
def extract_kernel(
    q,
    weight,
    cu_seqlens_q,
    seq_lens_encoder,
    seq_lens_decoder,
    output,
    out_weight,
    cache_seqlens,
    HIDDEN_DIM: tl.constexpr,
    WEIGHT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(axis=0)
    cache_kv_len = tl.load(seq_lens_decoder + batch_id)

    # 这个batch不是decoder，所以不需要动弹
    if cache_kv_len <= 0:
        return

    cu_len_this_batch = tl.load(cu_seqlens_q + batch_id)

    read_offsets = tl.arange(0, BLOCK_SIZE)
    read_weight_offsets = tl.arange(0, WEIGHT_DIM)

    q += cu_len_this_batch * HIDDEN_DIM
    weight += cu_len_this_batch * WEIGHT_DIM

    row_data = tl.load(q + read_offsets, mask=read_offsets < HIDDEN_DIM)
    weight_row_data = tl.load(weight + read_weight_offsets, mask=read_weight_offsets < WEIGHT_DIM)

    output += batch_id * HIDDEN_DIM
    out_weight += batch_id * WEIGHT_DIM

    tl.store(output + read_offsets, row_data, mask=read_offsets < HIDDEN_DIM)
    tl.store(out_weight + read_weight_offsets, weight_row_data, mask=read_weight_offsets < WEIGHT_DIM)

    tl.store(cache_seqlens + batch_id, cache_kv_len + 1)


def extract_decoder_token_from_q(
    q: paddle.Tensor,
    weight: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
):
    assert len(q.shape) == 2
    assert len(weight.shape) == 2
    assert len(cu_seqlens_q.shape) == 1
    assert len(seq_lens_encoder.shape) == 1
    assert len(seq_lens_decoder.shape) == 1

    max_bsz = seq_lens_decoder.shape[0]

    hidden_dim = q.shape[-1]
    weight_dim = weight.shape[-1]

    # if q.shape[0] <= max_bsz:
    #     max_bsz = q.shape[0]
    out = paddle.zeros([max_bsz, hidden_dim], dtype=q.dtype)
    out_weight = paddle.zeros([max_bsz, weight_dim], dtype=weight.dtype)

    cache_seqlens = paddle.zeros_like(seq_lens_decoder)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    grid = (max_bsz,)

    extract_kernel[grid](
        q,
        weight,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        out,
        out_weight,
        cache_seqlens,
        hidden_dim,
        weight_dim,
        BLOCK_SIZE,
    )

    return out, out_weight, cache_seqlens


class Indexer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.max_model_len = fd_config.model_config.max_model_len

        self.index_head_dim = fd_config.model_config.index_head_dim
        self.index_n_heads = fd_config.model_config.index_n_heads
        self.index_topk = fd_config.model_config.index_topk

        self.rope_dim = fd_config.model_config.qk_rope_head_dim  # 64
        self.q_lora_rank = fd_config.model_config.q_lora_rank  # 1536
        self.hidden_size = fd_config.model_config.hidden_size

        self.wq_b = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.wq_b",
            input_size=self.q_lora_rank,
            output_size=self.index_head_dim * self.index_n_heads,
            with_bias=False,
        )
        self.wk = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.wk",
            input_size=self.hidden_size,
            output_size=self.index_head_dim,
            with_bias=False,
        )
        self.k_norm = LayerNorm(
            fd_config=fd_config,
            hidden_size=self.index_head_dim,
            eps=1e-6,
            prefix=f"{prefix}.k_norm",
            with_bias=True,
        )

        self.weights_proj = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.weights_proj",
            input_size=self.hidden_size,
            output_size=self.index_n_heads,
            with_bias=False,
        )

        self.softmax_scale = self.index_head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config

        self.offsets = paddle.zeros([self.max_model_len], dtype="int32")
        self.lengths = paddle.zeros([self.max_model_len], dtype="int32")
        # self.buffer = paddle.zeros([2048 * 2048], dtype=paddle.uint8)

    def forward(
        self, forward_meta: ForwardMeta, hidden_states: paddle.Tensor, qr: paddle.Tensor, rotary_emb
    ) -> paddle.Tensor:
        self.indexer_cache = forward_meta.caches[2 * self.layer_id + 1]

        q = self.wq_b(qr)
        q = q.reshape([-1, self.index_n_heads, self.index_head_dim])
        q_pe, q_nope = paddle.split(q, [self.rope_dim, self.index_head_dim - self.rope_dim], axis=-1)

        k = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = paddle.split(k, [self.rope_dim, self.index_head_dim - self.rope_dim], axis=-1)

        q_pe, k_pe = rotary_emb(forward_meta.position_ids, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.reshape(-1, self.index_n_heads, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # [num_tokens, n_head, rope_dim].
        q = paddle.concat([q_pe, q_nope], axis=-1).reshape([-1, self.index_head_dim])
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = paddle.concat([k_pe.squeeze(-2), k_nope], axis=-1)

        # indexer q_quant
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )

        q_fp8 = q_fp8.reshape([-1, self.index_n_heads, self.index_head_dim])
        q_scale = q_scale.reshape([-1, self.index_n_heads, 1])

        weights = self.weights_proj(hidden_states)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.index_n_heads**-0.5
        weights = weights.squeeze(-1)

        slot_mapping = compute_slot_mapping(
            forward_meta.block_tables,
            forward_meta.position_ids,
            forward_meta.batch_id_per_token,
            64,
        )

        indexer_top_k = paddle.full([q_fp8.shape[0], self.index_topk], -1, dtype="int32")

        # indexer write_cache
        indexer_k_quant_and_cache(k, self.indexer_cache, slot_mapping, self.quant_block_size, self.scale_fmt)

        from fastdeploy.model_executor.layers.quantization.fp8_utils import deep_gemm

        if forward_meta.max_len_tensor_cpu[1]:

            # indexer_prefill read_cache
            k_fp8_cache = paddle.zeros_like(k, dtype=paddle.uint8)
            k_scale_cache = paddle.zeros([k.shape[0], 4], dtype=paddle.float32)
            cp_gather_indexer_k_quant_cache(
                self.indexer_cache, k_fp8_cache, k_scale_cache, forward_meta.block_tables, forward_meta.cu_seqlens_k
            )

            k_scale_cache_real = k_scale_cache.flatten()[: k.shape[0]].contiguous()
            k_cache = k_fp8_cache.view(paddle.float8_e4m3fn), k_scale_cache_real

            # TODO(changwenbin): Constructed using maskoffset
            # ks,ke = forward_meta.attn_mask_offsets[::2].contiguous(),forward_meta.attn_mask_offsets[1::2].contiguous()
            num_tokens = q_fp8.shape[0]
            ks = paddle.zeros(num_tokens, dtype=paddle.int32)
            ke = paddle.zeros(num_tokens, dtype=paddle.int32)

            bsz = forward_meta.seq_lens_this_time.shape[0]
            for i in range(bsz):
                if forward_meta.seq_lens_encoder[i] > 0:
                    token_start_k = forward_meta.cu_seqlens_k[i]
                    token_end_k = forward_meta.cu_seqlens_k[i + 1]
                    ks[token_start_k:token_end_k] = forward_meta.cu_seqlens_k[i]
                    ke[token_start_k:token_end_k] = paddle.arange(token_start_k, token_end_k, dtype=paddle.int32) + 1

            max_seqlen_k = (ke - ks).max().item()

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8, k_cache, weights, ks, ke, max_seqlen_k=max_seqlen_k, clean_logits=False
            ).contiguous()

            radix_topk_ragged_transform(
                logits,
                indexer_top_k,
                ks,  # self.offsets,# 初始K方向偏移，
                ke - ks,  # self.lengths,# 表明当前q 关注的k有多长;
                None,  # forward_meta.seq_lens_decoder,
                None,  # forward_meta.batch_id_per_token,
                None,
                None,  # self.buffer
                0,
                self.index_topk,
                1,
            )

        if forward_meta.max_len_tensor_cpu[2]:

            decoder_q, decoder_weight, cache_seqlens = extract_decoder_token_from_q(
                q_fp8.reshape(-1, self.index_n_heads * self.index_head_dim),
                weights,
                forward_meta.cu_seqlens_q,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
            )

            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(cache_seqlens, 64, deep_gemm.get_num_sms())

            logits = deep_gemm.fp8_paged_mqa_logits(
                decoder_q.reshape(-1, 1, self.index_n_heads, self.index_head_dim),
                self.indexer_cache.unsqueeze(2),
                decoder_weight,
                cache_seqlens,
                forward_meta.block_tables,
                schedule_metadata,
                self.max_model_len,
                clean_logits=True,
            ).contiguous()

            radix_topk_ragged_transform(
                logits,
                indexer_top_k,
                forward_meta.cu_seqlens_q,
                self.lengths,  # unused
                cache_seqlens,
                forward_meta.batch_id_per_token,
                forward_meta.block_tables,
                None,  # self.buffer
                forward_meta.block_tables.shape[1],
                self.index_topk,
                1,  # kv_head
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
        if fd_config.model_config.model_type == "glm_moe_dsa":
            self.rope_theta = fd_config.model_config.rope_parameters["rope_theta"]

        self.rms_norm_eps = fd_config.model_config.rms_norm_eps

        assert self.q_lora_rank is not None, "self.q_lora_rank is None, Please Check your config."

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
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling and "factor" in self.rope_scaling:
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
        else:
            # Default rope without scaling
            # The current `max_model_len` can cover the maximum context length.
            max_position_embeddings = getattr(fd_config.model_config, "max_model_len", 8192)
            self.rotary_emb = DeepseekScalingRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=self.rope_theta,
                scaling_factor=1.0,
            )
            self.indexer_rotary_emb = DeepseekScalingRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                base=self.rope_theta,
                scaling_factor=1.0,
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
    ):
        """ """
        qkv_a_out = self.qkv_a_proj_with_mqa(hidden_states)

        query, compressed_kv, key_pe = qkv_a_out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], axis=-1
        )
        key_pe.reshape_([-1, 1, self.qk_rope_head_dim])

        query = self.q_a_layernorm(query)[0]

        # DSA indexer
        indexer_top_k = self.indexer(forward_meta, hidden_states, query, rotary_emb=self.indexer_rotary_emb)

        query = self.q_b_proj(query)
        query.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
        query_nope, query_pe = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        query_pe, key_pe = self.rotary_emb(forward_meta.position_ids, query_pe, key_pe)
        q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]).contiguous(), proj_type="k")
        q_input = paddle.concat([q_nope_out.transpose([1, 0, 2]).contiguous(), query_pe], axis=-1)

        compressed_kv = self.kv_a_layernorm(compressed_kv)[0]
        kv = paddle.concat([compressed_kv, key_pe.squeeze(1)], axis=-1)

        # dsa attention
        fmha_out = self.dsa_attn(
            q=q_input.contiguous(),
            k=kv.unsqueeze(1).contiguous(),
            v=indexer_top_k.unsqueeze(1).contiguous(),
            qkv=None,
            compressed_kv=compressed_kv,
            k_pe=key_pe,
            forward_meta=forward_meta,
        )

        fmha_out = fmha_out.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose([1, 0, 2])
        fmha_out = (
            self.kv_b_proj_bmm(
                fmha_out,
                proj_type="v",
            )
            .transpose([1, 0, 2])
            .reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
        )

        output = self.o_proj(fmha_out)

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

        if fd_config.model_config.model_type in ["deepseek_v32", "glm_moe_dsa"]:
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

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor,
    ):
        """ """
        if hidden_states.shape[0] > 0:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual_input=residual, forward_meta=forward_meta
            )

            hidden_states = self.self_attn(forward_meta, hidden_states)

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
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

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](
                forward_meta,
                hidden_states,
                residual,
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
        pass

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

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
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
        inputs: Dict,
        forward_meta: ForwardMeta,
    ):
        ids_remove_padding = inputs["ids_remove_padding"]
        forward_meta.position_ids, forward_meta.mask_encoder_batch = self.pre_process(forward_meta)
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
        )
        return hidden_states

    def clear_graph_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.model.clear_graph_opt_backend(fd_config=self.fd_config)


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


@ModelRegistry.register_model_class(
    architecture="DeepseekV32ForCausalLM",
    module_name="deepseek_v3",  # TODO(changwenbin): trick using the current dsk-v3 model
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


@ModelRegistry.register_model_class(
    architecture="Glm4MoeLiteForCausalLM",
    module_name="deepseek_v3",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Glm4MoeLiteForCausalLM(DeepseekV3ForCausalLM):
    """
    Glm4MoeLiteForCausalLM
    """

    @classmethod
    def name(cls):
        return "Glm4MoeLiteForCausalLM"


class Glm4MoeLitePretrainedModel(DeepSeekV3PretrainedModel):
    """
    Glm4MoeLite
    """

    @classmethod
    def arch_name(self):
        return "Glm4MoeLiteForCausalLM"
