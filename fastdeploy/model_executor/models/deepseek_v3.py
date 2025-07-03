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
from functools import partial

import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication_op import \
    tensor_model_parallel_all_reduce
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear, KVBatchLinear, MergedColumnParallelLinear,
    ReplicatedLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.rotary_embedding import \
    DeepseekScalingRotaryEmbedding
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.platforms import current_platform
from fastdeploy.worker.forward_meta import ForwardMeta

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import \
        get_position_ids_and_mask_encoder_batch


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

        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            input_size=fd_config.model_config.hidden_size,
            output_size=intermediate_size * 2,
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
            use_fast_ffn=True,
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

    def load_state_dict(self, state_dict):
        """
        """
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x):
        """
        """
        gate_up_out = self.gate_up_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class DeepSeekV3MoE(nn.Layer):
    """
    DeepSeekV3MoE, for MoE Layer.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_degree

        weight_key_map = {
            "gate_weight_key": f"{prefix}.gate.weight",
            "gate_correction_bias_key":
            f"{prefix}.gate.e_score_correction_bias",
            "ffn1_expert_weight_key":
            f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "ffn2_expert_weight_key":
            f"{prefix}.experts.{{}}.down_proj.weight",
        }

        self.fused_moe = FusedMoE(
            fd_config=fd_config,
            reduce_results=False,
            moe_intermediate_size=fd_config.model_config.deepseekv3.
            moe_intermediate_size,
            num_experts=fd_config.model_config.deepseekv3.n_routed_experts,
            top_k=fd_config.model_config.deepseekv3.num_experts_per_tok,
            topk_method=fd_config.model_config.deepseekv3.topk_method,
            topk_group=fd_config.model_config.deepseekv3.topk_group,
            n_group=fd_config.model_config.deepseekv3.n_group,
            routed_scaling_factor=fd_config.model_config.deepseekv3.
            routed_scaling_factor,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

        self.num_shared_experts = fd_config.model_config.deepseekv3.n_shared_experts
        shared_experts_intermediate_size = (
            self.num_shared_experts *
            fd_config.model_config.deepseekv3.moe_intermediate_size)

        self.shared_experts = DeepSeekV3MLP(
            fd_config=fd_config,
            intermediate_size=shared_experts_intermediate_size,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.fused_moe.load_state_dict(state_dict)
        self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        """
        """
        shared_experts_out = self.shared_experts(hidden_states)
        moe_out = self.fused_moe(hidden_states)
        moe_out = moe_out + shared_experts_out
        # We do to TP all reduce after the sum of experts.
        if self.tp_size > 1:
            tensor_model_parallel_all_reduce(moe_out)
        return moe_out


class DeepseekV3MLAAttention(nn.Layer):
    """
    DeepseekV3MLAAttention
    """

    def __init__(self,
                 fd_config: FDConfig,
                 layer_id: int,
                 prefix: str = "") -> None:
        super().__init__()

        self.tp_size = fd_config.parallel_config.tensor_parallel_degree
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size

        # MLA
        self.qk_nope_head_dim = fd_config.model_config.deepseekv3.qk_nope_head_dim
        self.qk_rope_head_dim = fd_config.model_config.deepseekv3.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = fd_config.model_config.deepseekv3.v_head_dim
        self.q_lora_rank = fd_config.model_config.deepseekv3.q_lora_rank
        self.kv_lora_rank = fd_config.model_config.deepseekv3.kv_lora_rank

        self.attn_softmax_scale = self.qk_head_dim**-0.5
        self.rope_theta = fd_config.model_config.rope_theta
        self.rms_norm_eps = fd_config.model_config.rms_norm_eps

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(fd_config=fd_config,
                                             prefix=f"{prefix}.q_a_proj",
                                             input_size=self.hidden_size,
                                             output_size=self.q_lora_rank,
                                             with_bias=False)

            self.q_a_layernorm = RMSNorm(fd_config,
                                         hidden_size=self.q_lora_rank,
                                         eps=self.rms_norm_eps,
                                         prefix=f"{prefix}.q_a_layernorm")

            self.q_b_proj = ColumnParallelLinear(
                fd_config=fd_config,
                prefix=f"{prefix}.q_b_proj",
                input_size=self.q_lora_rank,
                output_size=self.num_attention_heads * self.qk_head_dim,
                with_bias=False,
            )
        else:
            assert (self.q_lora_rank is not None
                    ), "self.q_lora_rank is None, Please Check your config."

        # 不切TP,跑 W4A16 Gemm
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
            input_size=self.hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            with_bias=False)

        self.kv_a_layernorm = RMSNorm(fd_config,
                                      hidden_size=self.kv_lora_rank,
                                      eps=self.rms_norm_eps,
                                      prefix=f"{prefix}.kv_a_layernorm")

        self.kv_b_proj = ColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.kv_b_proj",
            input_size=self.kv_lora_rank,
            output_size=self.num_attention_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            with_bias=False,
        )

        self.o_proj = RowParallelLinear(fd_config,
                                        prefix=f"{prefix}.o_proj",
                                        input_size=self.num_attention_heads *
                                        self.v_head_dim,
                                        output_size=self.hidden_size,
                                        with_bias=False)

        self.kv_b_proj_bmm = KVBatchLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.kv_b_proj",
            kv_lora_rank=self.kv_lora_rank,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim)

        self.rope_scaling = fd_config.model_config.deepseekv3.rope_scaling
        if self.rope_scaling:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", False)
            scaling_factor = self.rope_scaling["factor"]
            mscale = self.yarn_get_mscale(scaling_factor,
                                          float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        rope_scaling_kwargs = {
            key: self.rope_scaling[key]
            for key in [
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ] if key in self.rope_scaling
        }
        self.rope_scaling_factor = self.rope_scaling["factor"]
        self.rope_scaling_original_max_position_embeddings = self.rope_scaling[
            "original_max_position_embeddings"]
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.
            rope_scaling_original_max_position_embeddings,
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
        """
        """
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
        """
        """
        layernorm_out = hidden_states
        fmha_out = paddle.zeros(shape=[
            layernorm_out.shape[0],
            self.num_attention_heads_tp * self.v_head_dim
        ],
                                dtype=layernorm_out.dtype)

        decode_stage = forward_meta.is_decode_batch
        prefill_stage = not (forward_meta.is_decode_batch)

        if prefill_stage:
            query = self.q_a_proj(layernorm_out)
            query = self.q_a_layernorm(query)
            query = self.q_b_proj(query)

            query = query.reshape(
                [-1, self.num_attention_heads_tp, self.qk_head_dim])
            query_nope, query_pe = query.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

            compressed_kv = self.kv_a_proj_with_mqa(layernorm_out)
            compressed_kv, key_pe = compressed_kv.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
            key_pe = key_pe.reshape([-1, 1, self.qk_rope_head_dim])
            compressed_kv = self.kv_a_layernorm(compressed_kv)

            query_pe, key_pe = self.rotary_emb(position_ids, query_pe, key_pe)

            key_value = self.kv_b_proj(compressed_kv)
            key_value = key_value.reshape([
                -1, self.num_attention_heads_tp,
                self.qk_nope_head_dim + self.v_head_dim
            ])
            key_nope, value = key_value.split(
                [self.qk_nope_head_dim, self.v_head_dim], axis=-1)

            query[..., self.qk_nope_head_dim:] = query_pe
            key = paddle.empty_like(query)
            key[..., :self.qk_nope_head_dim] = key_nope
            key[..., self.qk_nope_head_dim:] = key_pe
            value = paddle.nn.functional.pad(
                value, [0, self.qk_head_dim - self.v_head_dim], value=0)

            fmha_out_prefill = self.mla_attn(q=query,
                                             k=key,
                                             v=value,
                                             qkv=None,
                                             compressed_kv=compressed_kv,
                                             k_pe=key_pe,
                                             forward_meta=forward_meta)

            fmha_out_prefill = fmha_out_prefill.reshape(
                [-1, self.num_attention_heads_tp, self.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, :self.v_head_dim]
            fmha_out_prefill = fmha_out_prefill.reshape(
                [-1, self.num_attention_heads_tp * self.v_head_dim])
            fmha_out_prefill = fmha_out_prefill * mask_encoder_batch.cast(
                fmha_out_prefill.dtype)

            fmha_out = fmha_out + fmha_out_prefill

        if decode_stage:
            query = self.q_a_proj(layernorm_out)
            query = self.q_a_layernorm(query)
            ln_out_or_q_c = query

            compressed_kv = self.kv_a_proj_with_mqa(layernorm_out)
            compressed_kv, key_pe = compressed_kv.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
            key_pe = key_pe.reshape([-1, 1, self.qk_rope_head_dim])
            compressed_kv = self.kv_a_layernorm(compressed_kv)

            query = self.q_b_proj(ln_out_or_q_c)
            query = query.reshape(
                [-1, self.num_attention_heads_tp, self.qk_head_dim])

            query_nope, query_pe = query.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)
            query_pe, key_pe = self.rotary_emb(position_ids, query_pe, key_pe)

            q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]),
                                            proj_type='k').transpose([1, 0, 2])

            q_input = paddle.concat([q_nope_out, query_pe], axis=-1)
            q_input = q_input.reshape([
                -1,
                self.num_attention_heads_tp *
                (self.kv_lora_rank + self.qk_rope_head_dim),
            ])
            fmha_out_decode = self.mla_attn(q=q_input,
                                            k=None,
                                            v=None,
                                            qkv=None,
                                            compressed_kv=compressed_kv,
                                            k_pe=key_pe,
                                            forward_meta=forward_meta)

            fmha_out_decode = fmha_out_decode.reshape(
                [-1, self.num_attention_heads_tp,
                 self.kv_lora_rank]).transpose([1, 0, 2])

            fmha_out_decode = (self.kv_b_proj_bmm(
                fmha_out_decode, proj_type='v').transpose([1, 0, 2]).reshape(
                    [-1, self.num_attention_heads_tp * self.v_head_dim]))
            fmha_out = fmha_out + fmha_out_decode

        output = self.o_proj(fmha_out)
        return output

    def load_state_dict(self, state_dict):
        """
        """
        self.q_a_proj.load_state_dict(state_dict)
        self.q_a_layernorm.load_state_dict(state_dict)
        self.kv_a_proj_with_mqa.load_state_dict(state_dict)
        self.kv_a_layernorm.load_state_dict(state_dict)
        self.q_b_proj.load_state_dict(state_dict)
        self.kv_b_proj_bmm.load_state_dict(state_dict)
        self.kv_b_proj.load_state_dict(state_dict)
        # NOTE(Ryan):Make sure kv_b_proj_bmm loaded before kv_b_proj,
        # The same weight key will be poped after kv_b_proj.
        self.o_proj.load_state_dict(state_dict)


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
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = DeepseekV3MLAAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (fd_config.model_config.deepseekv3.n_routed_experts is not None
                and layer_id
                >= fd_config.model_config.deepseekv3.first_k_dense_replace):
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
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        """
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(forward_meta, hidden_states,
                                       position_ids, mask_encoder_batch)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


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
        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "deepseek_v3"

        self.embeddings = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix="deepseek_v3.embed_tokens",
        )

        self.decoder_layers = nn.LayerList([
            DeepSeekV3DecoderLayer(
                fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ])

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix="deepseek_v3.norm",
        )

    def pre_process(self, forward_meta):
        """
        """
        seq_lens_encoder = forward_meta.seq_lens_encoder
        seq_lens_decoder = forward_meta.seq_lens_decoder
        seq_lens_this_time = forward_meta.seq_lens_this_time
        position_ids_shape = paddle.sum(seq_lens_this_time)

        position_ids = paddle.empty(shape=position_ids_shape,
                                    dtype=seq_lens_encoder.dtype)
        mask_encoder_batch = paddle.empty(
            shape=position_ids_shape,
            dtype=seq_lens_encoder.dtype).unsqueeze(1)

        get_position_ids_and_mask_encoder_batch(seq_lens_encoder,
                                                seq_lens_decoder,
                                                seq_lens_this_time,
                                                position_ids,
                                                mask_encoder_batch)

        return position_ids, mask_encoder_batch

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.
        """
        self.embeddings.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.decoder_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        """
        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        position_ids, mask_encoder_batch = self.pre_process(forward_meta)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.decoder_layers[i](
                forward_meta, hidden_states, residual, position_ids,
                mask_encoder_batch)
        hidden_states = hidden_states + residual
        out = self.norm(hidden_states)

        return out


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

    @classmethod
    def name(cls):
        """
        """
        return "DeepseekV3ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.
        """
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        """
        """
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")
        return logits

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        """
        hidden_states = self.model(ids_remove_padding, forward_meta)
        return hidden_states


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
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        logger.info("DeepseekV3 inference model _get_tensor_parallel_mappings")

        from paddleformers.transformers.conversion_utils import \
            split_or_merge_func
        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn,
                                                            is_column=False),
            }

            # Self Attention Layer which are need TP.
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(
                fn, is_column=True)
            base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(
                fn, is_column=True)
            base_actions[
                "layers.0.self_attn.q_b_proj.weight_scale_inv"] = partial(
                    fn, is_column=True)
            base_actions[
                "layers.0.self_attn.kv_b_proj.weight_scale_inv"] = partial(
                    fn, is_column=True)

            # MLP Layer
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(
                fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(
                fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(
                fn, is_column=False)

            # Moe Layer
            for expert_idx in range(config.n_routed_experts):
                base_actions[
                    f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(
                        fn, is_column=True)
                base_actions[
                    f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(
                        fn, is_column=True)
                base_actions[
                    f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(
                        fn, is_column=False)

            # Shared Expert Layer
            base_actions[
                "layers.0.mlp.shared_experts.up_proj.weight"] = partial(
                    fn, is_column=True)
            base_actions[
                "layers.0.mlp.shared_experts.gate_proj.weight"] = partial(
                    fn, is_column=True)
            base_actions[
                "layers.0.mlp.shared_experts.down_proj.weight"] = partial(
                    fn, is_column=False)

            # MTP parts
            base_actions["layers.61.embed_tokens.weight"] = partial(
                fn, is_column=False)
            base_actions["layers.61.eh_proj.weight"] = partial(fn,
                                                               is_column=True)
            base_actions["layers.61.shared_head.head.weight"] = partial(
                fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.",
                                                  f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_layers)
        return mappings
