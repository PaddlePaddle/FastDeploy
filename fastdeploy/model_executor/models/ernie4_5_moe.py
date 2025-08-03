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

from functools import partial
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
from fastdeploy.model_executor.models.utils import WeightMeta


class Ernie4_5_MLP(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.nranks = fd_config.parallel_config.tensor_parallel_size
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

    def load_state_dict(self, state_dict):
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        gate_up_out = self.up_gate_proj(hidden_states)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Ernie4_5_MoE(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, "moe_quant_type"):
            moe_quant_type = fd_config.quant_config.moe_quant_type

        if moe_quant_type == "w4a8":
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        elif moe_quant_type == "w4w2":
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_super_scales_key": f"{prefix}.experts.{{}}.up_gate_proj.super_scales",
                "down_proj_expert_super_scales_key": f"{prefix}.experts.{{}}.down_proj.super_scales",
                "up_gate_proj_expert_code_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.code_scale",
                "down_proj_expert_code_scale_key": f"{prefix}.experts.{{}}.down_proj.code_scale",
                "up_gate_proj_expert_code_zp_key": f"{prefix}.experts.{{}}.up_gate_proj.code_zp",
                "down_proj_expert_code_zp_key": f"{prefix}.experts.{{}}.down_proj.code_zp",
            }
        elif moe_quant_type == "tensor_wise_fp8" or (
            moe_quant_type == "block_wise_fp8" and fd_config.model_config.is_quantized
        ):
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.quant_weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.quant_weight",
                "up_gate_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.weight_scale",
                "down_proj_expert_weight_scale_key": f"{prefix}.experts.{{}}.down_proj.weight_scale",
                "up_gate_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.up_gate_proj.activation_scale",
                "down_proj_expert_in_scale_key": f"{prefix}.experts.{{}}.down_proj.activation_scale",
            }
        else:
            weight_key_map = {
                "gate_weight_key": f"{prefix}.gate.weight",
                "gate_correction_bias_key": f"{prefix}.moe_statics.e_score_correction_bias",
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            }

        self.fused_moe = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.moe_num_experts,
            top_k=fd_config.model_config.moe_k,
            layer_idx=layer_id,
            weight_key_map=weight_key_map,
        )

        self.num_shared_experts = fd_config.model_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            shared_experts_hidden_dim = self.num_shared_experts * fd_config.model_config.moe_intermediate_size
            self.shared_experts = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_hidden_dim,
                prefix=f"{prefix}.shared_experts",
            )

    def load_state_dict(self, state_dict):
        self.fused_moe.load_state_dict(state_dict)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        out = self.fused_moe(hidden_states)
        if self.num_shared_experts > 0:
            s_x = self.shared_experts(hidden_states)
            out = out + s_x
        return out


class Ernie4_5_Attention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
        )
        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
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
        qkv_out = self.qkv_proj(hidden_states)

        attn_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )

        output = self.o_proj(attn_out)

        return output


class Ernie4_5_DecoderLayer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        self.self_attn = Ernie4_5_Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (
            getattr(fd_config.model_config, "moe_num_experts", None) is not None
            and layer_id >= fd_config.model_config.moe_layer_start_index
        ):
            self.mlp = Ernie4_5_MoE(
                fd_config=fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie4_5_MLP(
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
        self.fd_config = fd_config

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward_old(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


    def forward_attn(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):  
        hidden_states, residual = self.input_layernorm(hidden_states, residual)


        forward_meta.attn_backend.init_attention_metadata(forward_meta)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_out = paddle.matmul(hidden_states.cast("float32"), self.mlp.fused_moe.gate_weight)

        topk_idx, topk_weights = self.mlp.fused_moe.quant_method.ep_decoder_runner.moe_select(self.mlp.fused_moe, gate_out)

        return hidden_states, residual, topk_idx, topk_weights

    def m2n_dispatch(self, x, topk_idx, topk_weights):
        permute_input, token_nums_per_expert, handle = self.mlp.fused_moe.quant_method.ep_decoder_runner.dispatch(x, topk_idx, topk_weights)

        return permute_input, token_nums_per_expert, handle


    def compute_moe_ffn(
        self,
        permute_input: paddle.Tensor,
        token_nums_per_expert):
        ffn_out = self.mlp.fused_moe.quant_method.compute_ffn(
            self.mlp.fused_moe,
            permute_input,
            token_nums_per_expert,
            None,
            True,
        )

        return ffn_out

    def m2n_combine(self, ffn_out, topk_idx, topk_weights, handle):
        res = self.mlp.fused_moe.quant_method.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)
        return res

    def forward(self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None):

        IsH20 = self.fd_config.parallel_config.is_H20

        if IsH20:
            hidden_states, residual, topk_idx, topk_weights = self.forward_attn(forward_meta, hidden_states, residual)
        else:
            hidden_states = None
            residual = None
            topk_idx = None
            topk_weights = None

        permute_input, token_nums_per_expert, handle = self.m2n_dispatch(hidden_states, topk_idx, topk_weights)

        if IsH20:
            ffn_out = None
        else:
            ffn_out = self.compute_moe_ffn(permute_input, token_nums_per_expert)
        
        hidden_states = self.m2n_combine(ffn_out, topk_idx, topk_weights, handle)
        
        return hidden_states, residual

@support_graph_optimization
class Ernie4_5_Model(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie4_5_Model class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "ernie"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype(),
            prefix=(f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens"),
        )

        self.layers = nn.LayerList(
            [
                Ernie4_5_DecoderLayer(
                    fd_config=fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

        self.fd_config = fd_config

    def load_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):

        IsH20 = self.fd_config.parallel_config.is_H20
        all_hidden_states = []
        forward_metas = []
        all_residual = []
        bs = self.fd_config.parallel_config.max_num_seqs
        # 暂时设置成1
        mc_bs = bs // 1
        if IsH20:
            # H20 机器
            hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
            residual = None
            for i in range(3):
                hidden_states, residual = self.layers[i].forward_old(forward_meta, hidden_states, residual)

            for i in range(0, bs, mc_bs):
                from copy import copy
                forward_meta_copy = copy(forward_meta)
                
                start_bs = i
                end_bs = i + mc_bs
                end_bs = min(end_bs, bs)

                start_token_id = forward_meta.cu_seqlens_q[start_bs].item()
                end_token_id =   forward_meta.cu_seqlens_q[end_bs].item()

                if end_token_id == start_token_id:
                    # 这个microbatch是空的，不需要处理!!
                    continue

                forward_meta_copy.seq_lens_encoder = forward_meta.seq_lens_encoder[start_bs:end_bs]
                forward_meta_copy.seq_lens_decoder = forward_meta.seq_lens_decoder[start_bs:end_bs]
                forward_meta_copy.seq_lens_this_time = forward_meta.seq_lens_this_time[start_bs:end_bs]

                forward_meta_copy.batch_id_per_token = forward_meta.batch_id_per_token[start_token_id:end_token_id] - start_bs
                forward_meta_copy.cu_seqlens_q = forward_meta.cu_seqlens_q[start_bs:end_bs+1] - start_token_id
                forward_meta_copy.cu_seqlens_k = forward_meta.cu_seqlens_k[start_bs:end_bs+1] - start_token_id

                forward_meta_copy.block_tables = forward_meta.block_tables[start_bs:end_bs]

                forward_metas.append(forward_meta_copy)
                all_hidden_states.append(hidden_states[start_token_id:end_token_id])
                all_residual.append(residual[start_token_id:end_token_id])
        else:
            # MoE 机器啥也不需要做！
            for i in range(0, bs, mc_bs):
                forward_metas.append(None)
                all_hidden_states.append(None)
                all_residual.append(None)
        
        paddle.distributed.barrier()

        # 下面第3层开始H20和MoE机器！！一起跑啦！
        for mc_id, mc_forward_meta in enumerate(forward_metas):
            hidden_states = all_hidden_states[mc_id]
            residual = all_residual[mc_id]
            mc_forward_meta = forward_metas[mc_id]

            for i in range(3, self.num_layers):
                hidden_states, residual = self.layers[i](mc_forward_meta,
                                                                hidden_states,
                                                                residual)

            if IsH20:
                hidden_states = hidden_states + residual
                all_hidden_states[mc_id] = hidden_states
        
        if IsH20:
            hidden_states = paddle.concat(all_hidden_states, axis=0)
            out = self.norm(hidden_states)
            return out
        else:
            # MoE机器返回None
            return None

    def forward1(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):  

        IsH20 = self.fd_config.parallel_config.is_H20
        hidden_states = None
        if IsH20:
            hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
            forward_meta.attn_backend.init_attention_metadata(forward_meta)
        residual = None

        if IsH20:
            hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
            forward_meta.attn_backend.init_attention_metadata(forward_meta)
            for i in range(3):
                hidden_states, residual = self.layers[i].forward_old(forward_meta, hidden_states, residual)

        for i in range(3, self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        if IsH20:
            hidden_states = hidden_states + residual
            out = self.norm(hidden_states)
            return out
        else:
            return None


class Ernie4_5_MoeForCausalLM(ModelForCasualLM):
    """
    Ernie4_5_MoeForCausalLM
    """

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(Ernie4_5_MoeForCausalLM, self).__init__(fd_config)
        self.fd_config = fd_config
        self.ernie = Ernie4_5_Model(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings

        self.ii = 0

    @classmethod
    def name(self):
        return "Ernie4_5_MoeForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.ernie.load_state_dict(state_dict)
        if self.tie_word_embeddings:
            self.lm_head.linear.weight.set_value(self.ernie.embed_tokens.embeddings.weight.transpose([1, 0]))
        else:
            self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def empty_input_forward(self):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[0, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.ernie.layers[i].mlp.fused_moe(fake_hidden_states)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):  
        self.ii += 1
        
        if self.ii == 1:
            from paddle.framework import core
            core.nvprof_start()
        
        if ids_remove_padding is not None:
            print(ids_remove_padding.shape)
            print(forward_meta.seq_lens_decoder)
        
        hidden_states = self.ernie(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        
        if self.ii == 5:
            from paddle.framework import core
            core.nvprof_stop()

        return hidden_states


class Ernie4_5_ForCausalLM(Ernie4_5_MoeForCausalLM):
    """
    Ernie4_5_ForCausalLM
    """

    @classmethod
    def name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5_ForCausalLM"


class Ernie4_5_PretrainedModel(PretrainedModel):
    """
    Ernie4_5_PretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    weight_infos = [
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.weight", False),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.weight",
            False,
        ),
        WeightMeta(".embed_tokens.weight", False),
        WeightMeta("lm_head.weight", True),
        # quant tensorwise
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.qkv_proj.quant_weight",
            True,
            tsm.GQA,
        ),
        WeightMeta(
            f".layers.{{{layerid.LAYER_ID}}}.self_attn.o_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.FFN_LAYER_ID}}}.mlp.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.experts.{{{layerid.EXPERT_ID}}}.down_proj.quant_weight",
            False,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.up_gate_proj.quant_weight",
            True,
            tsm.PairFused,
        ),
        WeightMeta(
            f".layers.{{{layerid.MOE_LAYER_ID}}}.mlp.shared_experts.down_proj.quant_weight",
            False,
        ),
    ]

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: PretrainedConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("erine inference model _get_tensor_parallel_mappings")
        from fastdeploy.model_executor.models.tp_utils import (
            build_expanded_keys,
            has_prefix,
            split_or_merge_func_v1,
        )

        fn = split_or_merge_func_v1(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )

        def get_tensor_parallel_split_mappings(num_layers, moe_num_experts, moe_layer_start_index, prefix_name):
            base_actions = {}
            weight_infos = cls.weight_infos
            for weight_name, is_column, extra in weight_infos:
                params = {
                    "is_column": is_column,
                    **({extra.value: True} if extra else {}),
                }

                if "lm_head.weight" in weight_name:
                    key = weight_name
                elif not has_prefix(prefix_name, weight_name):
                    key = f"{prefix_name}{weight_name}"
                else:
                    key = weight_name
                base_actions[key] = partial(fn, **params)
            final_actions = {}
            start_layer = moe_layer_start_index if moe_layer_start_index > 0 else num_layers
            final_actions = build_expanded_keys(base_actions, num_layers, start_layer, moe_num_experts)
            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_hidden_layers,
            getattr(config, "moe_num_experts", 0),
            getattr(config, "moe_layer_start_index", -1),
            config.prefix_name,
        )
        return mappings
