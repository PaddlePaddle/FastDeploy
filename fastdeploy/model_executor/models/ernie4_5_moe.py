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

import inspect
import re
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
from fastdeploy.model_executor.ops.gpu import deep_gemm
import paddle.device.cuda.graphs as graphs
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.model_base import (
    ModelCategory,
    ModelForCasualLM,
    ModelRegistry,
)
from fastdeploy.model_executor.models.tp_utils import TensorSplitMode as tsm
from fastdeploy.model_executor.models.utils import LayerIdPlaceholder as layerid
from fastdeploy.model_executor.models.utils import WeightMeta
from fastdeploy.platforms import current_platform
from fastdeploy.worker.experts_manager import RedundantExpertManger


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
    def __init__(
        self, fd_config: FDConfig, layer_id: int, prefix: str, redundant_table_manger: RedundantExpertManger = None
    ) -> None:
        super().__init__()
        moe_quant_type = ""
        if hasattr(fd_config.quant_config, "moe_quant_type"):
            moe_quant_type = fd_config.quant_config.moe_quant_type

        self.expert_parallel_size = fd_config.parallel_config.expert_parallel_size
        self.tensor_parallel_size = fd_config.parallel_config.tensor_parallel_size
        self.tensor_parallel_rank = fd_config.parallel_config.tensor_parallel_rank
        self.tp_group = fd_config.parallel_config.tp_group

        self.use_ep = self.expert_parallel_size > 1
        self.use_tp = self.tensor_parallel_size > 1

        if moe_quant_type == "w4a8" or moe_quant_type == "w4afp8":
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

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=fd_config.model_config.hidden_size,
            output_size=fd_config.model_config.moe_num_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        self.experts = FusedMoE(
            fd_config=fd_config,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.moe_num_experts,
            top_k=fd_config.model_config.moe_k,
            layer_idx=layer_id,
            gate_correction_bias=None,
            redundant_table_manger=redundant_table_manger,
            weight_key_map=weight_key_map,
        )

        if fd_config.model_config.moe_use_aux_free:
            self.experts.gate_correction_bias = self.create_parameter(
                shape=[1, fd_config.model_config.moe_num_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )
        else:
            self.experts.gate_correction_bias = None

        self.num_shared_experts = fd_config.model_config.moe_num_shared_experts
        if self.num_shared_experts > 0:
            shared_experts_hidden_dim = self.num_shared_experts * fd_config.model_config.moe_intermediate_size
            self.shared_experts = Ernie4_5_MLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_hidden_dim,
                prefix=f"{prefix}.shared_experts",
            )

    def load_state_dict(self, state_dict):
        self.gate.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)
        if self.experts.gate_correction_bias is not None:
            gate_correction_bias_tensor = state_dict.pop(self.experts.gate_correction_bias_key)
            if self.experts.gate_correction_bias.shape != gate_correction_bias_tensor.shape:
                gate_correction_bias_tensor = gate_correction_bias_tensor.reshape(
                    self.experts.gate_correction_bias.shape
                )
            self.experts.gate_correction_bias.set_value(gate_correction_bias_tensor)
        if self.num_shared_experts > 0:
            self.shared_experts.load_state_dict(state_dict)

    def update_state_dict(self, state_dict):
        self.fused_moe.load_state_dict(state_dict, True)

    def split_allgather_out(self, hidden_states: paddle.Tensor, token_num: int):
        token_num_per_rank = (token_num + self.tensor_parallel_size - 1) // self.tensor_parallel_size
        # AllGather will hang when the data shapes on multi-ranks are different!
        part_hidden_states = paddle.zeros(
            shape=[token_num_per_rank, hidden_states.shape[1]], dtype=hidden_states.dtype
        )
        start_offset = self.tensor_parallel_rank * token_num_per_rank
        end_offset = (self.tensor_parallel_rank + 1) * token_num_per_rank
        if end_offset > token_num:
            end_offset = token_num
        part_hidden_states[: (end_offset - start_offset), :] = hidden_states[start_offset:end_offset, :]
        out = self.experts(part_hidden_states, self.gate)
        multi_outs = []
        paddle.distributed.all_gather(multi_outs, out, self.tp_group)
        out = paddle.concat(multi_outs, axis=0)
        out = out[:token_num, :]
        return out

    def forward(self, hidden_states: paddle.Tensor):
        token_num = hidden_states.shape[0]
        if self.use_ep and self.use_tp and token_num >= self.tensor_parallel_size:
            out = self.split_allgather_out(hidden_states, token_num)
        else:
            out = self.experts(hidden_states, self.gate)
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
        redundant_table_manger: RedundantExpertManger = None,
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
                redundant_table_manger=redundant_table_manger,
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

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def update_state_dict(self, state_dict):
        self.mlp.update_state_dict(state_dict)

    def forward_old(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        if hidden_states.shape[0] == 0:
            # 当某张卡上的输入shape为0的时候！
            # 直接返回一个大空的东西！
            hidden_states = paddle.empty([0,8192], dtype="bfloat16")
            residual = paddle.empty([0,8192], dtype="bfloat16")
            return hidden_states, residual
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
        metadata,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):  
        if hidden_states is None or hidden_states.shape[0] == 0:

            hidden_states = paddle.empty([0,8192], dtype="bfloat16")
            residual = paddle.empty([0,8192], dtype="bfloat16")
            topk_idx = paddle.empty([0,8], dtype="int64")
            topk_weights = paddle.empty([0,8], dtype="float32")

            return hidden_states, residual, topk_idx, topk_weights

        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 为了计算attn！
        forward_meta.attn_backend.attention_metadata = metadata

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_out = paddle.matmul(hidden_states.cast("float32"), self.mlp.gate.weight)

        topk_idx, topk_weights = self.mlp.experts.quant_method.ep_decoder_runner.moe_select(self.mlp.experts, gate_out)


        topk_idx += 128

        return hidden_states, residual, topk_idx, topk_weights

    def compute_moe_ffn(
        self,
        permute_input: paddle.Tensor,
        token_nums_per_expert):
        ffn_out = self.mlp.experts.quant_method.compute_ffn(
            self.mlp.experts,
            permute_input,
            token_nums_per_expert,
            None,
            True,
        )
        return ffn_out

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
        self.fd_config = fd_config
        self.redundant_table_manger = None
        if fd_config.model_config.enable_redundant_experts is True:
            self.redundant_table_manger = RedundantExpertManger(
                n_routed_experts=fd_config.model_config.moe_num_experts,
                num_hidden_layers=fd_config.model_config.num_hidden_layers,
                redundant_experts_num=fd_config.model_config.redundant_experts_num,
                ep_size=fd_config.parallel_config.expert_parallel_size,
            )

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
                    redundant_table_manger=self.redundant_table_manger,
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


        if self.fd_config.parallel_config.is_attention_role:
            pass
        else:
            for i in range(self.num_layers):
                del self.layers[i].self_attn
                del self.layers[i].input_layernorm


        self.cached_attention_in_out = None
        self.cuda_graph = None

        self.dispatch_allocated_memory = None

        split_num = 3
        self.attn_graph = [[None for _ in range(split_num)] for _ in range(self.num_layers)]

        self.attn_input0 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]
        self.attn_input1 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]

        self.attn_res0 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]
        self.attn_res1 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]
        self.attn_res2 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]
        self.attn_res3 = [[None for _ in range(split_num)] for _ in range(self.num_layers)]


    def update_state_dict(self, state_dict):
        """
        Update model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        for i in range(
            self.fd_config.model_config.moe_layer_start_index,
            self.fd_config.model_config.num_hidden_layers,
        ):
            logger.info(f"Start update layer {i}")
            self.layers[i].update_state_dict(state_dict)


    def forward(self, ids_remove_padding: paddle.Tensor, forward_meta: ForwardMeta):
        IsH20 = self.fd_config.parallel_config.is_attention_role
        # 暂时设置成1!
        split_num = 3
        all_hidden_states = [None] * split_num
        forward_metas = [None] * split_num
        all_residual = [None] * split_num
        max_bs = self.fd_config.scheduler_config.max_num_seqs

        mc_bs = (max_bs + split_num - 1) // split_num

        if IsH20:

            hidden_states = None
            residual = None
            if ids_remove_padding.shape[0] > 0:
                hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)
                residual = None
                for i in range(3):
                    hidden_states, residual = self.layers[i].forward_old(forward_meta, hidden_states, residual)
            else:
                hidden_states = paddle.empty([0,8192], dtype="bfloat16")
                residual = paddle.empty([0,8192], dtype="bfloat16")

            for i in range(0, split_num):
                from copy import copy
                forward_meta_copy = copy(forward_meta)

                start_bs = i * mc_bs
                end_bs = start_bs + mc_bs
                end_bs = min(end_bs, max_bs)

                start_token_id = forward_meta.cu_seqlens_q[start_bs].item()
                assert forward_meta.cu_seqlens_q.shape[0] == max_bs + 1
                
                end_token_id =   forward_meta.cu_seqlens_q[end_bs].item()

                if end_token_id == start_token_id:
                    # 这个microbatch是空的，按道理是不需要处理!!
                    # 但是为了保证逻辑上始终是三个microbatch，同时也为了全图跑cuda graph
                    # 所以这里也当成一个batch来处理！只不过shape为0而已！
                    # 一切为了cuda graph
                    pass

                # 注意啦！这里+0是为了返回一个新的tensor哦！
                # 但是这里我不加哦！
                forward_meta_copy.seq_lens_encoder = forward_meta.seq_lens_encoder[start_bs:end_bs]
                forward_meta_copy.seq_lens_decoder = forward_meta.seq_lens_decoder[start_bs:end_bs]
                forward_meta_copy.seq_lens_this_time = forward_meta.seq_lens_this_time[start_bs:end_bs]

                forward_meta_copy.cu_seqlens_q = forward_meta.cu_seqlens_q[start_bs:end_bs+1] - start_token_id

                # 这里千万不能加0 哦！
                forward_meta_copy.block_tables = forward_meta.block_tables[start_bs:end_bs]

                forward_meta_copy.batch_id_per_token = forward_meta.batch_id_per_token[start_token_id:end_token_id] - start_bs

                # 这里必须要+0！
                forward_meta_copy.decoder_batch_ids = forward_meta.decoder_batch_ids + 0
                forward_meta_copy.decoder_tile_ids_per_batch = forward_meta.decoder_tile_ids_per_batch + 0
                
                # 这个都是
                forward_meta_copy.decoder_num_blocks_cpu = paddle.full([1], 0, dtype="int32").pin_memory()
                forward_meta_copy.decoder_num_blocks_device = forward_meta.decoder_num_blocks_device + 0
                forward_meta_copy.decoder_chunk_size_device = forward_meta.decoder_chunk_size_device + 0
                forward_meta_copy.max_len_tensor_cpu = forward_meta.max_len_tensor_cpu + 0
                forward_meta_copy.encoder_batch_ids = forward_meta.encoder_batch_ids + 0
                forward_meta_copy.encoder_tile_ids_per_batch =  forward_meta.encoder_tile_ids_per_batch + 0
                forward_meta_copy.encoder_num_blocks_x_cpu = forward_meta.encoder_num_blocks_x_cpu + 0
                forward_meta_copy.kv_batch_ids = forward_meta.kv_batch_ids + 0
                forward_meta_copy.kv_tile_ids_per_batch = forward_meta.kv_tile_ids_per_batch + 0
                forward_meta_copy.kv_num_blocks_x_cpu = forward_meta.kv_num_blocks_x_cpu + 0
                forward_meta_copy.max_len_kv_cpu = forward_meta.max_len_kv_cpu + 0


                forward_metas[i] = forward_meta_copy
                all_hidden_states[i] = hidden_states[start_token_id:end_token_id]
                all_residual[i] = residual[start_token_id:end_token_id]
        else:
            # MoE 机器啥也不需要做！
            pass

        print("大王啊")

        can_replay_graph = False
        need_capature_graph = False
        all_is_decoder = IsH20 and (forward_meta.seq_lens_encoder > 0).sum().item() == 0

        if IsH20 and all_is_decoder:
            assert ((forward_meta.seq_lens_encoder.reshape([-1]) > 0) & (forward_meta.seq_lens_decoder.reshape([-1]) > 0)).sum().item() == 0
            decoder_bs = ((forward_meta.seq_lens_this_time.reshape([-1]) > 0) & (forward_meta.seq_lens_decoder.reshape([-1]) > 0)).sum().item()

            print("decoder_bs", decoder_bs)

            can_replay_graph = (self.attn_graph[3][0] is not None and decoder_bs > 1)

            # 利用最大size来捕获图
            need_capature_graph = (self.attn_graph[3][0] is None and decoder_bs == max_bs)

            print("can_replay_graph", can_replay_graph)
        if need_capature_graph:
            print("need_capature_graph", need_capature_graph)

        IsH20 = self.fd_config.parallel_config.is_attention_role
        IsH100 = self.fd_config.parallel_config.is_moe_role
        runner = self.layers[3].mlp.experts.quant_method.ep_decoder_runner

        class AttentionInOut:
            forward_meta = None
            attn_metadata = None
            hidden_states = None
            residual = None            
            # 他俩是个中间tensor哦！,所以cuda graph不需要手动cache地址！          
            topk_idx = None
            topk_weights = None

        attention_in_out = [None] * split_num

        handles = [None] * split_num
        send_hooks = [None] * split_num
        recv_hooks = [None] * split_num

        dispatch_events = [None] * split_num
        combine_events = [None] * split_num

        dispatch_allocated_memory = [None] * split_num

        from collections import deque
        for j in range(split_num):
            send_hooks[j] = deque()
            recv_hooks[j] = deque()
            handles[j] = deque()
            dispatch_events[j] = deque()
            combine_events[j] = deque()

            if IsH20:
                token_num = all_hidden_states[j].shape[0]
            else:
                token_num = 0

            a = paddle.empty([8, runner.num_max_tokens * 24, 8192], dtype="float8_e4m3fn")
            b = paddle.empty([token_num, 3], dtype="bool")
            c = paddle.empty([8, runner.num_max_tokens * 24], dtype="int32")
            d = paddle.empty([8, 24], dtype="int64")
            e = paddle.empty([8], dtype="int32")
            f = paddle.empty([3], dtype="int32")
            g = paddle.empty([3, runner.num_max_tokens, 8768], dtype="uint8")
            h = paddle.empty([8, 8192//128, runner.num_max_tokens * 24], dtype="float32")

            dispatch_allocated_memory[j] = (a, b, c, d, e, f, g, h)


            attention_in_out[j] = AttentionInOut()

            # 这个是永远不改变的！
            attention_in_out[j].forward_meta = forward_metas[j]

            if IsH20 and all_hidden_states[j].shape[0] > 0:
                forward_metas[j].attn_backend.init_attention_metadata(attention_in_out[j].forward_meta)
                attention_in_out[j].attn_metadata = forward_metas[j].attn_backend.attention_metadata

            # 下面俩是动态变化的，每层的时候是会变化的哦！
            attention_in_out[j].hidden_states = all_hidden_states[j]
            attention_in_out[j].residual = all_residual[j]

        if IsH20:
            if need_capature_graph:
                self.cached_attention_in_out = attention_in_out
                self.dispatch_allocated_memory = dispatch_allocated_memory

            elif can_replay_graph:
                # 是用这个预先分配的大空间哦！
                # 防止和capture的临时空间时候冲突！
                dispatch_allocated_memory = self.dispatch_allocated_memory

                # 需要把新产生的tensor 数据 拷贝到老的 cached tensor数据!
                for i in range(split_num):

                    from dataclasses import dataclass, fields
                    person_fields = fields(self.cached_attention_in_out[i].forward_meta)
                    for field in person_fields:
                        name = field.name
                        if name in ["decoder_batch_ids", 
                                    "decoder_tile_ids_per_batch",
                                    "cu_seqlens_q"]:
                            cache_tensor = getattr(self.cached_attention_in_out[i].forward_meta, name)
                            coming_tensor = getattr(attention_in_out[i].forward_meta, name)
                            assert cache_tensor.data_ptr() != coming_tensor.data_ptr()
                            assert cache_tensor.shape == coming_tensor.shape
                            cache_tensor.copy_(coming_tensor, False)

        self.barrier_id = -1
        def zkk_barrier():
            self.barrier_id += 1
            #paddle.device.synchronize()
            #paddle.distributed.barrier()
            # print("到达", self.barrier_id)
            #paddle.device.synchronize()

        if IsH20:

            def compute_atten(layer_id, i):
                #print(f"compute_atten({layer_id}, {i})")
                if need_capature_graph:
                    self.attn_graph[layer_id][i] = graphs.CUDAGraph()
                    self.attn_graph[layer_id][i].capture_begin()


                    hidden_states, residual, topk_idx, topk_weights = self.layers[layer_id].forward_attn(
                                                                    attention_in_out[i].attn_metadata, 
                                                                    attention_in_out[i].forward_meta, 
                                                                    attention_in_out[i].hidden_states, 
                                                                    attention_in_out[i].residual)
                    self.attn_graph[layer_id][i].capture_end()

                    self.attn_graph[layer_id][i].replay()

                    # 记住cuda graph的输入和输出地址！
                    # 千万不可以加零！因为我们要记住cuda graph的输入输出地址！
                    self.attn_input0[layer_id][i] = attention_in_out[i].hidden_states
                    self.attn_input1[layer_id][i] = attention_in_out[i].residual

                    self.attn_res0[layer_id][i] = hidden_states
                    self.attn_res1[layer_id][i] = residual
                    self.attn_res2[layer_id][i] = topk_idx
                    self.attn_res3[layer_id][i] = topk_weights

                    # 更新变量哈哈哈哈！
                    attention_in_out[i].hidden_states = self.attn_res0[layer_id][i] + 0
                    attention_in_out[i].residual = self.attn_res1[layer_id][i] + 0
                    attention_in_out[i].topk_idx = self.attn_res2[layer_id][i] + 0
                    attention_in_out[i].topk_weights = self.attn_res3[layer_id][i] + 0

                elif can_replay_graph:

                    valid_token_num = attention_in_out[i].hidden_states.shape[0]

                    if valid_token_num == 0:
                        # 如果是0，那我就干脆别计算了！
                        attention_in_out[i].hidden_states = paddle.empty([0,8192], dtype="bfloat16")
                        attention_in_out[i].residual = paddle.empty([0,8192], dtype="bfloat16")
                        attention_in_out[i].topk_idx = paddle.empty([0,8], dtype="int64")
                        attention_in_out[i].topk_weights = paddle.empty([0,8], dtype="float32")
                        return


                    self.attn_input0[layer_id][i].copy_(attention_in_out[i].hidden_states, False)
                    self.attn_input1[layer_id][i].copy_(attention_in_out[i].residual, False)

                    self.attn_graph[layer_id][i].replay()

                    # 将结果赋给attention_in_out啊！
                    attention_in_out[i].hidden_states = self.attn_res0[layer_id][i][:valid_token_num]
                    attention_in_out[i].residual = self.attn_res1[layer_id][i][:valid_token_num]
                    attention_in_out[i].topk_idx = self.attn_res2[layer_id][i][:valid_token_num]
                    attention_in_out[i].topk_weights = self.attn_res3[layer_id][i][:valid_token_num]

                else:

                    hidden_states, residual, topk_idx, topk_weights = self.layers[layer_id].forward_attn(
                                                                    attention_in_out[i].attn_metadata, 
                                                                    attention_in_out[i].forward_meta, 
                                                                    attention_in_out[i].hidden_states, 
                                                                    attention_in_out[i].residual)

                    attention_in_out[i].hidden_states = hidden_states
                    attention_in_out[i].residual = residual
                    attention_in_out[i].topk_idx = topk_idx
                    attention_in_out[i].topk_weights = topk_weights
                    assert hidden_states.isnan().any() == False
                    assert residual.isnan().any() == False

            def dispatch_send(i):
                #print(f"dispatch_send({i})")
                _, handle, event, a2e_isend_hook = runner.buffer.a2e_isend_two_stage_v3(
                    attention_in_out[i].hidden_states,
                    attention_in_out[i].topk_idx,
                    attention_in_out[i].topk_weights,
                    dispatch_allocated_memory[i],
                    runner.num_max_tokens,
                    runner.num_experts,
                    use_fp8=runner.use_fp8,
                )
                handles[i].appendleft(handle)
                send_hooks[i].appendleft(a2e_isend_hook)
                dispatch_events[i].appendleft(event)


            def dispatch_wait(i):
                #print(f"dispatch_wait({i})")
                a = dispatch_events[i].pop()
                tmp = send_hooks[i].pop()()

            def combine_receive(i):
                #print(f"combine_receive({i})")
                e2a_x, event, e2a_irecv_hook = runner.buffer.e2a_irecv_two_stage_v3(
                    attention_in_out[i].topk_idx,
                    attention_in_out[i].topk_weights,
                    handles[i].pop(),
                    dispatch_use_fp8=runner.use_fp8,
                    out=attention_in_out[i].hidden_states,
                )

                recv_hooks[i].appendleft(e2a_irecv_hook)

                combine_events[i].appendleft(event)

            def combine_wait(i):
                #print(f"combine_wait({i})")
                a = combine_events[i].pop()
                # a.current_stream_wait()

                tmp = recv_hooks[i].pop()()
                recv_hooks[i].appendleft(tmp)
                #tmp.current_stream_wait()

            def capatured_code():
                compute_atten(3, 0)
                dispatch_send(0)
                dispatch_wait(0)
                zkk_barrier()


                compute_atten(3, 1)

                for layer_id in range(3, self.num_layers):
                    tmp_split_num = range(split_num)
                    if layer_id == 3:
                        tmp_split_num = [2]
                    for j in tmp_split_num:

                        # 上一个batch
                        dispatch_send((j-1+split_num)%split_num)
                        dispatch_wait((j-1+split_num)%split_num)

                        if layer_id > 3:
                            # 计算attention之前一定要保证他的输入已经到来了！
                            tmp = recv_hooks[j].pop()
                            tmp.current_stream_wait()
                        compute_atten(layer_id, j)

                        # 上上个batch！
                        combine_receive((j-2+split_num)%split_num)
                        combine_wait((j-2+split_num)%split_num)

                dispatch_send(2)
                dispatch_wait(2)
                combine_receive(1)
                combine_wait(1)

                tmp = recv_hooks[1].pop()
                tmp.current_stream_wait()

                combine_receive(2)
                combine_wait(2)
                tmp = recv_hooks[2].pop()
                tmp.current_stream_wait()

            capatured_code()

        else:
            # 搞一个大槽子放东西！
            moe_input = [[None for _ in range(2)] for _ in range(split_num)]
            moe_out = [None for _ in range(split_num)]

            def dispatch_receive(i):
                #print(f"dispatch_receive({i})")
                (
                    packed_recv_x,
                    packed_recv_count,
                    rdma_send_flags,
                    handle,
                    event,
                    a2e_irecv_hook,
                ) = runner.buffer.a2e_irecv_two_stage_v3(
                    dispatch_allocated_memory[i],
                    runner.hidden,
                    runner.top_k,
                    runner.num_max_tokens,
                    runner.num_experts,
                    use_fp8=runner.use_fp8,
                )
                handles[i].appendleft(handle)
                recv_hooks[i].appendleft(a2e_irecv_hook)
                dispatch_events[i].appendleft(event)
                moe_input[i][0] = packed_recv_x
                moe_input[i][1] = packed_recv_count


            def dispatch_wait(i):
                #print(f"dispatch_wait({i})")
                a = dispatch_events[i].pop()
                # a.current_stream_wait()
                tmp = recv_hooks[i].pop()()
                recv_hooks[i].append(tmp)
                #tmp.current_stream_wait()

            def compute_moe(layer_id, i):
                #print(f"compute_moe({layer_id}, {i})")
                ffn_out = self.layers[layer_id].compute_moe_ffn(moe_input[i][0], moe_input[i][1])
                moe_out[i] = ffn_out

            def combine_send(i):
                #print(f"combine_send({i})")
                event, e2a_isend_hook = runner.buffer.e2a_isend_two_stage_v3(
                    moe_out[i], 
                    runner.top_k,
                    handles[i].pop(),
                    dispatch_use_fp8=runner.use_fp8,
                    out=None,
                )
                send_hooks[i].appendleft(e2a_isend_hook)
                combine_events[i].appendleft(event)

            def combine_wait(i, is_wait=False):
                #print(f"combine_wait({i})")
                a = combine_events[i].pop()
                # a.current_stream_wait()

                tmp = send_hooks[i].pop()()
                if is_wait:
                    # 这个是为了让通信流回归到主流而采取的措施！
                    # 只是为了适配cuda graph！
                    tmp.current_stream_wait()

            def main_code():

                # 这个必须要，是为了适配cuda graph
                common_stream = runner.buffer.all2all_buffer.runtime.get_comm_stream()

                dispatch_receive(0)
                dispatch_wait(0)
                zkk_barrier()
                haha = 9
                tmp = recv_hooks[0].pop()
                tmp.current_stream_wait()
                compute_moe(haha//3,0)

                dispatch_receive(1)
                dispatch_wait(1)

                for layer_id in range(3, self.num_layers):
                    tmp_split_num = range(split_num)
                    if layer_id == 3:
                        tmp_split_num = [1, 2]
                    if layer_id == self.num_layers - 1:
                        tmp_split_num = [0, 1]
                    for j in tmp_split_num:

                        # 上一个batch
                        combine_send((j-1+split_num)%split_num)
                        combine_wait((j-1+split_num)%split_num)

                        haha += 1
                        tmp = recv_hooks[j].pop()
                        tmp.current_stream_wait()
                        compute_moe(haha//3,j)

                        # 下一个batch
                        dispatch_receive((j+1)%split_num)
                        dispatch_wait((j+1)%split_num)

                haha += 1
                tmp = recv_hooks[2].pop()
                tmp.current_stream_wait()
                compute_moe(haha//3,2)
                combine_send(1)
                combine_wait(1)
                combine_send(2)
                combine_wait(2, True)

            # if self.cuda_graph is None:
            #     self.cuda_graph = graphs.CUDAGraph()
            #     self.cuda_graph.capture_begin()
            #     main_code()
            #     self.cuda_graph.capture_end()
            #     self.cuda_graph.replay()
            #     # capature住这个输入的变量！
            #     self.dispatch_allocated_memory = dispatch_allocated_memory
            # else:
            #     self.cuda_graph.replay()

            main_code()

        # 让三台机器一起结束！，暂时先注释掉！
        # paddle.distributed.barrier()

        if IsH20:
            if ids_remove_padding.shape[0] == 0:
                return None
            hidden_states = paddle.concat([attention_in_out[j].hidden_states for j in range(split_num)], axis=0)
            residuals = paddle.concat([attention_in_out[j].residual for j in range(split_num)], axis=0)
            hidden_states = hidden_states + residuals
            out = self.norm(hidden_states)
            assert out.isnan().any() == False
            return out
        else:
            # MoE机器返回None
            return None


    def forward_no_afd(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding)

        if current_platform.is_iluvatar() and forward_meta.attn_backend.mixed:
            hidden_states = forward_meta.attn_backend.transpose(hidden_states)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta, hidden_states, residual)

        hidden_states = hidden_states + residual

        out = self.norm(hidden_states)

        if current_platform.is_iluvatar() and forward_meta.attn_backend.mixed:
            out = forward_meta.attn_backend.reverse_transpose(out)

        return out


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_MoeForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
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
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.ernie.embed_tokens.embeddings.weight})
        else:
            self.lm_head.load_state_dict(state_dict)

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
            rename_offline_ckpt_suffix_to_fd_suffix,
        )

        general_params_mapping = [
            # (param_name, weight_name, expert_id, shard_id)
            ("embed_tokens.embeddings", "embed_tokens", None, None),
            ("lm_head.linear", "lm_head", None, None),
            ("experts.gate_correction_bias", "moe_statics.e_score_correction_bias", None, None),
            ("qkv_proj", "q_proj", None, "q"),
            ("qkv_proj", "k_proj", None, "k"),
            ("qkv_proj", "v_proj", None, "v"),
            ("up_gate_proj", "gate_proj", None, "gate"),
            ("up_gate_proj", "up_proj", None, "up"),
            ("attn.cache_k_scale", "cachek_matmul.activation_scale", None, None),
            ("attn.cache_v_scale", "cachev_matmul.activation_scale", None, None),
            ("attn.cache_k_zp", "cachek_matmul.activation_zero_point", None, None),
            ("attn.cache_v_zp", "cachev_matmul.activation_zero_point", None, None),
        ]

        expert_params_mapping = []
        if getattr(self.fd_config.model_config, "moe_num_experts", None) is not None:
            if self.fd_config.parallel_config.expert_parallel_size > 1:
                num_experts = self.fd_config.parallel_config.num_experts_per_rank
                num_experts_start_offset = self.fd_config.parallel_config.num_experts_start_offset
            else:
                num_experts = self.fd_config.model_config.moe_num_experts
                num_experts_start_offset = 0

            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                num_experts=num_experts,
                ckpt_down_proj_name="down_proj",
                ckpt_gate_up_proj_name="up_gate_proj",
                ckpt_gate_proj_name="gate_proj",
                ckpt_up_proj_name="up_proj",
                param_gate_up_proj_name="experts.up_gate_proj_",
                param_down_proj_name="experts.down_proj_",
                num_experts_start_offset=num_experts_start_offset,
            )
        all_param_mapping = [
            (param, weight, exp, shard, False) for param, weight, exp, shard in general_params_mapping
        ] + [(param, weight, exp, shard, True) for param, weight, exp, shard in expert_params_mapping]
        checkpoint_to_fd_key_fn = rename_offline_ckpt_suffix_to_fd_suffix(
            fd_config=self.fd_config, ckpt_weight_suffix="quant_weight", ckpt_scale_suffix="weight_scale"
        )
        params_dict = dict(self.named_parameters())

        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()))

        for loaded_weight_name, loaded_weight in weights_iterator:
            loaded_weight_name = loaded_weight_name.replace("model", "ernie")
            for param_name, weight_name, exp_id, shard_id, is_moe in all_param_mapping:
                loaded_weight_name = checkpoint_to_fd_key_fn(loaded_weight_name, is_moe)
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                expert_id = exp_id
                shard_id = shard_id
                break
            else:
                expert_id = None
                shard_id = None
                loaded_weight_name = checkpoint_to_fd_key_fn(loaded_weight_name, is_moe=False)
                model_param_name = loaded_weight_name
                if model_param_name not in params_dict.keys():
                    continue
                param = params_dict[model_param_name]

            # Get weight loader from parameter and set weight
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            sig = inspect.signature(weight_loader)
            if "expert_id" in sig.parameters:
                weight_loader(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            else:
                weight_loader(param, loaded_weight, shard_id)

            model_sublayer_name = re.sub(
                r"\.(up_gate_proj_weight|down_proj_weight|weight|cache_k_scale|cache_v_scale)$", "", model_param_name
            )
            process_weights_after_loading_fn(model_sublayer_name, param)

        if self.tie_word_embeddings:
            self.lm_head.load_state_dict({self.lm_head.weight_key: self.ernie.embed_tokens.embeddings.weight})

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
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
            self.ernie.layers[i].mlp.experts(fake_hidden_states, self.ernie.layers[i].mlp.gate)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.ernie(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.ernie.clear_grpah_opt_backend(fd_config=self.fd_config)


@ModelRegistry.register_model_class(
    architecture="Ernie4_5_ForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
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


@ModelRegistry.register_model_class(
    architecture="Ernie4_5ForCausalLM",
    module_name="ernie4_5_moe",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Ernie4_5ForCausalLM(Ernie4_5_ForCausalLM):
    """
    Ernie4_5ForCausalLM 0.3B-PT
    """

    @classmethod
    def name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5ForCausalLM"


class Ernie4_5_MoePretrainedModel(PretrainedModel):
    """
    Ernie4_5_MoePretrainedModel
    """

    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "Ernie4_5_MoeForCausalLM"

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


class Ernie4_5_PretrainedModel(Ernie4_5_MoePretrainedModel):
    """
    Ernie4_5_PretrainedModel
    """

    @classmethod
    def arch_name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5_ForCausalLM"


class Ernie4_5PretrainedModel(Ernie4_5_PretrainedModel):
    """
    Ernie4_5PretrainedModel 0.3B-PT
    """

    @classmethod
    def arch_name(self):
        """
        Model Architecture Name
        """
        return "Ernie4_5ForCausalLM"
