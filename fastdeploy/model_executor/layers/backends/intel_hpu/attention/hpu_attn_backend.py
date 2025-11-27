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
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import paddle

if TYPE_CHECKING:
    from paddle._typing.dtype_like import _DTypeLiteral

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import HPUForwardMeta

from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear


def get_attention_mask(seq_lens_encoder, seq_lens_decoder, batch_size, query_len):
    max_context_len = int(paddle.max(seq_lens_decoder).item())
    past_mask = paddle.arange(0, max_context_len, dtype=paddle.int32)
    past_mask = paddle.greater_equal(
        past_mask.reshape([1, -1]).expand([batch_size, -1]), seq_lens_decoder.reshape([-1, 1]).astype(paddle.int32)
    )
    past_mask = (
        past_mask.reshape([batch_size, 1, -1])
        .expand([batch_size, query_len, -1])
        .reshape([batch_size, 1, query_len, -1])
    )
    len_mask = paddle.greater_equal(
        paddle.arange(0, query_len, dtype=paddle.int32).reshape([1, query_len]),
        seq_lens_encoder.unsqueeze(-1).astype(paddle.int32),
    )
    len_mask = len_mask.reshape([batch_size, 1, 1, query_len])
    attn_mask = paddle.triu(paddle.ones((batch_size, 1, query_len, query_len), dtype=paddle.bool), diagonal=1)
    mask = attn_mask.logical_or(len_mask)
    mask = paddle.concat((past_mask, mask), axis=-1)
    off_value = -math.inf
    attn_mask = paddle.zeros_like(mask, dtype=paddle.bfloat16).masked_fill_(mask, off_value)
    attn_mask = paddle.unsqueeze(attn_mask, axis=1)
    return attn_mask


class AttentionBackend_HPU(AttentionBackend):
    """The base class of attention backends"""

    @abstractmethod
    def init_attention_metadata(self, forward_meta: HPUForwardMeta):
        """Initialize the forward metadata."""
        raise NotImplementedError()

    def forward(
        self,
        src: paddle.Tensor,
        qkv_proj: QKVParallelLinear,
        o_proj: RowParallelLinear,
        layer: paddle.nn.Layer,
        forward_meta: HPUForwardMeta,
    ):
        """
        Run a forward.
        args:
            src: the hidden states tensor
            residual_input: the residual tensor
            layer: The layer that will be used for the forward.
            forward_meta: The forward metadata.
        """
        if forward_meta.forward_mode.is_mixed():
            return self.forward_mixed(
                src,
                qkv_proj,
                o_proj,
                layer,
                forward_meta,
            )
        elif forward_meta.forward_mode.is_decode():
            return self.forward_decode(
                src,
                qkv_proj,
                o_proj,
                layer,
                forward_meta,
            )
        else:
            return self.forward_extend(
                src,
                qkv_proj,
                o_proj,
                layer,
                forward_meta,
            )

    def forward_mixed(
        self,
        src: paddle.Tensor,
        qkv_proj: QKVParallelLinear,
        o_proj: RowParallelLinear,
        layer: paddle.nn.Layer,
        forward_meta: HPUForwardMeta,
    ):
        """Run a forward for mix."""
        raise NotImplementedError()

    def forward_decode(
        self,
        src: paddle.Tensor,
        qkv_proj: QKVParallelLinear,
        o_proj: RowParallelLinear,
        layer: paddle.nn.Layer,
        forward_meta: HPUForwardMeta,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        src: paddle.Tensor,
        qkv_proj: QKVParallelLinear,
        o_proj: RowParallelLinear,
        layer: paddle.nn.Layer,
        forward_meta: HPUForwardMeta,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()


@dataclass
class HPUAttentionMetadata(AttentionMetadata):
    """
    HPUAttentionMetadata
    """

    max_len_kv: paddle.Tensor = None
    set_max_lengths: int = -1
    encoder_batch_ids: paddle.Tensor = None
    encoder_tile_ids_per_batch: paddle.Tensor = None
    encoder_num_blocks: paddle.Tensor = None
    kv_batch_ids: paddle.Tensor = None
    kv_tile_ids_per_batch: paddle.Tensor = None
    kv_num_blocks: paddle.Tensor = None
    decoder_batch_ids: paddle.Tensor = None
    decoder_tile_ids_per_batch: paddle.Tensor = None
    decoder_num_blocks: paddle.Tensor = None

    _dtype: _DTypeLiteral = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    encoder_block_shape_q: Optional[paddle.Tensor] = None
    decoder_block_shape_q: Optional[paddle.Tensor] = None
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[paddle.Tensor] = field(default_factory=list)


class HPUAttentionBackend(AttentionBackend_HPU):
    """
    HPUAttentionBackend backend implementation.
    """

    def __init__(
        self,
        llm_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        """
        HPUAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: HPUAttentionMetadata = None
        # TODO(gongshaotian): Use llm_config parameters in the correct location
        self.block_size = llm_config.cache_config.block_size
        self.max_seq_len = llm_config.model_config.max_model_len
        self.rope_theta = 10000.0 if llm_config.model_config.rope_theta is None else llm_config.model_config.rope_theta
        self.rope_3d = getattr(llm_config.model_config, "rope_3d", False)
        self.causal = getattr(llm_config.model_config, "causal", True)
        self.speculative_method: str = llm_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = llm_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = llm_config.speculative_config.model_type == "mtp"
        self.rank: int = llm_config.parallel_config.tensor_parallel_rank
        self.tp_size = llm_config.parallel_config.tensor_parallel_size

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = llm_config.model_config.num_hidden_layers

        # pd_disaggregation
        self.use_pd_disaggregation = int(os.getenv("FLAGS_use_pd_disaggregation", 0))
        self.start_layer_index = llm_config.model_config.start_layer_index

    def init_attention_metadata(self, forward_meta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = HPUAttentionMetadata()
        metadata.encoder_block_shape_q = 64
        metadata.decoder_block_shape_q = 16
        metadata.max_partition_size = 32768
        metadata.encoder_max_partition_size = 32768
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"
        metadata.block_tables = forward_meta.block_tables
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.attn_mask = forward_meta.attn_mask

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        self.attention_metadata = metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: Optional[str] = None,
    ):
        """
        Caculate kv cache shape
        """
        key_cache_shape = value_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]
        return key_cache_shape, value_cache_shape

    def forward_extend(
        self, src, qkv_proj: QKVParallelLinear, o_proj: RowParallelLinear, layer: Attention, forward_meta
    ):
        """
        forward_extend
        """
        # metadata = self.attention_metadata

        from fastdeploy.model_executor.ops.intel_hpu import (
            fused_qkv_rope,
            fused_sdpa_proj_t,
            index_copy_,
        )

        query_states, key_value_states = fused_qkv_rope(
            src,
            qkv_proj.weight,
            qkv_proj.bias,
            forward_meta.rotary_embs,
            self.head_dim,
            self.num_heads,
            forward_meta.total_batch,
            transpose=False,
            use_neox_style=layer.use_neox_rotary_style,
        )

        kv, B, BP_BS, M, H = key_value_states.shape
        key_value_states_reshape = key_value_states.reshape([kv, -1, forward_meta.block_size, M, H])
        key_states = key_value_states_reshape[0]
        value_states = key_value_states_reshape[1]
        k_cache = forward_meta.caches[2 * layer.layer_id]
        v_cache = forward_meta.caches[2 * layer.layer_id + 1]
        index_copy_(k_cache, forward_meta.block_indices, key_states, 0)
        index_copy_(v_cache, forward_meta.block_indices, value_states, 0)

        if forward_meta.block_list.shape == forward_meta.block_indices.shape:
            out_linear_out = fused_sdpa_proj_t(
                query_states,
                key_value_states,
                forward_meta.attn_mask,
                None,
                o_proj.weight,
                scaling_factor=self.head_dim**-0.5,
                causal=True,
                softmax_mode=0,
            )
        else:
            key_states_with_context = k_cache.index_select(forward_meta.block_list)
            val_states_with_context = v_cache.index_select(forward_meta.block_list)
            key_value_states_with_context = paddle.stack(
                [key_states_with_context, val_states_with_context], axis=0
            ).reshape([kv, B, -1, M, H])
            if forward_meta.attn_mask is None:
                forward_meta.attn_mask = get_attention_mask(
                    forward_meta.seq_lens_encoder[forward_meta.batch_ids],
                    forward_meta.seq_lens_decoder[forward_meta.batch_ids],
                    query_states.shape[0],
                    query_states.shape[1],
                )
            out_linear_out = fused_sdpa_proj_t(
                query_states,
                key_value_states_with_context,
                forward_meta.attn_mask,
                None,
                o_proj.weight,
                scaling_factor=self.head_dim**-0.5,
                causal=False,
                softmax_mode=0,
            )

        if self.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce_custom,
            )

            tensor_model_parallel_all_reduce_custom(out_linear_out)

        return out_linear_out

    def forward_decode(
        self, src, qkv_proj: QKVParallelLinear, o_proj: RowParallelLinear, layer: Attention, forward_meta
    ):
        """
        forward_decode
        """
        # metadata = self.attention_metadata
        from fastdeploy.model_executor.ops.intel_hpu import fused_block_attention

        res = fused_block_attention(
            src,
            forward_meta.rotary_embs,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.block_groups,
            forward_meta.block_list,
            forward_meta.block_mapping,
            forward_meta.attention_mask,
            forward_meta.block_indices,
            forward_meta.block_offsets,
            qkv_proj.weight,
            qkv_proj.bias,
            o_proj.weight,
            None,  # past_key: not used in decode mode
            None,  # past_value: not used in decode mode
            self.head_dim,
            self.num_heads,
            scaling_factor=self.head_dim**-0.5,
            transpose=False,
            use_neox_style=layer.use_neox_rotary_style,
            epsilon=1e-6,
        )

        # all_reduce
        if self.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce_custom,
            )

            tensor_model_parallel_all_reduce_custom(res)
        return res
