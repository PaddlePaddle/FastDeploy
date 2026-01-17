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

import os
from dataclasses import dataclass, field
from typing import List, Optional

import paddle

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.layers.backends.metax.attention.flash_attention_interface import (
    flash_attn_kvcache_func,
    flash_attn_unpadded_func,
)
from fastdeploy.model_executor.ops.gpu import cache_kv_with_rope
from fastdeploy.model_executor.ops.gpu import merge_qkv as merge_qkv_cu
from fastdeploy.model_executor.ops.gpu import split_qkv as split_qkv_cu


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """
    FlashAttentionMetadata
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
    cu_seqlens_q_decode: paddle.Tensor = None
    batch_ids_per_token_decode: paddle.Tensor = None
    seq_lens_decode: paddle.Tensor = None
    block_table_decode: paddle.Tensor = None

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    encoder_block_shape_q: int = -1
    decoder_block_shape_q: int = -1
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)


class FlashAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: FlashAttentionMetadata

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ) -> None:
        """
        FlashAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: FlashAttentionMetadata = FlashAttentionMetadata()
        self.record_block_table_metadata = {}
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.total_num_heads = self.num_heads + 2 * self.kv_num_heads
        self.total_hidden_dim = self.total_num_heads * self.head_dim
        self.dtype = paddle.get_default_dtype()
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 32768))

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.enable_mm = fd_config.model_config.enable_mm
        self.model_type = fd_config.model_config.model_type
        self.is_neox_style = False
        if "paddleocr" in fd_config.model_config.model_type:
            self.is_neox_style = True

        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self.attention_metadata.decoder_batch_ids = paddle.empty(shape=[max_num_seqs], dtype="int32")
        self.attention_metadata.cu_seqlens_q_decode = paddle.empty(shape=[max_num_seqs + 1], dtype="int32")
        self.attention_metadata.batch_ids_per_token_decode = paddle.empty(shape=[max_num_seqs], dtype="int32")
        self.attention_metadata.seq_lens_decode = paddle.empty(shape=[max_num_seqs, 1], dtype="int32")
        self.attention_metadata.block_table_decode = paddle.empty(
            shape=[
                max_num_seqs,
                self.max_seq_len // self.block_size + fd_config.cache_config.enc_dec_block_num,
            ],
            dtype="int32",
        )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        forward_meta.forward_mode = ForwardMode.NATIVE
        self.prefill_info_dict = {}
        self.decode_info_dict = {}

        prefill_non_zeros_ids = forward_meta.seq_lens_this_time > 1
        decode_non_zeros_ids = forward_meta.seq_lens_this_time == 1
        self.prefill_info_dict["batch_ids"] = paddle.where(prefill_non_zeros_ids)[0].astype("int32")
        self.decode_info_dict["batch_ids"] = paddle.where(decode_non_zeros_ids)[0].astype("int32")

        self.prefill_len = len(self.prefill_info_dict["batch_ids"])
        self.decode_len = len(self.decode_info_dict["batch_ids"])
        self.has_prefill = self.prefill_len > 0
        self.has_decode = self.decode_len > 0

        if self.has_prefill:
            batch_ids_prefill = self.prefill_info_dict["batch_ids"]

            seq_lens_this_time_prefill = forward_meta.seq_lens_this_time[batch_ids_prefill, 0]
            self.prefill_info_dict["cu_seqlens_q"] = paddle.concat(
                [paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens_this_time_prefill, axis=0).astype("int32")],
                axis=0,
            )
            self.prefill_info_dict["seq_lens_prefill"] = paddle.zeros(self.prefill_len, dtype="int32")

            local_ids = paddle.arange(self.prefill_len, dtype="int32")
            self.prefill_info_dict["batch_ids_per_token"] = paddle.repeat_interleave(
                local_ids, repeats=seq_lens_this_time_prefill, axis=0
            )

        if self.has_decode:
            batch_ids_decode = self.decode_info_dict["batch_ids"]

            seq_lens_this_time_decode = forward_meta.seq_lens_this_time[batch_ids_decode, 0]
            cu_seqlens_q_decode = paddle.concat(
                [paddle.zeros([1], dtype="int32"), paddle.cumsum(seq_lens_this_time_decode, axis=0).astype("int32")],
                axis=0,
            )

            local_ids = paddle.arange(self.decode_len, dtype="int32")
            batch_ids_per_token_decode = paddle.repeat_interleave(local_ids, repeats=seq_lens_this_time_decode, axis=0)

            self.attention_metadata.decoder_batch_ids[: self.decode_len].copy_(batch_ids_decode)  # global batch id
            self.attention_metadata.cu_seqlens_q_decode[: self.decode_len + 1].copy_(cu_seqlens_q_decode)
            self.attention_metadata.batch_ids_per_token_decode[: self.decode_len].copy_(batch_ids_per_token_decode)
            self.attention_metadata.seq_lens_decode[: self.decode_len].copy_(
                forward_meta.seq_lens_decoder[batch_ids_decode, 0]
            )
            self.attention_metadata.block_table_decode[: self.decode_len].copy_(
                forward_meta.block_tables[batch_ids_decode, :]
            )

        if self.has_prefill and self.has_decode:
            non_zeros_mask = forward_meta.seq_lens_this_time != 0
            seq_lens_non_zeros = forward_meta.seq_lens_this_time[non_zeros_mask].astype("int32")

            global_sequence_offsets = paddle.zeros(seq_lens_non_zeros.shape[0] + 1, dtype="int32")
            global_sequence_offsets[1:] = paddle.cumsum(seq_lens_non_zeros)

            is_prefill_array = seq_lens_non_zeros > 1

            group_boundary = paddle.where(is_prefill_array[1:] != is_prefill_array[:-1])[0].astype("int32") + 1
            group_starts = paddle.concat((paddle.zeros([1], dtype="int32"), group_boundary))
            group_ends = paddle.concat(
                (group_boundary, paddle.full([1], fill_value=seq_lens_non_zeros.shape[0], dtype="int32"))
            )

            compact_meta = []
            prefill_ptr = 0
            decode_ptr = 0

            for start, end in zip(group_starts, group_ends):
                is_prefill = is_prefill_array[start]
                g_start = global_sequence_offsets[start]
                g_end = global_sequence_offsets[end]
                num_tokens = g_end - g_start

                if is_prefill:
                    # [0, prefill_start, prefill_end, global_start, global_end]
                    compact_meta.append([0, prefill_ptr, prefill_ptr + num_tokens, g_start, g_end])
                    prefill_ptr += num_tokens
                else:
                    # [1, decode_start, decode_end, global_start, global_end]
                    compact_meta.append([1, decode_ptr, decode_ptr + num_tokens, g_start, g_end])
                    decode_ptr += num_tokens

            self.hybrid_stage_meta = paddle.to_tensor(compact_meta, dtype="int32")
            self.prefill_qkv = paddle.zeros([prefill_ptr, self.total_hidden_dim], dtype=self.dtype)
            self.decode_qkv = paddle.zeros([decode_ptr, self.total_hidden_dim], dtype=self.dtype)
            self.merged_output = paddle.zeros(
                [prefill_ptr + decode_ptr, self.num_heads, self.head_dim], dtype=self.dtype
            )

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = value_cache_shape = [max_num_blocks, self.block_size, self.kv_num_heads, self.head_dim]

        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]

        return key_cache_shape, value_cache_shape

    def split_pd_qkv(self, qkv):
        split_qkv_cu(qkv, self.hybrid_stage_meta, self.prefill_qkv, self.decode_qkv)

    def merge_pd_output(self, prefill_out, decode_out):
        merge_qkv_cu(prefill_out, decode_out, self.hybrid_stage_meta, self.merged_output)

    def apply_rope_prefill(self, qkv, rotary_embs, caches_k, caches_v, block_tables):
        return cache_kv_with_rope(
            qkv,
            rotary_embs,
            self.prefill_info_dict["batch_ids_per_token"],
            self.prefill_info_dict["batch_ids"],
            self.prefill_info_dict["cu_seqlens_q"],
            self.prefill_info_dict["seq_lens_prefill"],
            caches_k,
            caches_v,
            block_tables,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            self.block_size,
            out_dims=3,
            neox_style=self.is_neox_style,  # is neox style
        )

    def apply_rope_decode(self, qkv, rotary_embs):
        return cache_kv_with_rope(
            qkv,
            rotary_embs,
            self.attention_metadata.batch_ids_per_token_decode,
            self.attention_metadata.decoder_batch_ids,
            self.attention_metadata.cu_seqlens_q_decode,
            self.attention_metadata.seq_lens_decode,
            None,
            None,
            None,
            self.num_heads,
            self.kv_num_heads,
            self.head_dim,
            -1,
            out_dims=4,
            neox_style=self.is_neox_style,  # is neox style
        )

    def forward_prefill(self, prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        q, k, v = self.apply_rope_prefill(
            prefill_qkv,
            forward_meta.rotary_embs,
            forward_meta.caches[k_cache_id],
            forward_meta.caches[v_cache_id],
            forward_meta.block_tables,
        )

        prefill_out = flash_attn_unpadded_func(
            q,
            k,
            v,
            self.prefill_info_dict["cu_seqlens_q"],
            self.prefill_info_dict["cu_seqlens_q"],
            max_seqlen_q=self.max_seq_len,
            max_seqlen_k=self.max_seq_len,
            attn_mask=forward_meta.attn_mask,
            causal=self.causal,
        )[0]

        return prefill_out

    def forward_decode(self, decode_qkv, k_cache_id, v_cache_id, forward_meta: ForwardMeta):
        q, k, v = self.apply_rope_decode(decode_qkv, forward_meta.rotary_embs)

        decode_out = flash_attn_kvcache_func(
            q,
            forward_meta.caches[k_cache_id],
            forward_meta.caches[v_cache_id],
            self.attention_metadata.seq_lens_decode,
            self.attention_metadata.block_table_decode,
            k,
            v,
            rotary_cos=None,
            rotary_sin=None,
            causal=self.causal,
            is_rotary_interleaved=True,
        )[0].squeeze(1)

        return decode_out

    @paddle.no_grad()
    def forward_native_backend(self, q, k, v, qkv, layer, forward_meta: ForwardMeta):

        layer_id = layer.layer_id
        k_cache_id = layer_id * 2
        v_cache_id = k_cache_id + 1

        if self.has_prefill and not self.has_decode:
            out = self.forward_prefill(qkv, layer_id, k_cache_id, v_cache_id, forward_meta)

        elif self.has_decode and not self.has_prefill:
            out = self.forward_decode(qkv, k_cache_id, v_cache_id, forward_meta)

        else:
            self.split_pd_qkv(qkv)
            prefill_output = self.forward_prefill(self.prefill_qkv, layer_id, k_cache_id, v_cache_id, forward_meta)
            decode_output = self.forward_decode(self.decode_qkv, k_cache_id, v_cache_id, forward_meta)
            self.merge_pd_output(prefill_output, decode_output)
            out = self.merged_output

        if qkv.dim() == 2:
            out = out.view([-1, self.num_heads * self.head_dim])

        return out
