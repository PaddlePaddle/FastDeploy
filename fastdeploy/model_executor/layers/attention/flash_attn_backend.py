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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded

try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except:
    flash_attention_v3_varlen = None

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output
else:
    merge_prefill_decode_output = None

import os


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """
    FlashAttentionMetadata
    """

    rotary_embs: Optional[paddle.Tensor] = None
    block_tables: Optional[paddle.Tensor] = None

    cu_seqlens_q: paddle.Tensor = None
    cu_seqlens_k: paddle.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    _fuse_kernel_compute_dtype: str = "bf16"
    _dtype: paddle.dtype = paddle.bfloat16

    max_len_tensor_cpu: paddle.Tensor = None
    max_len_tensor_cpu_decoder: paddle.Tensor = None


class FlashAttentionBackend(AttentionBackend):
    """
    FlashAttentionBackend backend implementation
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: FlashAttentionMetadata
    flash_attn_func: callable = None

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        """
        FlashAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: FlashAttentionMetadata = None
        self.max_seq_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.attn_outputsize_tp = self.num_heads * self.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        if fd_config.parallel_config.expert_parallel_rank is None:
            fd_config.parallel_config.expert_parallel_rank = 0

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        if self.flash_attn_func is None:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                print("The current platform supports Flash Attention V3.")
                self.flash_attn_kwargs = {}
            else:
                self.flash_attn_func = flash_attn_unpadded
                self.flash_attn_kwargs = {"scale": self.head_dim**-0.5, "training": False}
                print(
                    "The current platform does not support Flash Attention V3, so Flash Attention V2 will be used instead."
                )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", "32768"))
        self.zero_seq_enc_lens_for_decode = paddle.zeros(
            shape=[fd_config.scheduler_config.max_num_seqs, 1], dtype=paddle.int32
        )

    def get_attntion_meta(self):
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
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            key_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
            value_cache_shape = [
                max_num_blocks,
                self.kv_num_heads,
                self.block_size,
                self.head_dim // 2,
            ]
        return key_cache_shape, value_cache_shape

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = FlashAttentionMetadata()
        metadata.cu_seqlens_q = forward_meta.cu_seqlens_q
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.block_tables = forward_meta.block_tables
        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_cpu,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.group_size,
            self.block_size,
        )

        (
            metadata.cu_seqlens_k,
            metadata.pre_cache_batch_ids,
            metadata.pre_cache_tile_ids_per_batch,
            metadata.pre_cache_num_blocks_cpu,
            metadata.kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.max_len_tensor_cpu[2],
            self.block_size,
        )

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
            if not self.keep_pd_step_flag and not forward_meta.is_dummy_or_profile_run:
                init_kv_signal_per_query(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_decoder,
                    self.rank,
                    self.num_layers + self.num_layers_draft_model,
                )
        elif self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, int(self.device_id), self.keep_pd_step_flag
            )

        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        metadata.max_len_tensor_cpu = forward_meta.max_len_tensor_cpu
        metadata.max_len_tensor_cpu_decoder = paddle.clone(metadata.max_len_tensor_cpu)
        metadata.max_len_tensor_cpu_decoder[1] = 0

        self.attention_metadata = metadata

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ):
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        if metadata.max_len_tensor_cpu[1] > 0:
            q, k, v, _ = gqa_rope_write_cache(
                qkv,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.rotary_embs,
                forward_meta.seq_lens_this_time,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                metadata.block_tables,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                metadata.pre_cache_batch_ids,
                metadata.pre_cache_tile_ids_per_batch,
                metadata.pre_cache_num_blocks_cpu,
                getattr(layer, "cache_k_scale", None),
                getattr(layer, "cache_v_scale", None),
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                metadata.kv_signal_data_list[layer.layer_id],
                metadata.kv_token_num_cpu[0].item(),
                self.max_seq_len,
                getattr(layer, "cache_quant_type_str", "none"),
                self.rope_3d,
            )

            res_encoder = self.flash_attn_func(
                q,
                k,
                v,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                max_seqlen_q=forward_meta.max_len_tensor_cpu[0],
                max_seqlen_k=forward_meta.max_len_tensor_cpu[3],
                causal=self.causal,
                **self.flash_attn_kwargs,
            )[0].reshape([-1, self.attn_outputsize_tp])

        res_decoder = append_attention(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            self.zero_seq_enc_lens_for_decode,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            forward_meta.decoder_batch_ids,  # from buffer
            forward_meta.decoder_tile_ids_per_batch,  # from buffer
            forward_meta.decoder_num_blocks_cpu,
            metadata.max_len_tensor_cpu_decoder,
            metadata.rotary_embs,
            forward_meta.attn_mask,
            layer.qkv_bias,
            layer.qkv_scale,
            getattr(layer, "cache_k_scale", None),
            getattr(layer, "cache_v_scale", None),
            getattr(layer, "cache_k_out_scale", None),
            getattr(layer, "cache_v_out_scale", None),
            getattr(layer, "cache_k_zp", None),
            getattr(layer, "cache_v_zp", None),
            layer.linear_shift,
            layer.linear_smooth,
            forward_meta.attn_mask_offsets,
            metadata.kv_signal_data_list[layer.layer_id],
            getattr(layer, "q_norm_weight", None),
            getattr(layer, "k_norm_weight", None),
            getattr(layer, "sinks", None),
            getattr(layer, "rms_norm_eps", 1e-6),
            metadata._fuse_kernel_compute_dtype,
            getattr(layer, "cache_quant_type_str", "none"),
            layer.use_neox_rotary_style,
            self.rope_3d,
            self.max_seq_len,
            getattr(layer, "quant_max_bound", 0.0),
            getattr(layer, "quant_min_bound", 0.0),
            getattr(layer, "out_scale", -1.0),
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.max_partition_size,
            self.max_seq_len,
            self.speculate_max_draft_token_num + 1,
            self.causal,
            self.speculative_method is not None,
        )

        if metadata.max_len_tensor_cpu[1] > 0:
            merge_prefill_decode_output(
                res_encoder,
                res_decoder,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.cu_seqlens_q,
                self.num_heads,
                self.head_dim,
                self.speculate_max_draft_token_num + 1,
            )
            return res_encoder
        else:
            return res_decoder
