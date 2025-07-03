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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded

from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block, init_signal_layerwise,
    open_shm_and_get_meta_signal)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (decode_mla_write_cache,
                                                   multi_head_latent_attention,
                                                   prefill_mla_write_cache)

if TYPE_CHECKING:
    from paddle._typing.dtype_like import _DTypeLiteral

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend, AttentionMetadata)
from fastdeploy.worker.forward_meta import ForwardMeta


def yarn_get_mscale(scale=1, mscale=1):
    """
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class MLAAttentionMetadata(AttentionMetadata):
    """
    MLAAttentionMetadata for Multi-Layer Attention
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


class MLAAttentionBackend(AttentionBackend):
    """
    MLA Attention Backend implementation.
    """

    def __init__(self, fd_config: FDConfig, kv_num_heads: int, num_heads: int,
                 head_dim: int) -> None:
        """
        MLAAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: MLAAttentionMetadata = None

        # 基础配置
        self.block_size: int = fd_config.parallel_config.block_size
        self.max_seq_len: int = fd_config.parallel_config.max_model_len
        self.rope_theta: float = (10000.0
                                  if fd_config.model_config.rope_theta is None
                                  else fd_config.model_config.rope_theta)
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method: str = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.rank: int = fd_config.parallel_config.tensor_parallel_rank

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_layers

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.deepseekv3.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.deepseekv3.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.deepseekv3.qk_nope_head_dim \
            + fd_config.model_config.deepseekv3.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        if fd_config.model_config.deepseekv3.rope_scaling:
            mscale_all_dim = fd_config.model_config.deepseekv3.rope_scaling.get(
                "mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.deepseekv3.rope_scaling[
                "factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        # pd_disaggregation
        self.use_pd_disaggregation: int = int(
            os.getenv("FLAGS_use_pd_disaggregation", 0))
        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if self.device_id is None:
            self.device_id = self.rank
        else:
            self.device_id = self.device_id.split(",")[self.rank]

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata hence all layers in the forward pass can reuse it."""
        metadata = MLAAttentionMetadata()
        metadata.encoder_block_shape_q = 64
        metadata.decoder_block_shape_q = 16
        metadata.max_partition_size = 32768
        metadata.encoder_max_partition_size = self.max_seq_len
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
        metadata.pre_caches_length = forward_meta.pre_caches_length

        (
            metadata.encoder_batch_ids,
            metadata.encoder_tile_ids_per_batch,
            metadata.encoder_num_blocks,
            metadata.kv_batch_ids,
            metadata.kv_tile_ids_per_batch,
            metadata.kv_num_blocks,
            metadata.decoder_batch_ids,
            metadata.decoder_tile_ids_per_batch,
            metadata.decoder_num_blocks,
            metadata.max_len_kv,
            metadata.set_max_lengths,
        ) = get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.cum_offsets,
            metadata.encoder_block_shape_q,
            metadata.decoder_block_shape_q,
            self.num_heads // self.kv_num_heads,
            self.block_size,
            self.speculate_max_draft_token_num + 1,
        )

        # MLA
        metadata.max_enc_len_this_time = metadata.set_max_lengths[1]
        metadata.max_dec_len_this_time = metadata.set_max_lengths[2]

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.use_pd_disaggregation:
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, int(self.device_id), self.keep_pd_step_flag)

        self.attention_metadata: AttentionMetadata = metadata

    def get_attntion_meta(self) -> AttentionMetadata:
        """get_attntion_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(self,
                           max_num_blocks: int) -> Tuple[int, int, int, int]:
        """
        Calculate kv cache shape for MLA
        """
        return (max_num_blocks, 1, self.block_size,
                self.kv_lora_rank + self.qk_rope_head_dim)

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Prefill阶段的前向传播
        """
        metadata = self.attention_metadata

        if self.use_pd_disaggregation:
            metadata.kv_signal_data_list[
                layer.layer_id] = init_signal_layerwise(
                    metadata.kv_signal_metadata,
                    layer.layer_id + self.start_layer_index)

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(
            forward_meta, 'caches') else None

        # 写入缓存
        prefill_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.padding_offset,
            forward_meta.cum_offsets,
            metadata.block_tables,
            "none",
            getattr(forward_meta, 'max_input_length', -1),
        )

        # Flash注意力计算
        fmha_out = flash_attn_unpadded(
            q,
            k,
            v,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
            metadata.max_enc_len_this_time,
            metadata.max_enc_len_this_time,
            self.attn_softmax_scale,
            causal=True,
            training=False,
        )[0]

        return fmha_out

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Decode阶段的前向传播
        """
        metadata = self.attention_metadata

        if self.use_pd_disaggregation:
            metadata.kv_signal_data_list[
                layer.layer_id] = init_signal_layerwise(
                    metadata.kv_signal_metadata,
                    layer.layer_id + self.start_layer_index)

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(
            forward_meta, 'caches') else None

        # 获取推测解码参数
        speculate_decoder = self.speculative_method is not None
        speculate_max_tokens = self.speculate_max_draft_token_num

        # 写入缓存
        decode_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_encoder,
            forward_meta.padding_offset,
            forward_meta.cum_offsets,
            metadata.block_tables,
            "none",
            self.max_seq_len,
            speculate_decoder,
        )

        # 多头潜在注意力计算
        fmha_out = multi_head_latent_attention(
            q,
            latent_cache,
            latent_cache,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.cu_seqlens_q,
            forward_meta.padding_offset,
            forward_meta.cum_offsets,
            metadata.block_tables,
            metadata.encoder_batch_ids,
            metadata.encoder_tile_ids_per_batch,
            metadata.encoder_num_blocks,
            metadata.kv_batch_ids,
            metadata.kv_tile_ids_per_batch,
            metadata.kv_num_blocks,
            metadata.decoder_batch_ids,
            metadata.decoder_tile_ids_per_batch,
            metadata.decoder_num_blocks,
            metadata.
            decoder_num_blocks,  # PaddleNLP 传入的是 decoder_num_blocks_cpu
            metadata.max_enc_len_this_time,
            metadata.max_dec_len_this_time,
            metadata.max_len_kv,
            None,  # attn_mask
            None,  # qkv_bias
            None,  # qkv_out_scales
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # out_shifts
            None,  # out_smooths
            metadata._fuse_kernel_compute_dtype,
            "none",  # cache_quant_type
            self.kv_lora_rank,
            self.max_seq_len,
            self.attn_softmax_scale,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            0.0,  # out_linear_in_scale
            speculate_max_tokens,
            True,  # causal
            speculate_decoder,
        )

        return fmha_out

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
    ) -> paddle.Tensor:
        """
        Mixed模式的前向传播
        """
        metadata = self.attention_metadata
        speculate_decoder = self.speculative_method is not None
        speculate_max_tokens = self.speculate_max_draft_token_num

        decode_stage = forward_meta.is_decode_batch
        prefill_stage = not (forward_meta.is_decode_batch)

        if self.use_pd_disaggregation:
            metadata.kv_signal_data_list[
                layer.layer_id] = init_signal_layerwise(
                    metadata.kv_signal_metadata,
                    layer.layer_id + self.start_layer_index)

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(
            forward_meta, 'caches') else None

        if prefill_stage:
            # 写入缓存
            prefill_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.padding_offset,
                forward_meta.cum_offsets,
                metadata.block_tables,
                "none",
                self.max_seq_len,
            )

            # FA
            fmha_out = flash_attn_unpadded(
                q,
                k,
                v,
                forward_meta.cu_seqlens_q,
                forward_meta.cu_seqlens_k,
                metadata.max_enc_len_this_time,
                metadata.max_enc_len_this_time,
                self.attn_softmax_scale,
                causal=True,
                training=False,
            )[0]

            return fmha_out

        # Decode
        if decode_stage:
            # mla写入缓存
            decode_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_encoder,
                forward_meta.padding_offset,
                forward_meta.cum_offsets,
                metadata.block_tables,
                "none",
                self.max_seq_len,
                speculate_decoder,
            )

            # 多头潜在注意力计算
            fmha_out = multi_head_latent_attention(
                q,
                latent_cache,
                latent_cache,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.cu_seqlens_q,
                forward_meta.padding_offset,
                forward_meta.cum_offsets,
                metadata.block_tables,
                metadata.encoder_batch_ids,
                metadata.encoder_tile_ids_per_batch,
                metadata.encoder_num_blocks,
                metadata.kv_batch_ids,
                metadata.kv_tile_ids_per_batch,
                metadata.kv_num_blocks,
                metadata.decoder_batch_ids,
                metadata.decoder_tile_ids_per_batch,
                metadata.decoder_num_blocks,
                metadata.
                decoder_num_blocks,  # PaddleNLP 传入的是 decoder_num_blocks_cpu
                metadata.max_enc_len_this_time,
                metadata.max_dec_len_this_time,
                metadata.max_len_kv,
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # out_shifts
                None,  # out_smooths
                metadata._fuse_kernel_compute_dtype,
                "none",  # cache_quant_type
                self.kv_lora_rank,
                self.max_seq_len,
                self.attn_softmax_scale,
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                0.0,  # out_linear_in_scale
                speculate_max_tokens,
                True,  # causal
                speculate_decoder,
            )

            return fmha_out
