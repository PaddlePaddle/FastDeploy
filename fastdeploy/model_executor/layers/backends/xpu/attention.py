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
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)
from fastdeploy.model_executor.ops.xpu import block_attn

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

from fastdeploy.model_executor.ops.xpu import split_rope_kvcache, block_attn_decouple


@dataclass
class XPUAttentionMetadata(AttentionMetadata):
    """
    XPUAttentionMetadata
    """

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None

    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)


class XPUAttentionBackend(AttentionBackend):
    """
    XPUAttentionBackend backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: XPUAttentionMetadata

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
        XPUAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: XPUAttentionMetadata = None
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
            fd_config.model_config, "use_3d_rope", False
        )
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.kv_num_heads: int = kv_num_heads
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # pd_disaggregation
        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.rank, self.device_id = init_rank_and_device_id(fd_config)

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attntion metadata hence all layers in the forward pass can reuse it."""
        metadata = XPUAttentionMetadata()
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
        metadata.pre_caches_length = forward_meta.pre_caches_length

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk" and not forward_meta.is_profiling:
            if not self.keep_pd_step_flag:
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

        self.attention_metadata: AttentionMetadata = metadata

    def get_attention_meta(self) -> AttentionMetadata:
        """get_attention_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[list, list]:
        """
        Calculate kv cache shape
        """
        key_cache_shape = value_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        return key_cache_shape, value_cache_shape

    def decouple_block_attn(
        self,
        qkv,
        key_cache,
        value_cache,
        cum_offsets,
        rotary_embs,
        block_tables,
        prefix_block_tables,
        len_info_cpu,
        encoder_seq_lod_cpu,
        decoder_seq_lod_cpu,
        encoder_kv_lod_cpu,
        encoder_batch_map_cpu,
        decoder_context_len_cpu,
        decoder_context_len_cache_cpu,
        decoder_batch_map_cpu,
        prefix_len_cpu,
        encoder_seq_lod,
        decoder_seq_lod,
        encoder_kv_lod,
        encoder_batch_map,
        decoder_context_len,
        decoder_context_len_cache,
        decoder_batch_map,
        prefix_len,
        k_scales,
        v_scales,
        k_scales_inv,
        v_scales_inv,
        k_zeros,
        v_zeros,
        shift,
        smooth,
        q_norm_weight,
        k_norm_weight,
        kv_signal_data_cpu,
        cachekv_signal_thread_cpu,
        use_neox_rotary_style,
        rope_3d):
        
        # has_zp目前没有模型用（c8模型也没有用上），smooth和shift也是hardcode为None，以下代码先注释掉
        # is_cache_int8 = key_cache.dtype == paddle.int8
        # has_zp = k_zeros is not None and v_zeros is not None
        # is_prefix_cache = len_info_cpu[5] > 0
        #
        # token_num = qkv.shape[0]
        # head_dim = key_cache.shape[3]
        # total_num_head = qkv.shape[-1] // head_dim
        # kv_num_heads = key_cache.shape[1]
        # num_heads = total_num_head - 2 * kv_num_heads
        # hidden_dim = num_heads * head_dim
        #
        # enc_batch = len_info_cpu[0]
        # dec_batch = len_info_cpu[1]
        # total_enc_len = len_info_cpu[2]
        # total_dec_len = token_num - total_enc_len
        
        q_enc, k_enc, v_enc, q_dec, k_dec, v_dec = split_rope_kvcache(
            qkv,
            key_cache,
            value_cache,
            cum_offsets,
            rotary_embs,
            block_tables,
            len_info_cpu,
            encoder_seq_lod_cpu,
            decoder_seq_lod_cpu,
            encoder_batch_map_cpu,
            decoder_context_len_cpu,
            decoder_context_len_cache_cpu,
            decoder_batch_map_cpu,
            prefix_len_cpu,
            encoder_seq_lod,
            decoder_seq_lod,
            encoder_batch_map,
            decoder_context_len,
            decoder_context_len_cache,
            decoder_batch_map,
            prefix_len,
            k_scales,
            v_scales,
            k_zeros,
            v_zeros,
            q_norm_weight,
            k_norm_weight,
            kv_signal_data_cpu,
            cachekv_signal_thread_cpu,
            use_neox_rotary_style,
            rope_3d)
        
        # has_zp目前没有模型用（c8模型也没有用上），以下代码先注释掉
        # q = q * k_scales_inv
        # if is_cache_int8 and has_zp:
        #     if enc_batch > 0 and is_prefix_cache:
        #         origin_shape = q_enc.shape
        #         q_enc_reshaped = paddle.view(
        #             q_enc,
        #             [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
        #         q_enc_reshaped = q_enc_reshaped * paddle.view(k_scales_inv, [1, kv_num_heads, 1, head_dim])
        #         q_enc = paddle.view(q_enc_reshaped, origin_shape)
        #
        #     if dec_batch > 0:
        #         origin_shape = q_dec.shape
        #         q_dec_reshaped = paddle.view(
        #             q_dec,
        #             [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
        #         q_dec_reshaped = q_dec_reshaped * paddle.view(k_scales_inv, [1, kv_num_heads, 1, head_dim])
        #         q_dec = paddle.view(q_dec_reshaped, origin_shape)
                
        out = block_attn_decouple(
            q_enc,
            k_enc,
            v_enc,
            q_dec,
            k_dec,
            v_dec,
            key_cache,
            value_cache,
            block_tables,
            prefix_block_tables,
            len_info_cpu,
            encoder_seq_lod_cpu,
            decoder_seq_lod_cpu,
            encoder_kv_lod_cpu,
            encoder_batch_map_cpu,
            decoder_context_len_cpu,
            decoder_context_len_cache_cpu,
            decoder_batch_map_cpu,
            encoder_seq_lod,
            decoder_seq_lod,
            encoder_kv_lod,
            encoder_batch_map,
            decoder_context_len,
            decoder_batch_map,
            k_scales_inv,
            v_scales_inv,
            k_zeros,
            v_zeros)
        
        # has_zp目前没有模型用（c8模型也没有用上），smooth和shift也是hardcode为None，以下代码先注释掉
        # if enc_batch > 0:
        #     if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
        #         sliced_out = out[:total_enc_len, :]
        #         origin_shape = sliced_out.shape
        #     if is_cache_int8 and has_zp and is_prefix_cache:
        #         # out = (out - v_zeros) * v_scales_inv
        #         out_reshaped = paddle.view(
        #             sliced_out,
        #             [total_enc_len, kv_num_heads, num_heads // kv_num_heads, head_dim]) - paddle.view(v_zeros, [1, kv_num_heads, 1, head_dim])
        #         out_reshaped = out_reshaped * paddle.view(v_scales_inv, [1, kv_num_heads, 1, head_dim])
        #         sliced_out = paddle.view(out_reshaped, origin_shape)
        #     if shift:
        #         sliced_out = sliced_out + shift
        #     if smooth:
        #         sliced_out = sliced_out * smooth
        #     if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
        #         out[:total_enc_len, :] = sliced_out
        #     
        # if dec_batch > 0:
        #     if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
        #         sliced_out = out[total_enc_len:, :]
        #         origin_shape = sliced_out.shape
        #     if is_cache_int8 and has_zp:
        #         # out = (out - v_zeros) * v_scales_inv
        #         out_reshaped = paddle.view(
        #             sliced_out,
        #             [total_dec_len, kv_num_heads, num_heads // kv_num_heads, head_dim])
        #         if v_zeros is not None:
        #             out_reshaped = out_reshaped - paddle.view(v_zeros, [1, kv_num_heads, 1, head_dim])
        #         out_reshaped = out_reshaped * paddle.view(v_scales_inv, [1, kv_num_heads, 1, head_dim])
        #         sliced_out = paddle.view(out_reshaped, origin_shape)
        #     if shift:
        #         sliced_out = sliced_out + shift
        #     if smooth:
        #         sliced_out = sliced_out * smooth
        #     if is_cache_int8 and has_zp and is_prefix_cache or shift or smooth:
        #         out[total_enc_len:, :] = sliced_out
            
        return out

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
        forward_mixed
        """
        metadata = self.attention_metadata
        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        cache_k_scale = getattr(layer, "cache_k_scale", None)
        cache_v_scale = getattr(layer, "cache_v_scale", None)
        cache_k_out_scale = getattr(layer, "cache_k_out_scale", None)
        cache_v_out_scale = getattr(layer, "cache_v_out_scale", None)
        cache_k_zp = getattr(self, "cache_k_zp", None)
        cache_v_zp = getattr(self, "cache_v_zp", None)

        if layer.use_qk_norm:
            q_norm_weight = layer.q_norm_weight
            k_norm_weight = layer.k_norm_weight
        else:
            q_norm_weight = None
            k_norm_weight = None

        # func = block_attn
        func = self.decouple_block_attn
        res = func(
            qkv,
            forward_meta.caches[2 * layer.layer_id],
            forward_meta.caches[2 * layer.layer_id + 1],
            forward_meta.cum_offsets,
            metadata.rotary_embs,
            metadata.block_tables,
            forward_meta.prefix_block_tables,
            forward_meta.len_info_cpu,
            forward_meta.encoder_seq_lod_cpu,
            forward_meta.decoder_seq_lod_cpu,
            forward_meta.encoder_kv_lod_cpu,
            forward_meta.encoder_batch_map_cpu,
            forward_meta.decoder_context_len_cpu,
            forward_meta.decoder_context_len_cache_cpu,
            forward_meta.decoder_batch_map_cpu,
            forward_meta.prefix_len_cpu,
            forward_meta.encoder_seq_lod,
            forward_meta.decoder_seq_lod,
            forward_meta.encoder_kv_lod,
            forward_meta.encoder_batch_map,
            forward_meta.decoder_context_len,
            forward_meta.decoder_context_len_cache,
            forward_meta.decoder_batch_map,
            forward_meta.prefix_len,
            cache_k_scale,
            cache_v_scale,
            cache_k_out_scale,
            cache_v_out_scale,
            cache_k_zp,
            cache_v_zp,
            None,  # shift
            None,  # smooth
            q_norm_weight,
            k_norm_weight,
            metadata.kv_signal_data_list[layer.layer_id],
            forward_meta.kv_signal_sender,
            layer.use_neox_rotary_style,
            self.rope_3d,
        )
        
        return res
