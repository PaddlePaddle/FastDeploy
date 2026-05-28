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

import paddle

paddle.enable_compat(scope={"flash_mla"})  # Enable paddle.enable_compat before importing flash_mla
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded
from paddleformers.utils.log import logger

try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except Exception as e:
    logger.debug(f"flash_attention_v3_varlen not available: {e}")
    flash_attention_v3_varlen = None

from fastdeploy import envs
from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        decode_mla_write_cache,
        multi_head_latent_attention,
        prefill_mla_write_cache,
    )

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import triton
import triton.language as tl

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.spec_decode import SpecMethod


@enable_compat_on_triton_kernel
@triton.jit()
def extract_kernel(
    q,
    cu_seqlens_q,
    seq_lens_encoder,
    seq_lens_decoder,
    output,
    cache_seqlens,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(axis=0)
    cache_kv_len = tl.load(seq_lens_decoder + batch_id)

    # 这个batch不是decoder，所以不需要动弹
    if cache_kv_len <= 0:
        return

    cu_len_this_batch = tl.load(cu_seqlens_q + batch_id)

    read_offsets = tl.arange(0, BLOCK_SIZE)
    q += cu_len_this_batch * HIDDEN_DIM

    row_data = tl.load(q + read_offsets, mask=read_offsets < HIDDEN_DIM)

    output += batch_id * HIDDEN_DIM

    tl.store(output + read_offsets, row_data, mask=read_offsets < HIDDEN_DIM)

    tl.store(cache_seqlens + batch_id, cache_kv_len + 1)


def extract_decoder_token_from_q(
    q: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
):
    assert len(q.shape) == 2
    assert len(cu_seqlens_q.shape) == 1
    assert len(seq_lens_encoder.shape) == 1
    assert len(seq_lens_decoder.shape) == 1

    max_bsz = seq_lens_decoder.shape[0]

    hidden_dim = q.shape[-1]
    out = paddle.empty([max_bsz, hidden_dim], dtype=q.dtype)

    cache_seqlens = paddle.zeros_like(seq_lens_decoder)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    grid = (max_bsz,)

    extract_kernel[grid](
        q,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        out,
        cache_seqlens,
        hidden_dim,
        BLOCK_SIZE,
    )

    return out, cache_seqlens


@enable_compat_on_triton_kernel
@triton.jit()
def insert_kernel(
    decoder_res,
    cu_seqlens_q,
    seq_lens_encoder,
    seq_lens_decoder,
    output,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(axis=0)
    cache_kv_len = tl.load(seq_lens_decoder + batch_id)

    # 这个batch不是decoder，所以不需要动弹
    if cache_kv_len <= 0:
        return

    cu_len_this_batch = tl.load(cu_seqlens_q + batch_id)

    read_offsets = tl.arange(0, BLOCK_SIZE)

    decoder_res += batch_id * HIDDEN_DIM

    row_data = tl.load(decoder_res + read_offsets, mask=read_offsets < HIDDEN_DIM)

    output += cu_len_this_batch * HIDDEN_DIM

    tl.store(output + read_offsets, row_data, mask=read_offsets < HIDDEN_DIM)


def insert_decoder_result_back(
    decoder_result: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    mixed_token_num,
):
    assert len(decoder_result.shape) == 4
    assert len(cu_seqlens_q.shape) == 1
    assert len(seq_lens_encoder.shape) == 1

    max_bsz = seq_lens_encoder.shape[0]

    hidden_dim = decoder_result.shape[-2] * decoder_result.shape[-1]
    out = paddle.zeros([mixed_token_num, hidden_dim], dtype=decoder_result.dtype)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    grid = (max_bsz,)

    insert_kernel[grid](
        decoder_result,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        out,
        hidden_dim,
        BLOCK_SIZE,
    )

    return out


def yarn_get_mscale(scale=1, mscale=1):
    """ """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class MLAAttentionMetadata(AttentionMetadata):
    """
    MLAAttentionMetadata for Multi-Layer Attention
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

    max_enc_len_this_time: Optional[paddle.Tensor] = None
    max_dec_len_this_time: Optional[paddle.Tensor] = None
    max_kv_len_this_time: Optional[paddle.Tensor] = None


class MLAAttentionBackend(AttentionBackend):
    """
    MLA Attention Backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: MLAAttentionMetadata
    flash_attn_func: callable = None

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
        MLAAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: MLAAttentionMetadata = None

        # 基础配置
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = fd_config.enable_rope_3d_runtime
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method == SpecMethod.MTP)

        self.num_heads: int = num_heads
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        if fd_config.model_config.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.rope_scaling["factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.useless_tensor = paddle.randn([1]).cast("int32")

        if self.flash_attn_func is None:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                print("The current platform supports Flash Attention V3.")
                self.flash_attn_kwargs = {"softmax_scale": self.attn_softmax_scale}
            else:
                self.flash_attn_func = flash_attn_unpadded
                self.flash_attn_kwargs = {"scale": self.attn_softmax_scale, "training": False}
                print(
                    "The current platform does not support Flash Attention V3, so Flash Attention V2 will be used instead."
                )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata hence all layers in the forward pass can reuse it."""
        metadata = MLAAttentionMetadata()
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

        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            self.useless_tensor,  # not used in mla
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            -1,  # not need.
            -1,  # not need.
            -1,  # not need.
            self.block_size,
        )
        # MLA
        metadata.max_enc_len_this_time = forward_meta.max_len_tensor_cpu[1]
        metadata.max_dec_len_this_time = forward_meta.max_len_tensor_cpu[2]
        metadata.max_kv_len_this_time = forward_meta.max_len_tensor_cpu[5]

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
            if (
                not self.keep_pd_step_flag
                and not forward_meta.is_dummy_or_profile_run
                and not envs.FD_PD_TRANSFER_VIA_STORAGE
            ):
                init_kv_signal_per_query(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_decoder,
                    self.rank,
                    self.num_layers + self.num_layers_draft_model,
                )
        elif self.pd_disaggregation_mode == "per_query" and not envs.FD_PD_TRANSFER_VIA_STORAGE:
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
    ) -> Tuple[int, int, int, int]:
        """
        Calculate kv cache shape for MLA
        """
        key_cache_shape = [max_num_blocks, 1, self.block_size, self.kv_lora_rank + self.qk_rope_head_dim]
        value_cache_shape = []
        return key_cache_shape, value_cache_shape

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

        if self.pd_disaggregation_mode == "per_query" and not envs.FD_PD_TRANSFER_VIA_STORAGE:
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 写入缓存
        prefill_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            metadata.kv_signal_data_list[layer.layer_id],
            "none",
            getattr(forward_meta, "max_input_length", -1),
        )

        # Flash注意力计算
        fmha_out = self.flash_attn_func(
            q,
            k,
            v,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
            metadata.max_enc_len_this_time,
            metadata.max_enc_len_this_time,
            causal=self.causal,
            **self.flash_attn_kwargs,
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

        if self.pd_disaggregation_mode == "per_query" and not envs.FD_PD_TRANSFER_VIA_STORAGE:
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

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
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
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
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.cu_seqlens_q,
            forward_meta.batch_id_per_token,
            metadata.block_tables,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            metadata.max_dec_len_this_time,
            metadata.max_kv_len_this_time,
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

        if self.pd_disaggregation_mode == "per_query" and not envs.FD_PD_TRANSFER_VIA_STORAGE:
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        if k is not None:
            prefill_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                metadata.kv_signal_data_list[layer.layer_id],
                "none",
                self.max_seq_len,
            )
            # FA
            fmha_out = self.flash_attn_func(
                q,
                k,
                v,
                forward_meta.cu_seqlens_q,
                forward_meta.cu_seqlens_k,
                metadata.max_enc_len_this_time,
                metadata.max_enc_len_this_time,
                causal=self.causal,
                **self.flash_attn_kwargs,
            )[0]

            return fmha_out

        # Decode
        if k is None:
            decode_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_encoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                "none",
                self.max_seq_len,
                speculate_decoder,
            )

            if int(os.getenv("USE_FLASH_MLA", "0")) == 0:
                assert self.num_heads <= 64, "paddle mla attention support failed"
                # 多头潜在注意力计算
                fmha_out = multi_head_latent_attention(
                    q,
                    latent_cache,
                    latent_cache,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.cu_seqlens_q,
                    forward_meta.batch_id_per_token,
                    metadata.block_tables,
                    forward_meta.kv_batch_ids,
                    forward_meta.kv_tile_ids_per_batch,
                    forward_meta.kv_num_blocks_x_cpu,
                    forward_meta.decoder_batch_ids,
                    forward_meta.decoder_tile_ids_per_batch,
                    forward_meta.decoder_num_blocks_device,
                    forward_meta.decoder_chunk_size_device,
                    metadata.max_dec_len_this_time,
                    metadata.max_kv_len_this_time,
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
            else:
                import flash_mla

                decoder_q, cache_seqlens = extract_decoder_token_from_q(
                    q,
                    forward_meta.cu_seqlens_q,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                )

                tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()
                token_num = q.shape[0]
                decoder_q.reshape_([-1, 1, self.num_heads, 576])

                new_cache_shape = latent_cache.shape
                assert new_cache_shape[1] == 1
                new_cache_shape[1], new_cache_shape[2] = new_cache_shape[2], new_cache_shape[1]

                decoder_res, _ = flash_mla.flash_mla_with_kvcache(
                    decoder_q,
                    # 外面的开源仓库的kv cache存储格式和FD的不同
                    # 幸好这里缓存的头是1，直接view即可，否则上上下下要改很多！
                    latent_cache.view(new_cache_shape),
                    metadata.block_tables,
                    cache_seqlens,
                    512,  # t.dv,
                    tile_scheduler_metadata,
                    num_splits,
                    softmax_scale=self.attn_softmax_scale,
                    causal=True,
                )

                final_res = insert_decoder_result_back(
                    decoder_res,
                    forward_meta.cu_seqlens_q,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    token_num,
                )

                return final_res
