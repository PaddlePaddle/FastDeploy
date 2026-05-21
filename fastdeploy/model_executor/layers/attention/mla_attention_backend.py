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

from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)
from fastdeploy.platforms import current_platform

compiled_mla = None
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

# ============================================================================
# Fused Read-Cache + Interleave Kernel for Prefix Cache Support
#
# For each output token position t in [0, total_tokens):
#   - if cached: load latent vector from paged latent_cache (physical block)
#   - if new:    load from new_compressed_kv / new_k_pe at the given index
# A single kernel produces full_compressed_kv / full_k_pe directly, avoiding
# an intermediate (cached_kv_c, cached_k_pe) allocation and an extra launch.
# ============================================================================


def fused_read_cache_and_interleave_naive(
    latent_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    new_compressed_kv: paddle.Tensor,
    new_k_pe: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Python/Paddle reference of the fused read+interleave path.

    Takes with-cache ``cu_seqlens_k`` (cumsum of ``cached + new``
    per batch) plus ``cu_seqlens_q`` (new-only); the cached length of each
    batch is derived as ``k_len - new_len``.
    """
    bsz = cu_seqlens_k.shape[0] - 1
    cu_seqlens_k = cu_seqlens_k.tolist()
    cu_seqlens_q = cu_seqlens_q.tolist()
    total_tokens = cu_seqlens_k[bsz]

    full_compressed_kv = paddle.empty([total_tokens, kv_lora_rank], dtype=new_compressed_kv.dtype)
    full_k_pe = paddle.empty([total_tokens, qk_rope_head_dim], dtype=new_k_pe.dtype)

    out_pos = 0
    for b in range(bsz):
        k_len = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
        q_len = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
        cache_token = k_len - q_len
        # cached tokens first
        for t in range(cache_token):
            block_idx = t // block_size
            block_offset = t % block_size
            physical_block_id = block_tables[b, block_idx].item()
            latent_vec = latent_cache[physical_block_id, 0, block_offset, :]
            full_compressed_kv[out_pos] = latent_vec[:kv_lora_rank]
            full_k_pe[out_pos] = latent_vec[kv_lora_rank:]
            out_pos += 1
        # new tokens after cached
        new_base = cu_seqlens_q[b]
        for t in range(q_len):
            full_compressed_kv[out_pos] = new_compressed_kv[new_base + t]
            full_k_pe[out_pos] = new_k_pe[new_base + t]
            out_pos += 1

    assert (
        out_pos == total_tokens
    ), f"fused_read_cache_and_interleave_naive: out_pos={out_pos} != total_tokens={total_tokens}"
    return full_compressed_kv, full_k_pe


@triton.jit()
def _fused_read_interleave_kernel(
    latent_cache_ptr,  # [num_blocks, 1, block_size, LATENT_DIM]
    new_kv_c_ptr,  # [total_new, kv_lora_rank]
    new_k_pe_ptr,  # [total_new, qk_rope_head_dim]
    cu_total_ptr,  # [bsz+1] int32  (= forward_meta.cu_seqlens_k)
    cu_new_ptr,  # [bsz+1] int32  (= cu_seqlens_q)
    block_tables_ptr,  # [bsz, max_blocks_per_seq] int32
    out_kv_c_ptr,  # [total_tokens, kv_lora_rank]
    out_k_pe_ptr,  # [total_tokens, qk_rope_head_dim]
    total_tokens,
    bsz,
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    LOG2_MAX_BSZ: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Tiled fused read+interleave kernel.

    Each program handles ``BLOCK_M`` contiguous output tokens. The owning
    batch of each token is recovered via an in-kernel binary search on the
    cumsum ``cu_total`` (O(log bsz), fully L1-resident because bsz+1 is
    small), avoiding a host-side ``repeat_interleave`` of shape
    ``[total_tokens]``.

    kv_c / k_pe are loaded as two separate vectors because ``kv_lora_rank``
    and ``qk_rope_head_dim`` are individually power-of-2 but their sum
    (``LATENT_DIM``) is generally not (e.g. 512+64=576).
    """
    pid = tl.program_id(axis=0)
    kv_c_offs = tl.arange(0, kv_lora_rank)
    k_pe_offs = tl.arange(0, qk_rope_head_dim)

    # Unrolled at compile time - BLOCK_M is small
    for m in tl.static_range(0, BLOCK_M):
        token_idx = pid * BLOCK_M + m
        if token_idx < total_tokens:
            # Binary search: find largest b such that cu_total[b] <= token_idx.
            # Loop bound LOG2_MAX_BSZ is compile-time; cu_total fits in L1.
            lo = 0
            hi = bsz - 1
            for _ in tl.static_range(0, LOG2_MAX_BSZ):
                mid = (lo + hi + 1) // 2
                cu_mid = tl.load(cu_total_ptr + mid)
                if cu_mid <= token_idx:
                    lo = mid
                else:
                    hi = mid - 1
            b = lo

            ct_b = tl.load(cu_total_ptr + b)
            ct_b1 = tl.load(cu_total_ptr + b + 1)
            cn_b = tl.load(cu_new_ptr + b)
            cn_b1 = tl.load(cu_new_ptr + b + 1)

            k_len = ct_b1 - ct_b
            nn = cn_b1 - cn_b
            nc = k_len - nn
            local_t = token_idx - ct_b

            if local_t < nc:
                # cached token: walk paged latent_cache via block_tables
                block_idx = local_t // block_size
                block_off = local_t % block_size
                pbid = tl.load(block_tables_ptr + b * max_blocks_per_seq + block_idx)
                base = latent_cache_ptr + (pbid * block_size + block_off) * LATENT_DIM
                kv_c_val = tl.load(base + kv_c_offs)
                k_pe_val = tl.load(base + kv_lora_rank + k_pe_offs)
            else:
                # new token: linear region in new_compressed_kv / new_k_pe
                src = cn_b + (local_t - nc)
                kv_c_val = tl.load(new_kv_c_ptr + src * kv_lora_rank + kv_c_offs)
                k_pe_val = tl.load(new_k_pe_ptr + src * qk_rope_head_dim + k_pe_offs)

            tl.store(out_kv_c_ptr + token_idx * kv_lora_rank + kv_c_offs, kv_c_val)
            tl.store(out_k_pe_ptr + token_idx * qk_rope_head_dim + k_pe_offs, k_pe_val)


def fused_read_cache_and_interleave_triton(
    latent_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    new_compressed_kv: paddle.Tensor,
    new_k_pe: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Triton-accelerated fused read+interleave.

    Single kernel launch with zero Python metadata computation. All per-batch
    geometry is derived in-kernel from ``cu_seqlens_k`` and ``cu_seqlens_q``
    via a compile-time-bounded binary search. Only one scalar D2H is needed:
    the final cumsum entry, used to size the output tensors.

    ``BLOCK_M=4`` and ``num_warps=8`` are tuned defaults (autotune is avoided
    because ``triton.testing.do_bench`` is incompatible with paddle worker
    subprocesses during ``profile_run``).
    """
    bsz = cu_seqlens_k.shape[0] - 1
    total_tokens = cu_seqlens_k[-1].item()

    full_compressed_kv = paddle.empty([total_tokens, kv_lora_rank], dtype=new_compressed_kv.dtype)
    full_k_pe = paddle.empty([total_tokens, qk_rope_head_dim], dtype=new_k_pe.dtype)
    if total_tokens == 0:
        return full_compressed_kv, full_k_pe

    max_blocks_per_seq = block_tables.shape[1]
    # Compile-time binary-search loop bound. ceil(log2(bsz)), clamped >=1.
    log2_max_bsz = max(1, (bsz - 1).bit_length()) if bsz > 1 else 1

    # Default tuned config: BLOCK_M=4 is robust across decode / prefill /
    # chunk-prefill; num_warps=8 matches the 512-wide kv_lora_rank vector.
    BLOCK_M = 4
    num_warps = 8

    grid = ((total_tokens + BLOCK_M - 1) // BLOCK_M,)
    _fused_read_interleave_kernel[grid](
        latent_cache,
        new_compressed_kv,
        new_k_pe,
        cu_seqlens_k,
        cu_seqlens_q,
        block_tables,
        full_compressed_kv,
        full_k_pe,
        total_tokens,
        bsz,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        LATENT_DIM=kv_lora_rank + qk_rope_head_dim,
        LOG2_MAX_BSZ=log2_max_bsz,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
    )
    return full_compressed_kv, full_k_pe


def fused_read_cache_and_interleave(*args, **kwargs):
    """Unified entry. ``FD_MLA_USE_NAIVE=1`` forces the Python reference path."""
    if os.environ.get("FD_MLA_USE_NAIVE", "0") == "1":
        return fused_read_cache_and_interleave_naive(*args, **kwargs)
    return fused_read_cache_and_interleave_triton(*args, **kwargs)


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
    assert seq_lens_encoder.shape == seq_lens_decoder.shape

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
    out = paddle.empty([mixed_token_num, hidden_dim], dtype=decoder_result.dtype)

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

    # For prefix cache and chunked prefill support
    # ``cu_seqlens_k`` semantics produced by ``get_padding_offset``.
    # forward_meta.cu_seqlens_k: Optional[paddle.Tensor] = None
    max_seqlen_k: int = 0


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
        self.heads_need_padding = False
        if self.num_heads < 64 and fd_config.parallel_config.tensor_parallel_size > 1:
            self.padding_num_heads = 64 - self.num_heads
            self.heads_need_padding = True
            logger.warning(
                f"MLA num attention heads is less than 64, force to use 64 num heads. "
                f"current num_heads={self.num_heads}, tp_size={fd_config.parallel_config.tensor_parallel_size}"
            )
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling and "factor" in self.rope_scaling:
            # if fd_config.model_config.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.rope_scaling["factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.useless_tensor = paddle.randn([1]).cast("int32")
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        self.prop = prop
        self.is_blackwell = cc >= 100

        if self.flash_attn_func is None:
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                logger.info("The current platform supports Flash Attention V3.")
                self.flash_attn_kwargs = {"softmax_scale": self.attn_softmax_scale}
            else:
                self.flash_attn_func = flash_attn_unpadded
                self.flash_attn_kwargs = {"scale": self.attn_softmax_scale, "training": False}
                logger.info(
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
        metadata.max_seqlen_k = max(metadata.max_kv_len_this_time.item(), metadata.max_enc_len_this_time.item())

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
        Prefill阶段的前向传播，支持 chunk prefill /prefix cache

        对于 MLA 模型的 chunk prefill /prefix cache 支持：
        1. 如果开启 chunk prefill /prefix cache
           - k 和 v 应该已经包含了 cached KV 和 new KV 的拼接
           - cu_seqlens_k 应该已经调整为包含 cached tokens
        2. 如果不存在 prefix cache，行为与之前相同
        """
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 写入新的 KV 到缓存 (只写入新 tokens，不写入 cached 部分)
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

        fmha_out = self.flash_attn_func(
            q,
            k,
            v,
            forward_meta.cu_seqlens_q,
            forward_meta.cu_seqlens_k,
            metadata.max_enc_len_this_time,
            metadata.max_seqlen_k,
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

        if self.pd_disaggregation_mode == "per_query":
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
        Mixed模式的前向传播，支持 chunk prefill /prefix cache

        对于 MLA 模型的 chunk prefill /prefix cache 支持：
        1. Prefill 分支：k 和 v 应该已包含 cached + new tokens
        2. Decode 分支：保持原有 latent attention 逻辑
        """
        metadata = self.attention_metadata
        speculate_decoder = self.speculative_method is not None
        speculate_max_tokens = self.speculate_max_draft_token_num

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # Prefill branch: k is not None
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

            if self.prop.major == 10:
                # TODO support FA4
                fmha_out = MLAAttentionBackend.mha_baseline(
                    q,
                    k,
                    v,
                    forward_meta.cu_seqlens_q,
                    forward_meta.cu_seqlens_k,
                    causal=self.causal,
                    **self.flash_attn_kwargs,
                )
                return fmha_out

            # FlashAttention for prefill
            fmha_out = self.flash_attn_func(
                q,
                k,
                v,
                forward_meta.cu_seqlens_q,
                forward_meta.cu_seqlens_k,
                metadata.max_enc_len_this_time,
                metadata.max_seqlen_k,
                causal=self.causal,
                **self.flash_attn_kwargs,
            )[0]

            return fmha_out

        # Decode branch: k is None
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

            if int(os.getenv("USE_FLASH_MLA", "0")) == 0 and self.prop.major == 9:
                assert self.num_heads <= 64, "paddle mla attention support failed"
                if self.heads_need_padding:
                    q = paddle.nn.functional.pad(
                        q, [0, (self.padding_num_heads) * (self.kv_lora_rank + self.qk_rope_head_dim)], value=0.0
                    ).contiguous()
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
                if self.heads_need_padding:
                    fmha_out = fmha_out[:, : self.num_heads * self.kv_lora_rank].contiguous()

                return fmha_out
            else:
                decoder_q = q
                decoder_q.reshape_([-1, 1, self.num_heads, 576])
                if self.heads_need_padding:
                    padded_q = paddle.zeros(
                        [decoder_q.shape[0], decoder_q.shape[1], 64, decoder_q.shape[3]], dtype=decoder_q.dtype
                    )
                    padded_q[:, :, : self.num_heads, :] = decoder_q
                    decoder_q = padded_q

                new_cache_shape = latent_cache.shape
                assert new_cache_shape[1] == 1
                new_cache_shape[1], new_cache_shape[2] = new_cache_shape[2], new_cache_shape[1]

                if self.prop.major == 10:
                    # blackwell
                    decoder_res = MLAAttentionBackend.mla_blackwell(
                        decoder_q,
                        latent_cache,
                        metadata.block_tables,
                        forward_meta.cache_seqlens,
                        attn_softmax_scale=self.attn_softmax_scale,
                    )
                else:

                    import flash_mla

                    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

                    decoder_res, _ = flash_mla.flash_mla_with_kvcache(
                        decoder_q,
                        # 外面的开源仓库的kv cache存储格式和FD的不同
                        # 幸好这里缓存的头是1，直接view即可，否则上上下下要改很多！
                        latent_cache.view(new_cache_shape),
                        metadata.block_tables,
                        forward_meta.cache_seqlens,
                        512,  # t.dv,
                        tile_scheduler_metadata,
                        num_splits,
                        softmax_scale=self.attn_softmax_scale,
                        causal=True,
                    )
                if self.heads_need_padding:
                    decoder_res = decoder_res[:, :, : self.num_heads, :].contiguous()

                return decoder_res

    @staticmethod
    def mla_blackwell(decoder_q, latent_cache, block_table, cache_seqlens, attn_softmax_scale):

        page_size = latent_cache.shape[2]
        q_num_heads = decoder_q.shape[2]
        assert decoder_q.shape[1:] == [1, q_num_heads, 576]
        assert latent_cache.shape[1:] == [1, page_size, 576]

        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        q_latent = decoder_q[:, :, :, :512]
        q_rope = decoder_q[:, :, :, 512:]

        q_latent = from_dlpack(q_latent.transpose([2, 3, 1, 0]), assumed_align=16).mark_layout_dynamic(leading_dim=1)

        q_rope = from_dlpack(q_rope.transpose([2, 3, 1, 0]), assumed_align=16).mark_layout_dynamic(leading_dim=1)

        c_latent = latent_cache[:, 0, :, :512]
        c_rope = latent_cache[:, 0, :, 512:]
        c_latent = from_dlpack(c_latent.transpose([1, 2, 0])).mark_layout_dynamic(leading_dim=1)
        c_rope = from_dlpack(c_rope.transpose([1, 2, 0])).mark_layout_dynamic(leading_dim=1)

        page_table = from_dlpack(block_table.transpose([1, 0]))

        paddle_output = paddle.zeros_like(decoder_q[:, :, :, :512]).contiguous()
        output = from_dlpack(paddle_output.transpose([2, 3, 1, 0])).mark_layout_dynamic(leading_dim=1)

        bsz = decoder_q.shape[0]
        q_len = decoder_q.shape[1]
        num_heads = decoder_q.shape[2]
        lse = paddle.empty([bsz, q_len, num_heads], dtype="float32")
        lse = from_dlpack(lse.transpose([2, 1, 0]))

        workspace = paddle.empty([1024, 1024, 1024])
        workspace = from_dlpack(workspace)

        split_kv = 1
        if split_kv == 1:
            workspace = None
        cache_seqs = from_dlpack(cache_seqlens)

        block_split_kvs = paddle.zeros([bsz], dtype="int32") + 1
        block_split_kvs = from_dlpack(block_split_kvs)
        softmax_scale = attn_softmax_scale
        output_scale = 1.0

        from mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16

        mla = BlackwellMultiHeadLatentAttentionForwardFP16(
            cutlass.Float32,
            cutlass.Float32,
            mma_qk_tiler_mn=(128, 128),
            mma_pv_tiler_mn=(128, 256),
            max_active_clusters=74,
            page_size=page_size,
            skip_correction_threshold=0.0,
            is_persistent=True,
            is_var_seq=True,
            is_var_split_kv=True,
        )
        import cuda.bindings.driver as cuda

        stream = cuda.CUstream(paddle.cuda.current_stream().stream_base.raw_stream)

        global compiled_mla

        if compiled_mla is None:
            compiled_mla = cute.compile(
                mla,
                q_latent,
                q_rope,
                c_latent,
                c_rope,
                page_table,
                output,
                lse,
                workspace,
                split_kv,
                cache_seqs,
                block_split_kvs,
                softmax_scale,
                output_scale,
                stream,
                options="--opt-level 2",
            )

        compiled_mla(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            output,
            lse,
            workspace,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )

        return paddle_output

    @staticmethod
    def flashmla_baseline(decoder_q, latent_cache, block_table, cache_seqlens, attn_softmax_scale):
        page_size = latent_cache.shape[2]
        q_num_heads = decoder_q.shape[2]
        assert decoder_q.shape[1:] == [1, q_num_heads, 576]
        assert latent_cache.shape[1:] == [1, page_size, 576]

        res_baseline = paddle.zeros([decoder_q.shape[0], 1, q_num_heads, 512])
        for batch_id in range(decoder_q.shape[0]):
            kv_len = cache_seqlens[batch_id].item()
            extract_k = paddle.zeros([kv_len, 576], dtype=decoder_q.dtype)
            extract_v = paddle.zeros([kv_len, 512], dtype=decoder_q.dtype)

            for local_seq_id in range(0, kv_len, page_size):
                start = local_seq_id
                end = min(local_seq_id + page_size, kv_len)
                physical_id = block_table[batch_id, local_seq_id // page_size].item()

                page_end = page_size if end % page_size == 0 else end % page_size
                extract_k[start:end, :] = latent_cache[physical_id, 0, :page_end, :]
                extract_v[start:end, :] = latent_cache[physical_id, 0, :page_end, :512]

            this_batch_q = decoder_q[batch_id, 0, :, :]
            p = paddle.matmul(this_batch_q, extract_k.transpose([1, 0]).contiguous())
            p = p * attn_softmax_scale
            p = paddle.nn.functional.softmax(p, -1)
            res_baseline[batch_id, 0, :, :] = paddle.matmul(p, extract_v).contiguous()
        return res_baseline

    @staticmethod
    def mha_baseline(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale):

        assert causal, "Only support causal attention for now"
        bsz = cu_seqlens_q.shape[0] - 1
        assert bsz + 1 == cu_seqlens_k.shape[0]

        q_token_num = q.shape[0]
        q_num_heads = q.shape[1]
        head_dim_q = q.shape[2]
        head_dim_v = v.shape[2]

        assert q_num_heads == k.shape[1]
        assert head_dim_q == k.shape[2]
        assert q_num_heads == v.shape[1]

        res_baseline = paddle.zeros([q_token_num, q_num_heads, head_dim_v], dtype=q.dtype)

        for batch_id in range(bsz):
            start = cu_seqlens_q[batch_id].item()
            end = cu_seqlens_q[batch_id + 1].item()
            this_q = q[start:end, :, :].contiguous()
            q_len = end - start

            start_k = cu_seqlens_k[batch_id].item()
            end_k = cu_seqlens_k[batch_id + 1].item()
            this_k = k[start_k:end_k, :, :].contiguous()
            this_v = v[start_k:end_k, :, :].contiguous()
            kv_len = end_k - start_k

            p = paddle.einsum("ilk, jlk->lij", this_q, this_k)
            p = p * softmax_scale

            import numpy as np

            tmp_zeros = np.zeros((q_len, kv_len)) - 1
            for i in range(q_len):
                tmp_zeros[i][: i + 1] = 0
            mask = tmp_zeros * 1000
            mask = paddle.to_tensor(mask, dtype=q.dtype)
            p = p + mask[None, :]
            p = paddle.nn.functional.softmax(p, -1)

            out = paddle.einsum("lij, jlk->ilk", p, this_v).reshape([q_len, q_num_heads, head_dim_v])

            res_baseline[start:end, :, :] = out

        return res_baseline
