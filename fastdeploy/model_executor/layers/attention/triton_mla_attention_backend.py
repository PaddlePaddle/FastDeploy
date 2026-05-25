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

# Triton-based MLA Attention Backend for FastDeploy.
# Uses triton kernels for KV cache write and decode attention,
# and flash_attn_unpadded for extend (prefill) attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

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
)
from fastdeploy.model_executor.layers.attention.triton_ops.decode_attention import (
    compute_num_kv_splits,
    decode_attention_fwd,
)
from fastdeploy.model_executor.layers.attention.triton_ops.mla_cache_kernel import (
    mla_write_cache_triton,
)
from fastdeploy.model_executor.layers.attention.triton_ops.unified_extend_attention import (
    build_kv_indices_from_block_tables,
)

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class TritonMLAAttentionMetadata(AttentionMetadata):
    _dtype: str = "bfloat16"
    block_tables: Optional[paddle.Tensor] = None
    max_enc_len_this_time: Optional[paddle.Tensor] = None
    max_dec_len_this_time: Optional[paddle.Tensor] = None
    max_kv_len_this_time: Optional[paddle.Tensor] = None
    # Pre-computed decode indices (CUDAGraph compatible)
    kv_indptr: Optional[paddle.Tensor] = None
    kv_indices: Optional[paddle.Tensor] = None
    num_kv_splits: Optional[paddle.Tensor] = None
    decode_bs: int = 0


class TritonMLAAttentionBackend(AttentionBackend):
    """
    Triton-based MLA Attention Backend.
    Uses triton kernels for KV cache write and decode attention.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: TritonMLAAttentionMetadata
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
        super().__init__()
        self.attention_metadata: TritonMLAAttentionMetadata = None

        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.causal: bool = getattr(fd_config.model_config, "causal", True)

        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling and "factor" in self.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)
            scaling_factor = fd_config.model_config.rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.max_kv_splits: int = 32

        self.rank, self.device_id = init_rank_and_device_id(fd_config)
        self.useless_tensor = paddle.zeros([1], dtype="int32")

        # Pre-allocate buffers for CUDAGraph compatibility (stable memory addresses)
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs
        max_blocks_per_seq = fd_config.cache_config.max_block_num_per_seq
        self._kv_indptr_buf = paddle.zeros([self.max_num_seqs + 1], dtype="int32")
        self._kv_indices_buf = paddle.zeros([self.max_num_seqs * max_blocks_per_seq * self.block_size], dtype="int32")
        self._num_kv_splits_buf = paddle.ones([self.max_num_seqs], dtype="int32")

        # Pre-allocate decode kernel intermediate buffers for CUDAGraph address stability
        Lv = fd_config.model_config.kv_lora_rank
        self._attn_logits_buf = paddle.empty([self.max_num_seqs, num_heads, self.max_kv_splits, Lv], dtype="float32")
        self._attn_lse_buf = paddle.empty([self.max_num_seqs, num_heads, self.max_kv_splits], dtype="float32")
        self._o_buf = paddle.empty([self.max_num_seqs, num_heads, Lv], dtype=paddle.get_default_dtype())

        if self.flash_attn_func is None:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                logger.info("TritonMLAAttentionBackend: Using Flash Attention V3.")
                self.flash_attn_kwargs = {"softmax_scale": self.attn_softmax_scale}
            else:
                self.flash_attn_func = flash_attn_unpadded
                logger.info("TritonMLAAttentionBackend: Using Flash Attention V2.")
                self.flash_attn_kwargs = {"scale": self.attn_softmax_scale, "training": False}

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = TritonMLAAttentionMetadata()
        metadata._dtype = paddle.get_default_dtype()
        metadata.block_tables = forward_meta.block_tables

        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            self.useless_tensor,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            self.useless_tensor,
            self.useless_tensor,
            self.useless_tensor,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            -1,
            -1,
            -1,
            self.block_size,
        )
        metadata.max_enc_len_this_time = forward_meta.max_len_tensor_cpu[1]
        metadata.max_dec_len_this_time = forward_meta.max_len_tensor_cpu[2]
        metadata.max_kv_len_this_time = forward_meta.max_len_tensor_cpu[5]

        # Pre-compute decode kv_indptr/kv_indices into stable pre-allocated buffers.
        # CUDAGraph requires tensors at the same memory address between capture and replay.
        seq_lens_decoder = forward_meta.seq_lens_decoder
        seq_lens_this_time = forward_meta.seq_lens_this_time
        decode_mask = seq_lens_decoder > 0
        decode_bs = int(decode_mask.sum().item())
        metadata.decode_bs = decode_bs

        if decode_bs > 0:
            decode_seq_lens = (seq_lens_decoder + seq_lens_this_time)[decode_mask]
            decode_block_tables = forward_meta.block_tables[decode_mask]
            total_kv_len = int(paddle.sum(decode_seq_lens).item())

            build_kv_indices_from_block_tables(
                decode_block_tables,
                decode_seq_lens,
                self.block_size,
                decode_bs,
                total_kv_len=total_kv_len,
                kv_indptr_buf=self._kv_indptr_buf,
                kv_indices_buf=self._kv_indices_buf,
            )
            # Fill padded entries in kv_indptr so out-of-range batches see 0 KV length.
            # kv_indptr[decode_bs] = total_kv_len; positions beyond must equal the same
            # so that (kv_indptr[i+1] - kv_indptr[i]) = 0 for padded batches.
            if decode_bs < self.max_num_seqs:
                self._kv_indptr_buf[decode_bs + 1 :] = total_kv_len

            # Compute num_kv_splits into the pre-allocated buffer
            compute_num_kv_splits(decode_seq_lens, decode_bs, self.max_kv_splits, out_buf=self._num_kv_splits_buf)
            # Padded entries must be >= 1 to avoid division by zero in kernel
            if decode_bs < self.max_num_seqs:
                self._num_kv_splits_buf[decode_bs:] = 1
        else:
            # No decode sequences: fill buffers with safe defaults
            self._kv_indptr_buf[:] = 0
            self._num_kv_splits_buf[:] = 1

        # Always use the full pre-allocated buffers (stable memory for CUDAGraph)
        metadata.kv_indptr = self._kv_indptr_buf
        metadata.kv_indices = self._kv_indices_buf
        metadata.num_kv_splits = self._num_kv_splits_buf

        self.attention_metadata = metadata

    def get_attention_meta(self) -> AttentionMetadata:
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[int, int, int, int]:
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
        metadata = self.attention_metadata
        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        if latent_cache is not None and forward_meta.slot_mapping is not None:
            mla_write_cache_triton(compressed_kv, k_pe, latent_cache, forward_meta.slot_mapping)

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
        metadata = self.attention_metadata
        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        if latent_cache is not None and forward_meta.slot_mapping is not None:
            mla_write_cache_triton(compressed_kv, k_pe, latent_cache, forward_meta.slot_mapping)

        return self._run_decode_kernel(q, latent_cache, metadata)

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
        if k is not None:
            return self.forward_extend(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

        metadata = self.attention_metadata
        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        if latent_cache is not None and forward_meta.slot_mapping is not None:
            mla_write_cache_triton(compressed_kv, k_pe, latent_cache, forward_meta.slot_mapping)

        # CUDAGraph path: q contains exactly the captured batch size of decode tokens.
        # Must always take this path during CUDAGraph capture/replay to keep the
        # execution trace identical (same kernel launches, same tensor shapes).
        if forward_meta.step_use_cudagraph or q.shape[0] == metadata.decode_bs:
            return self._run_decode_kernel(q, latent_cache, metadata)

        # Mixed batch (no CUDAGraph): q has all tokens (extend + decode).
        # Extract decode tokens (1 per decode sequence), run kernel, scatter back.
        decode_bs = metadata.decode_bs
        if decode_bs == 0:
            Lv = self.kv_lora_rank
            return paddle.zeros([q.shape[0], self.num_heads * Lv], dtype=q.dtype)

        total_tokens = q.shape[0]
        Lv = self.kv_lora_rank

        # Decode tokens are at positions cu_seqlens_q[i] for sequences with seq_lens_decoder > 0
        cu_seqlens = forward_meta.cu_seqlens_q
        seq_lens_decoder = forward_meta.seq_lens_decoder
        decode_mask = seq_lens_decoder > 0
        max_num_seqs = seq_lens_decoder.shape[0]
        seq_indices = paddle.arange(max_num_seqs, dtype="int32")
        decode_seq_indices = seq_indices[decode_mask]
        decode_token_positions = cu_seqlens[decode_seq_indices]

        q_decode = q[decode_token_positions]
        decode_out = self._run_decode_kernel(q_decode, latent_cache, metadata)

        output = paddle.zeros([total_tokens, self.num_heads * Lv], dtype=q.dtype)
        output[decode_token_positions] = decode_out
        return output

    def _run_decode_kernel(
        self,
        q: paddle.Tensor,
        latent_cache: paddle.Tensor,
        metadata: TritonMLAAttentionMetadata,
    ) -> paddle.Tensor:
        """Run triton decode attention kernel. q must have shape [bs, num_heads * latent_dim]."""
        bs = q.shape[0]
        Lv = self.kv_lora_rank
        latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        q_reshaped = q.reshape([bs, self.num_heads, latent_dim])

        # Use pre-allocated buffers sliced to current batch size for CUDAGraph address stability
        attn_logits = self._attn_logits_buf[:bs]
        attn_lse = self._attn_lse_buf[:bs]
        o = self._o_buf[:bs]

        decode_attention_fwd(
            q_reshaped,
            latent_cache,
            latent_cache[:, :, :, : self.kv_lora_rank],
            o,
            metadata.kv_indptr,
            metadata.kv_indices,
            attn_logits,
            attn_lse,
            metadata.num_kv_splits,
            self.max_kv_splits,
            self.attn_softmax_scale,
            self.block_size,
        )

        return o.reshape([-1, self.num_heads * Lv])
