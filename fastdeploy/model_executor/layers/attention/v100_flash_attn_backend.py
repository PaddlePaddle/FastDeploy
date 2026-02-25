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

V100 (SM70) compatible Attention backend.

This backend is designed for NVIDIA V100 GPUs (SM70) which do not support:
1. cp.async instructions required by append_attention and gqa_rope_write_cache
2. flash_attn_unpadded which requires SM80+ (check: is_sm8x || is_sm90_or_larger)

It uses Triton kernels (SM70 compatible) for:
1. Position computation (v100_compute_positions)
2. KV cache write (v100_write_kv_cache)
3. Decode attention via 2-stage flash-decoding (v100_decode_attention)
4. Prefill attention via tiled flash attention (v100_extend_attention)

RoPE is applied using Paddle native vectorized ops for better performance at small token counts.

Falls back to pure Python/Paddle implementations when Triton is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle.nn.functional import scaled_dot_product_attention
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

# Try importing Triton kernels
try:
    from fastdeploy.model_executor.ops.triton_ops.v100_attn_kernels import (
        v100_compute_positions,
        v100_paged_attention,
        v100_write_kv_cache,
    )

    _TRITON_KERNELS_AVAILABLE = True
except Exception:
    _TRITON_KERNELS_AVAILABLE = False


@dataclass
class V100FlashAttentionMetadata(AttentionMetadata):
    """
    Metadata for V100 FlashAttention backend.
    Simplified compared to FlashAttentionMetadata since we don't use SM80+ features.
    """

    cu_seqlens_k: paddle.Tensor = None
    _fuse_kernel_compute_dtype: str = "fp16"  # V100 prefers FP16 over BF16
    _dtype: paddle.dtype = paddle.float16

    # Cached tensors for decode phase
    max_len_tensor_cpu_decoder: paddle.Tensor = None


class V100FlashAttentionBackend(AttentionBackend):
    """
    V100 (SM70) compatible attention backend.

    Uses Triton kernels for GPU-side position computation, fused RoPE,
    KV cache writes, and paged attention (decode: 2-stage flash-decoding,
    prefill: tiled flash attention). Falls back to Python implementations
    when Triton is not available.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: V100FlashAttentionMetadata

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
        Initialize V100FlashAttentionBackend.
        """
        super().__init__()
        self.max_seq_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.attn_outputsize_tp = self.num_heads * self.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False) or getattr(
            fd_config.model_config, "use_3d_rope", False
        )

        # V100 specific: prefer FP16 over BF16
        self._use_fp16 = True

        self._use_triton = _TRITON_KERNELS_AVAILABLE
        if self._use_triton:
            logger.info("V100FlashAttentionBackend initialized for SM70 GPU (using Triton kernels).")
        else:
            logger.info(
                "V100FlashAttentionBackend initialized for SM70 GPU "
                "(Triton kernels unavailable, using Python fallback)."
            )

    def get_attention_meta(self):
        """Get attention metadata."""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate KV cache shape.
        V100 uses the same block-based cache format as other backends.
        """
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        # Note: int4_zp quantization is not well supported on V100
        if kv_cache_quant_type is not None and kv_cache_quant_type == "int4_zp":
            logger.warning("int4_zp KV cache quantization is not recommended on V100. Using full precision.")
        value_cache_shape = key_cache_shape
        return key_cache_shape, value_cache_shape

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata for a forward pass."""
        metadata = V100FlashAttentionMetadata()

        # Set dtype based on default dtype, prefer FP16 for V100
        default_dtype = paddle.get_default_dtype()
        if default_dtype == "bfloat16":
            # V100 has limited BF16 support, warn user
            logger.warning(
                "BF16 dtype detected but V100 has limited BF16 support. " "Consider using FP16 for better performance."
            )
            metadata._dtype = paddle.bfloat16
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif default_dtype == "float16":
            metadata._dtype = paddle.float16
            metadata._fuse_kernel_compute_dtype = "fp16"
        else:
            metadata._dtype = paddle.float32
            metadata._fuse_kernel_compute_dtype = "fp32"

        forward_meta.attention_metadata = metadata

    def _split_qkv(
        self,
        qkv: paddle.Tensor,
        layer: Attention,
    ):
        """
        Split fused QKV tensor into separate Q, K, V tensors.

        Args:
            qkv: Fused QKV tensor of shape [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]
            layer: Attention layer containing num_heads, kv_num_heads, head_dim info

        Returns:
            q: Query tensor [num_tokens, num_heads * head_dim]
            k: Key tensor [num_tokens, kv_num_heads * head_dim]
            v: Value tensor [num_tokens, kv_num_heads * head_dim]
        """
        q_size = layer.num_heads * layer.qk_head_dim
        kv_size = layer.kv_num_heads * layer.qk_head_dim

        q = qkv[:, :q_size]
        k = qkv[:, q_size : q_size + kv_size]
        v = qkv[:, q_size + kv_size :]

        return q, k, v

    # ------------------------------------------------------------------
    # Python fallback implementations (kept as _python_* methods)
    # ------------------------------------------------------------------

    def _python_compute_positions(
        self,
        batch_id_per_token,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        num_tokens,
    ):
        """Python fallback: compute per-token positions with a for-loop."""
        positions = []
        batch_token_counts = {}

        for token_idx in range(num_tokens):
            batch_id = int(batch_id_per_token[token_idx].item())
            if batch_id not in batch_token_counts:
                batch_token_counts[batch_id] = 0

            encoder_len = int(seq_lens_encoder[batch_id].item())
            decoder_len = int(seq_lens_decoder[batch_id].item())
            this_time_len = int(seq_lens_this_time[batch_id].item()) if seq_lens_this_time is not None else 0

            is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)

            if is_prefill:
                pos = batch_token_counts[batch_id]
            else:
                pos = encoder_len + decoder_len + batch_token_counts[batch_id]

            positions.append(pos)
            batch_token_counts[batch_id] += 1

        return paddle.to_tensor(positions, dtype="int64")

    def _python_apply_rope_to_qk(
        self,
        q,
        k,
        rotary_embs,
        positions,
        use_neox_rotary_style,
    ):
        """Python fallback: apply RoPE to Q and K using Paddle vectorized ops."""
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        kv_num_heads = k.shape[1]
        head_dim = q.shape[2]
        original_dtype = q.dtype

        cos_all = rotary_embs[0, 0, positions, 0, :]
        sin_all = rotary_embs[1, 0, positions, 0, :]
        cos_expanded = cos_all.unsqueeze(1)
        sin_expanded = sin_all.unsqueeze(1)

        if use_neox_rotary_style:
            rotary_dim = cos_all.shape[-1]
            half_dim = head_dim // 2

            q1 = q[:, :, :half_dim]
            q2 = q[:, :, half_dim:]
            k1 = k[:, :, :half_dim]
            k2 = k[:, :, half_dim:]

            if rotary_dim == head_dim:
                cos_half = cos_expanded[:, :, :half_dim]
                sin_half = sin_expanded[:, :, :half_dim]
            else:
                cos_half = cos_expanded
                sin_half = sin_expanded

            q1_new = q1 * cos_half - q2 * sin_half
            q2_new = q2 * cos_half + q1 * sin_half
            k1_new = k1 * cos_half - k2 * sin_half
            k2_new = k2 * cos_half + k1 * sin_half

            q_out = paddle.concat([q1_new, q2_new], axis=-1)
            k_out = paddle.concat([k1_new, k2_new], axis=-1)
        else:
            q_even = q[:, :, 0::2]
            q_odd = q[:, :, 1::2]
            k_even = k[:, :, 0::2]
            k_odd = k[:, :, 1::2]

            q_even_new = q_even * cos_expanded - q_odd * sin_expanded
            q_odd_new = q_odd * cos_expanded + q_even * sin_expanded
            k_even_new = k_even * cos_expanded - k_odd * sin_expanded
            k_odd_new = k_odd * cos_expanded + k_even * sin_expanded

            q_out = paddle.stack([q_even_new, q_odd_new], axis=-1).reshape([num_tokens, num_heads, head_dim])
            k_out = paddle.stack([k_even_new, k_odd_new], axis=-1).reshape([num_tokens, kv_num_heads, head_dim])

        return q_out.cast(original_dtype), k_out.cast(original_dtype)

    def _python_write_kv_to_block_cache(
        self,
        k,
        v,
        key_cache,
        value_cache,
        block_tables,
        positions,
        batch_id_per_token,
        kv_num_heads,
        head_dim,
    ):
        """Python fallback: write K/V to block cache with a for-loop."""
        num_tokens = k.shape[0]
        k_reshaped = k.reshape([num_tokens, kv_num_heads, head_dim])
        v_reshaped = v.reshape([num_tokens, kv_num_heads, head_dim])

        for token_idx in range(num_tokens):
            pos = int(positions[token_idx].item())
            batch_id = int(batch_id_per_token[token_idx].item())

            block_idx = pos // self.block_size
            block_offset = pos % self.block_size
            physical_block = int(block_tables[batch_id, block_idx].item())

            key_cache[physical_block, :, block_offset, :] = k_reshaped[token_idx]
            value_cache[physical_block, :, block_offset, :] = v_reshaped[token_idx]

    def _python_compute_total_seq_lens(
        self,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_size,
    ):
        """Python fallback: compute total_seq_lens per batch."""
        total_seq_lens = paddle.zeros_like(seq_lens_this_time)
        for batch_id in range(batch_size):
            encoder_len = int(seq_lens_encoder[batch_id].item())
            decoder_len = int(seq_lens_decoder[batch_id].item())
            this_time_len = int(seq_lens_this_time[batch_id].item())

            is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)
            if is_prefill:
                total_seq_lens[batch_id] = encoder_len
            else:
                total_seq_lens[batch_id] = encoder_len + decoder_len + this_time_len
        return total_seq_lens

    def _python_read_kv_from_block_cache(
        self,
        key_cache,
        value_cache,
        block_tables,
        total_seq_lens,
        batch_size,
        kv_num_heads,
        head_dim,
    ):
        """Python fallback: read K/V from block cache."""
        k_list = []
        v_list = []
        seq_lens_list = []
        batch_ids = []

        for batch_id in range(batch_size):
            seq_len = int(total_seq_lens[batch_id].item())
            if seq_len == 0:
                continue

            seq_lens_list.append(seq_len)
            batch_ids.append(batch_id)
            num_blocks = (seq_len + self.block_size - 1) // self.block_size

            k_seq = []
            v_seq = []

            for block_idx in range(num_blocks):
                physical_block = int(block_tables[batch_id, block_idx].item())
                if block_idx == num_blocks - 1:
                    tokens_in_block = seq_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size

                k_block = key_cache[physical_block, :, :tokens_in_block, :]
                v_block = value_cache[physical_block, :, :tokens_in_block, :]
                k_seq.append(k_block.transpose([1, 0, 2]))
                v_seq.append(v_block.transpose([1, 0, 2]))

            k_list.append(paddle.concat(k_seq, axis=0))
            v_list.append(paddle.concat(v_seq, axis=0))

        return k_list, v_list, seq_lens_list, batch_ids

    def _python_scaled_dot_product_attention_per_seq(
        self,
        query,
        key,
        value,
        is_causal=False,
    ):
        """Python fallback: SDPA for a single sequence."""
        q_len = query.shape[0]
        kv_len = key.shape[0]
        head_dim = query.shape[2]

        q = query.transpose([1, 0, 2])
        k = key.transpose([1, 0, 2])
        v = value.transpose([1, 0, 2])

        original_dtype = q.dtype
        q_f32 = q.cast("float32")
        k_f32 = k.cast("float32")

        scale = head_dim**-0.5
        scores = paddle.matmul(q_f32, k_f32.transpose([0, 2, 1])) * scale

        if is_causal:
            if q_len == kv_len:
                mask = paddle.triu(paddle.full([q_len, kv_len], float("-inf"), dtype=scores.dtype), diagonal=1)
            else:
                mask = paddle.zeros([q_len, kv_len], dtype=scores.dtype)
                for i in range(q_len):
                    pos = kv_len - q_len + i
                    if pos + 1 < kv_len:
                        mask[i, pos + 1 :] = float("-inf")
            scores = scores + mask.unsqueeze(0)

        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        v_f32 = v.cast("float32")
        output = paddle.matmul(attn_weights, v_f32)
        output = output.transpose([1, 0, 2]).cast(original_dtype)
        return output

    def _python_attention_forward(
        self,
        q_reshaped,
        forward_meta,
        key_cache,
        value_cache,
        total_seq_lens,
        num_heads,
        kv_num_heads,
        qk_head_dim,
        v_head_dim,
    ):
        """Python fallback: per-sequence attention using KV read + SDPA."""
        batch_size = forward_meta.seq_lens_this_time.shape[0]

        k_list, v_list, seq_lens_list, batch_ids = self._python_read_kv_from_block_cache(
            key_cache,
            value_cache,
            forward_meta.block_tables,
            total_seq_lens,
            batch_size,
            kv_num_heads,
            qk_head_dim,
        )

        output_list = []
        token_start = 0

        for k_seq, v_seq, kv_len, batch_id in zip(k_list, v_list, seq_lens_list, batch_ids):
            q_len = int(forward_meta.seq_lens_this_time[batch_id].item())
            if q_len == 0:
                continue

            q_seq = q_reshaped[token_start : token_start + q_len]

            if self.group_size > 1:
                k_seq_expanded = (
                    k_seq.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([kv_len, num_heads, qk_head_dim])
                )
                v_seq_expanded = (
                    v_seq.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([kv_len, num_heads, qk_head_dim])
                )
            else:
                k_seq_expanded = k_seq
                v_seq_expanded = v_seq

            out_seq = self._python_scaled_dot_product_attention_per_seq(
                q_seq, k_seq_expanded, v_seq_expanded, is_causal=self.causal
            )

            output_list.append(out_seq)
            token_start += q_len

        if output_list:
            output = paddle.concat(output_list, axis=0)
            output = output.reshape([-1, num_heads * v_head_dim])
        else:
            output = paddle.empty([0, num_heads * v_head_dim], dtype=q_reshaped.dtype)

        return output

    # ------------------------------------------------------------------
    # Main forward path
    # ------------------------------------------------------------------

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
        Forward pass for mixed prefill and decode.

        When Triton kernels are available:
        1. Split QKV
        2. Compute positions on GPU (Kernel 1)
        3. Apply RoPE in-place via Triton (Kernel 2)
        4. Write KV to cache via Triton (Kernel 3)
        5. Compute total_seq_lens on GPU
        6. Run paged attention via Triton (Kernel 4/5)

        Falls back to pure Python/Paddle when Triton is unavailable.
        """
        # Step 1: Split QKV tensor
        if qkv is not None:
            q, k, v = self._split_qkv(qkv, layer)

        num_tokens = q.shape[0]
        num_heads = layer.num_heads
        kv_num_heads = layer.kv_num_heads
        qk_head_dim = layer.qk_head_dim
        v_head_dim = getattr(layer, "v_head_dim", qk_head_dim)

        # Check if this is a dummy/profile run
        is_dummy_run = getattr(forward_meta, "is_dummy_or_profile_run", False)

        if is_dummy_run:
            return self._simple_attention_forward(q, k, v, num_heads, kv_num_heads, qk_head_dim, v_head_dim)

        # Get RoPE style from layer
        use_neox_rotary_style = getattr(layer, "use_neox_rotary_style", False)

        # Reshape Q and K
        q_reshaped = q.reshape([num_tokens, num_heads, qk_head_dim])
        k_reshaped = k.reshape([num_tokens, kv_num_heads, qk_head_dim])

        # Get KV cache
        key_cache = forward_meta.caches[2 * layer.layer_id]
        value_cache = forward_meta.caches[2 * layer.layer_id + 1]

        batch_size = forward_meta.seq_lens_this_time.shape[0]

        if self._use_triton:
            return self._triton_forward(
                q_reshaped,
                k_reshaped,
                v,
                forward_meta,
                key_cache,
                value_cache,
                num_tokens,
                num_heads,
                kv_num_heads,
                qk_head_dim,
                v_head_dim,
                batch_size,
                use_neox_rotary_style,
            )
        else:
            return self._python_forward(
                q_reshaped,
                k_reshaped,
                v,
                forward_meta,
                key_cache,
                value_cache,
                num_tokens,
                num_heads,
                kv_num_heads,
                qk_head_dim,
                v_head_dim,
                batch_size,
                use_neox_rotary_style,
            )

    def _triton_forward(
        self,
        q_reshaped,
        k_reshaped,
        v,
        forward_meta,
        key_cache,
        value_cache,
        num_tokens,
        num_heads,
        kv_num_heads,
        qk_head_dim,
        v_head_dim,
        batch_size,
        use_neox_rotary_style,
    ):
        """Forward using Triton kernels — no Python for-loops."""
        # Step 2: Compute positions on GPU (Kernel 1)
        positions = v100_compute_positions(
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
        )

        # Step 3: Apply RoPE using Paddle native ops
        if forward_meta.rotary_embs is not None:
            q_reshaped, k_reshaped = self._python_apply_rope_to_qk(
                q_reshaped,
                k_reshaped,
                forward_meta.rotary_embs,
                positions,
                use_neox_rotary_style,
            )

        # Step 4: Write KV to cache (Kernel 3)
        v_reshaped = v.reshape([num_tokens, kv_num_heads, qk_head_dim])
        v100_write_kv_cache(
            k_reshaped,
            v_reshaped,
            key_cache,
            value_cache,
            forward_meta.block_tables,
            positions,
            forward_meta.batch_id_per_token,
        )

        # Step 5: Compute total_seq_lens on GPU (no Python loop)
        is_prefill = (forward_meta.seq_lens_this_time == forward_meta.seq_lens_encoder) & (
            forward_meta.seq_lens_decoder == 0
        )
        total_seq_lens = paddle.where(
            is_prefill,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_encoder + forward_meta.seq_lens_decoder + forward_meta.seq_lens_this_time,
        )

        # Step 6: Paged attention (Kernel 4 for decode, Kernel 5 for prefill)
        output = paddle.empty([num_tokens, num_heads, qk_head_dim], dtype=q_reshaped.dtype)
        v100_paged_attention(
            q_reshaped,
            key_cache,
            value_cache,
            output,
            forward_meta.block_tables,
            forward_meta.seq_lens_this_time,
            total_seq_lens,
            forward_meta.cu_seqlens_q,
            forward_meta.batch_id_per_token,
            num_heads,
            kv_num_heads,
            qk_head_dim,
            is_causal=self.causal,
        )

        return output.reshape([-1, num_heads * v_head_dim])

    def _python_forward(
        self,
        q_reshaped,
        k_reshaped,
        v,
        forward_meta,
        key_cache,
        value_cache,
        num_tokens,
        num_heads,
        kv_num_heads,
        qk_head_dim,
        v_head_dim,
        batch_size,
        use_neox_rotary_style,
    ):
        """Forward using pure Python/Paddle — original implementation."""
        # Step 2: Compute positions
        positions = self._python_compute_positions(
            forward_meta.batch_id_per_token,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            num_tokens,
        )

        # Step 3: Apply RoPE
        if forward_meta.rotary_embs is not None:
            q_reshaped, k_reshaped = self._python_apply_rope_to_qk(
                q_reshaped,
                k_reshaped,
                forward_meta.rotary_embs,
                positions,
                use_neox_rotary_style,
            )

        # Step 4: Write KV to cache
        k_flat = k_reshaped.reshape([num_tokens, kv_num_heads * qk_head_dim])
        self._python_write_kv_to_block_cache(
            k_flat,
            v,
            key_cache,
            value_cache,
            forward_meta.block_tables,
            positions,
            forward_meta.batch_id_per_token,
            kv_num_heads,
            qk_head_dim,
        )

        # Step 5: Compute total_seq_lens
        total_seq_lens = self._python_compute_total_seq_lens(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            batch_size,
        )

        # Step 6: Per-sequence attention
        return self._python_attention_forward(
            q_reshaped,
            forward_meta,
            key_cache,
            value_cache,
            total_seq_lens,
            num_heads,
            kv_num_heads,
            qk_head_dim,
            v_head_dim,
        )

    def _simple_attention_forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        num_heads: int,
        kv_num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
    ) -> paddle.Tensor:
        """
        Simple attention forward without KV cache.
        Used for dummy/profile runs where block_tables may not be properly sized.
        """
        num_tokens = q.shape[0]

        # Reshape tensors
        q_reshaped = q.reshape([num_tokens, num_heads, qk_head_dim])
        k_reshaped = k.reshape([num_tokens, kv_num_heads, qk_head_dim])
        v_reshaped = v.reshape([num_tokens, kv_num_heads, qk_head_dim])

        # Expand K and V for GQA if needed
        if self.group_size > 1:
            k_reshaped = (
                k_reshaped.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([num_tokens, num_heads, qk_head_dim])
            )
            v_reshaped = (
                v_reshaped.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([num_tokens, num_heads, qk_head_dim])
            )

        # Simple self-attention (treat all tokens as one sequence)
        # Transpose to [num_heads, num_tokens, head_dim]
        q_t = q_reshaped.transpose([1, 0, 2])
        k_t = k_reshaped.transpose([1, 0, 2])
        v_t = v_reshaped.transpose([1, 0, 2])

        # Add batch dimension
        q_t = q_t.unsqueeze(0)
        k_t = k_t.unsqueeze(0)
        v_t = v_t.unsqueeze(0)

        # Run attention
        output = scaled_dot_product_attention(q_t, k_t, v_t, is_causal=self.causal)

        # Reshape output
        output = output.squeeze(0).transpose([1, 0, 2])
        output = output.reshape([num_tokens, num_heads * v_head_dim])

        return output

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
        Forward pass for decode-only (single token per sequence).

        Uses the same implementation as forward_mixed since the Triton
        paged attention dispatcher handles decode vs prefill automatically.
        """
        return self.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

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
        save_kv_cache: bool = True,
    ) -> paddle.Tensor:
        """
        Forward pass for extend (prompt cache hit).

        Uses the same implementation as forward_mixed.
        """
        return self.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
