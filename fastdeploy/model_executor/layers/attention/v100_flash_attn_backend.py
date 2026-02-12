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

Instead, it uses:
1. Manual KV cache write operations (pure Python/Paddle)
2. paddle.nn.functional.scaled_dot_product_attention for attention computation (SM70 compatible)

Limitations compared to SM80+ backends:
- No fused KV cache write kernel (separate RoPE and cache write)
- Lower performance due to non-fused operations and per-sequence attention
- Basic KV cache quantization only (no int4_zp support)
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


def _apply_rope(qk: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor) -> paddle.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to query or key tensor.

    Args:
        qk: Query or Key tensor [seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, 1, head_dim] or [seq_len, 1, head_dim//2]
        sin: Sine values [seq_len, 1, head_dim] or [seq_len, 1, head_dim//2]

    Returns:
        Tensor with RoPE applied [seq_len, num_heads, head_dim]
    """
    # Interleaved rotation: rotate pairs of elements
    # rotate_half: [..., x1, x0, x3, x2, ...] -> [..., -x1, x0, -x3, x2, ...]
    rotate_half = paddle.reshape(
        paddle.stack([-qk[..., 1::2], qk[..., 0::2]], axis=-1),
        paddle.shape(qk),
    )
    out = paddle.add(paddle.multiply(qk, cos), paddle.multiply(rotate_half, sin))
    return paddle.cast(out, qk.dtype)


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

    This backend provides an attention implementation that works on V100 GPUs
    without requiring SM80+ specific instructions like cp.async or flash_attn_unpadded.

    Key differences from standard FlashAttentionBackend:
    1. Uses paddle.nn.functional.scaled_dot_product_attention instead of flash_attn_unpadded
    2. Manual KV cache write instead of fused append_attention for cache updates
    3. Per-sequence attention computation (less efficient but SM70 compatible)
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

        logger.info("V100FlashAttentionBackend initialized for SM70 GPU (using scaled_dot_product_attention).")

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

    def _apply_rope_to_qk(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        rotary_embs: paddle.Tensor,
        seq_lens_encoder: paddle.Tensor,
        seq_lens_decoder: paddle.Tensor,
        batch_id_per_token: paddle.Tensor,
        seq_lens_this_time: paddle.Tensor,
        use_neox_rotary_style: bool = False,
    ):
        """
        Apply RoPE to Q and K tensors using vectorized operations.

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, kv_num_heads, head_dim]
            rotary_embs: Rotary embeddings [2, 1, max_seq_len, 1, head_dim] for interleaved style
                         or [2, 1, max_seq_len, 1, head_dim//2] for neox style
            seq_lens_encoder: Encoder sequence lengths [batch_size]
            seq_lens_decoder: Decoder sequence lengths [batch_size]
            batch_id_per_token: Batch ID for each token [num_tokens]
            seq_lens_this_time: Tokens being processed this time [batch_size]
            use_neox_rotary_style: Whether to use neox style (half rotation) or interleaved style

        Returns:
            q_with_rope: Query with RoPE applied [num_tokens, num_heads, head_dim]
            k_with_rope: Key with RoPE applied [num_tokens, kv_num_heads, head_dim]
        """
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        kv_num_heads = k.shape[1]
        head_dim = q.shape[2]
        original_dtype = q.dtype
        forward_meta_seq_lens_this_time = seq_lens_this_time

        # Calculate positions for each token
        # The position for each token is determined by:
        # - seq_lens_encoder: total encoder tokens (for prefill, this includes current tokens)
        # - seq_lens_decoder: total decoder tokens generated so far (before this call)
        # - seq_lens_this_time: tokens being processed in this call
        #
        # Key insight:
        # - Prefill: seq_lens_this_time == seq_lens_encoder (processing all encoder tokens)
        #   => positions should be 0, 1, 2, ..., seq_lens_this_time-1
        # - Decode: seq_lens_this_time = 1 (processing one new token)
        #   => position should be seq_lens_encoder + seq_lens_decoder
        #
        # We can distinguish by checking if this_time_len == encoder_len (prefill) or not (decode)
        positions = []
        batch_token_counts = {}

        for token_idx in range(num_tokens):
            batch_id = int(batch_id_per_token[token_idx].item())
            if batch_id not in batch_token_counts:
                batch_token_counts[batch_id] = 0

            encoder_len = int(seq_lens_encoder[batch_id].item())
            decoder_len = int(seq_lens_decoder[batch_id].item())
            this_time_len = (
                int(forward_meta_seq_lens_this_time[batch_id].item())
                if forward_meta_seq_lens_this_time is not None
                else 0
            )

            # Determine if this is prefill or decode for this batch
            is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)

            if is_prefill:
                # Prefill: positions start from 0
                pos = batch_token_counts[batch_id]
            else:
                # Decode: positions start from encoder_len + decoder_len
                pos = encoder_len + decoder_len + batch_token_counts[batch_id]

            positions.append(pos)
            batch_token_counts[batch_id] += 1

        positions = paddle.to_tensor(positions, dtype="int64")

        # Get cos and sin for all positions at once
        # rotary_embs shape: [2, 1, max_seq_len, 1, rotary_dim]
        # where rotary_dim = head_dim for interleaved, head_dim//2 for neox
        cos_all = rotary_embs[0, 0, positions, 0, :]  # [num_tokens, rotary_dim]
        sin_all = rotary_embs[1, 0, positions, 0, :]  # [num_tokens, rotary_dim]

        # Expand for heads: [num_tokens, 1, rotary_dim]
        cos_expanded = cos_all.unsqueeze(1)  # [num_tokens, 1, rotary_dim]
        sin_expanded = sin_all.unsqueeze(1)  # [num_tokens, 1, rotary_dim]

        if use_neox_rotary_style:
            # Neox style (split half): split q/k into first half and second half
            # x = [x1, x2] where x1 = x[:, :, :half], x2 = x[:, :, half:]
            # rotate_half(x) = [-x2, x1]
            # output = x * cos + rotate_half(x) * sin
            #
            # For Qwen3: rotary_embs shape is [2, 1, max_seq_len, 1, head_dim]
            # cos/sin are already head_dim, need to use first half for rotation
            rotary_dim = cos_all.shape[-1]
            half_dim = head_dim // 2

            # Split Q and K into first half and second half
            q1 = q[:, :, :half_dim]  # [num_tokens, num_heads, head_dim//2]
            q2 = q[:, :, half_dim:]  # [num_tokens, num_heads, head_dim//2]
            k1 = k[:, :, :half_dim]  # [num_tokens, kv_num_heads, head_dim//2]
            k2 = k[:, :, half_dim:]  # [num_tokens, kv_num_heads, head_dim//2]

            # cos/sin from rotary_embs - may be head_dim or head_dim//2 depending on model
            # Slice to head_dim//2 for the rotation
            if rotary_dim == head_dim:
                # Full head_dim cos/sin, need to slice
                cos_half = cos_expanded[:, :, :half_dim]  # [num_tokens, 1, head_dim//2]
                sin_half = sin_expanded[:, :, :half_dim]  # [num_tokens, 1, head_dim//2]
            else:
                # Already head_dim//2
                cos_half = cos_expanded
                sin_half = sin_expanded

            # Apply rotation: [q1, q2] * cos + [-q2, q1] * sin
            # = [q1*cos - q2*sin, q2*cos + q1*sin]
            q1_new = q1 * cos_half - q2 * sin_half
            q2_new = q2 * cos_half + q1 * sin_half
            k1_new = k1 * cos_half - k2 * sin_half
            k2_new = k2 * cos_half + k1 * sin_half

            q_out = paddle.concat([q1_new, q2_new], axis=-1)
            k_out = paddle.concat([k1_new, k2_new], axis=-1)
        else:
            # Interleaved style (use_neox_rotary_style=False):
            # For each pair (x_even, x_odd), apply rotation:
            # x_even_new = x_even * cos - x_odd * sin
            # x_odd_new = x_odd * cos + x_even * sin
            #
            # rotary_embs already has shape [2, 1, max_seq_len, 1, head_dim//2]
            # so cos_expanded/sin_expanded are [num_tokens, 1, head_dim//2]
            # which matches q_even/q_odd shape [num_tokens, num_heads, head_dim//2]

            # Split Q and K into even and odd parts
            q_even = q[:, :, 0::2]  # [num_tokens, num_heads, head_dim//2]
            q_odd = q[:, :, 1::2]  # [num_tokens, num_heads, head_dim//2]
            k_even = k[:, :, 0::2]  # [num_tokens, kv_num_heads, head_dim//2]
            k_odd = k[:, :, 1::2]  # [num_tokens, kv_num_heads, head_dim//2]

            # Apply RoPE formula vectorized
            # cos_expanded/sin_expanded: [num_tokens, 1, head_dim//2] will broadcast
            q_even_new = q_even * cos_expanded - q_odd * sin_expanded
            q_odd_new = q_odd * cos_expanded + q_even * sin_expanded
            k_even_new = k_even * cos_expanded - k_odd * sin_expanded
            k_odd_new = k_odd * cos_expanded + k_even * sin_expanded

            # Interleave back: stack and reshape
            # [num_tokens, num_heads, head_dim//2, 2] -> [num_tokens, num_heads, head_dim]
            q_out = paddle.stack([q_even_new, q_odd_new], axis=-1).reshape([num_tokens, num_heads, head_dim])
            k_out = paddle.stack([k_even_new, k_odd_new], axis=-1).reshape([num_tokens, kv_num_heads, head_dim])

        return q_out.cast(original_dtype), k_out.cast(original_dtype)

    def _write_kv_to_block_cache(
        self,
        k: paddle.Tensor,
        v: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        seq_lens_encoder: paddle.Tensor,
        seq_lens_decoder: paddle.Tensor,
        seq_lens_this_time: paddle.Tensor,
        batch_id_per_token: paddle.Tensor,
        kv_num_heads: int,
        head_dim: int,
    ):
        """
        Write K and V tensors to block-based cache.

        This is a manual (non-fused) implementation for V100.

        Args:
            k: Key tensor [num_tokens, kv_num_heads * head_dim]
            v: Value tensor [num_tokens, kv_num_heads * head_dim]
            key_cache: Key cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            value_cache: Value cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            block_tables: Block table [batch_size, max_num_blocks_per_seq]
            seq_lens_encoder: Encoder sequence lengths [batch_size]
            seq_lens_decoder: Decoder sequence lengths [batch_size]
            seq_lens_this_time: Sequence lengths processed this time [batch_size]
            batch_id_per_token: Batch ID for each token [num_tokens]
            kv_num_heads: Number of KV heads
            head_dim: Head dimension
        """
        num_tokens = k.shape[0]

        # Reshape K and V to [num_tokens, kv_num_heads, head_dim]
        k_reshaped = k.reshape([num_tokens, kv_num_heads, head_dim])
        v_reshaped = v.reshape([num_tokens, kv_num_heads, head_dim])

        # Track position within each sequence
        batch_token_counts = {}

        for token_idx in range(num_tokens):
            batch_id = int(batch_id_per_token[token_idx].item())

            # Initialize or increment token count for this batch
            if batch_id not in batch_token_counts:
                batch_token_counts[batch_id] = 0

            # Calculate position in the full sequence
            encoder_len = int(seq_lens_encoder[batch_id].item())
            decoder_len = int(seq_lens_decoder[batch_id].item())
            this_time_len = int(seq_lens_this_time[batch_id].item())
            token_pos_in_batch = batch_token_counts[batch_id]

            # Determine if this is prefill or decode for this batch
            is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)

            if is_prefill:
                # Prefill: positions start from 0
                full_seq_pos = token_pos_in_batch
            else:
                # Decode: positions start from encoder_len + decoder_len
                full_seq_pos = encoder_len + decoder_len + token_pos_in_batch

            # Calculate block index and offset within block
            block_idx = full_seq_pos // self.block_size
            block_offset = full_seq_pos % self.block_size

            # Get physical block number from block table
            physical_block = int(block_tables[batch_id, block_idx].item())

            # Write K and V to cache
            # key_cache shape: [max_num_blocks, kv_num_heads, block_size, head_dim]
            key_cache[physical_block, :, block_offset, :] = k_reshaped[token_idx]
            value_cache[physical_block, :, block_offset, :] = v_reshaped[token_idx]

            batch_token_counts[batch_id] += 1

    def _read_kv_from_block_cache(
        self,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        total_seq_lens: paddle.Tensor,
        batch_size: int,
        kv_num_heads: int,
        head_dim: int,
    ):
        """
        Read K and V from block-based cache for all sequences.

        Args:
            key_cache: Key cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            value_cache: Value cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            block_tables: Block table [batch_size, max_num_blocks_per_seq]
            total_seq_lens: Total sequence lengths [batch_size]
            batch_size: Number of sequences in batch
            kv_num_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            k_list: List of key tensors per batch [seq_len, kv_num_heads, head_dim]
            v_list: List of value tensors per batch [seq_len, kv_num_heads, head_dim]
            seq_lens_list: List of sequence lengths
            batch_ids: List of original batch IDs (for sequences with seq_len > 0)
        """
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

                # Calculate how many tokens in this block
                if block_idx == num_blocks - 1:
                    # Last block may be partial
                    tokens_in_block = seq_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size

                # Read from cache: [kv_num_heads, tokens_in_block, head_dim]
                k_block = key_cache[physical_block, :, :tokens_in_block, :]
                v_block = value_cache[physical_block, :, :tokens_in_block, :]

                # Transpose to [tokens_in_block, kv_num_heads, head_dim]
                k_seq.append(k_block.transpose([1, 0, 2]))
                v_seq.append(v_block.transpose([1, 0, 2]))

            # Concatenate all blocks for this sequence
            k_list.append(paddle.concat(k_seq, axis=0))
            v_list.append(paddle.concat(v_seq, axis=0))

        return k_list, v_list, seq_lens_list, batch_ids

    def _scaled_dot_product_attention_per_seq(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        is_causal: bool = False,
    ) -> paddle.Tensor:
        """
        Run scaled dot-product attention for a single sequence.

        Uses manual implementation to avoid Paddle's GQA head count validation.

        Args:
            query: [q_len, num_heads, head_dim]
            key: [kv_len, num_heads, head_dim] (already expanded for GQA)
            value: [kv_len, num_heads, head_dim] (already expanded for GQA)
            is_causal: Whether to apply causal mask

        Returns:
            output: [q_len, num_heads, head_dim]
        """
        q_len = query.shape[0]
        kv_len = key.shape[0]
        head_dim = query.shape[2]

        # Transpose to [num_heads, seq_len, head_dim]
        q = query.transpose([1, 0, 2])  # [num_heads, q_len, head_dim]
        k = key.transpose([1, 0, 2])  # [num_heads, kv_len, head_dim]
        v = value.transpose([1, 0, 2])  # [num_heads, kv_len, head_dim]

        # Compute attention scores: [num_heads, q_len, kv_len]
        # Use float32 for numerical stability
        original_dtype = q.dtype
        q_f32 = q.cast("float32")
        k_f32 = k.cast("float32")

        scale = head_dim**-0.5
        scores = paddle.matmul(q_f32, k_f32.transpose([0, 2, 1])) * scale

        # Apply causal mask if needed
        if is_causal:
            # Create causal mask
            # For prefill: mask positions where query position < key position
            # For decode (q_len=1): mask future positions
            if q_len == kv_len:
                # Standard causal mask for prefill
                mask = paddle.triu(paddle.full([q_len, kv_len], float("-inf"), dtype=scores.dtype), diagonal=1)
            else:
                # For decode or partial prefill
                # Query at position i can attend to key positions 0 to (kv_len - q_len + i)
                mask = paddle.zeros([q_len, kv_len], dtype=scores.dtype)
                for i in range(q_len):
                    pos = kv_len - q_len + i
                    if pos + 1 < kv_len:
                        mask[i, pos + 1 :] = float("-inf")
            scores = scores + mask.unsqueeze(0)

        # Softmax and output
        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        v_f32 = v.cast("float32")
        output = paddle.matmul(attn_weights, v_f32)  # [num_heads, q_len, head_dim]

        # Transpose back to [q_len, num_heads, head_dim] and cast back
        output = output.transpose([1, 0, 2]).cast(original_dtype)

        return output

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

        This V100 implementation uses a simplified approach:
        1. Split QKV
        2. Apply RoPE to Q and K
        3. Write KV to cache (if not dummy run)
        4. Read KV from cache (or use current K/V for dummy run)
        5. Use scaled_dot_product_attention per sequence (SM70 compatible)

        Note: This is less efficient than the SM80+ fused implementation
        but provides full functionality on V100.
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
            # For dummy run, use simple attention without KV cache
            # This avoids block_tables index out of bounds issues
            return self._simple_attention_forward(q, k, v, num_heads, kv_num_heads, qk_head_dim, v_head_dim)

        # Step 2: Apply RoPE to Q and K
        # Reshape Q and K for RoPE application
        q_reshaped = q.reshape([num_tokens, num_heads, qk_head_dim])
        k_reshaped = k.reshape([num_tokens, kv_num_heads, qk_head_dim])

        # Get RoPE style from layer
        use_neox_rotary_style = getattr(layer, "use_neox_rotary_style", False)

        # Apply RoPE if rotary_embs is available
        if forward_meta.rotary_embs is not None:
            q_reshaped, k_reshaped = self._apply_rope_to_qk(
                q_reshaped,
                k_reshaped,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.seq_lens_this_time,
                use_neox_rotary_style,
            )

        # Reshape back for cache write
        k_with_rope = k_reshaped.reshape([num_tokens, kv_num_heads * qk_head_dim])
        v_flat = v  # V doesn't need RoPE

        # Get KV cache from forward_meta.caches
        key_cache = forward_meta.caches[2 * layer.layer_id]
        value_cache = forward_meta.caches[2 * layer.layer_id + 1]

        # Step 3: Write KV to cache (with RoPE already applied to K)
        self._write_kv_to_block_cache(
            k_with_rope,
            v_flat,
            key_cache,
            value_cache,
            forward_meta.block_tables,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
            kv_num_heads,
            qk_head_dim,
        )

        # Step 4: Read all KV from cache
        # Use seq_lens_this_time.shape[0] as batch_size to ensure consistency
        batch_size = forward_meta.seq_lens_this_time.shape[0]

        # Calculate total sequence lengths for KV cache reading
        # Key insight:
        # - Prefill: seq_lens_encoder already includes current tokens (this_time == encoder_len)
        #   => total = encoder_len (don't add this_time again)
        # - Decode: this_time is a new token not yet in encoder_len/decoder_len
        #   => total = encoder_len + decoder_len + this_time

        total_seq_lens = paddle.zeros_like(forward_meta.seq_lens_this_time)
        for batch_id in range(batch_size):
            encoder_len = int(forward_meta.seq_lens_encoder[batch_id].item())
            decoder_len = int(forward_meta.seq_lens_decoder[batch_id].item())
            this_time_len = int(forward_meta.seq_lens_this_time[batch_id].item())

            # Determine if this is prefill or decode
            is_prefill = (this_time_len == encoder_len) and (decoder_len == 0)

            if is_prefill:
                # Prefill: cache has encoder_len tokens
                total_seq_lens[batch_id] = encoder_len
            else:
                # Decode: cache has encoder_len + decoder_len + this_time_len tokens
                total_seq_lens[batch_id] = encoder_len + decoder_len + this_time_len

        k_list, v_list, seq_lens_list, batch_ids = self._read_kv_from_block_cache(
            key_cache,
            value_cache,
            forward_meta.block_tables,
            total_seq_lens,
            batch_size,
            kv_num_heads,
            qk_head_dim,
        )

        # Step 5: Use Q with RoPE applied (q_reshaped already has RoPE if available)
        # Note: q_reshaped was already set to [num_tokens, num_heads, qk_head_dim] with RoPE above

        # Step 6: Run attention per sequence
        output_list = []
        token_start = 0

        for k_seq, v_seq, kv_len, batch_id in zip(k_list, v_list, seq_lens_list, batch_ids):
            # Get Q for this sequence using original batch_id
            q_len = int(forward_meta.seq_lens_this_time[batch_id].item())
            if q_len == 0:
                continue

            q_seq = q_reshaped[token_start : token_start + q_len]

            # Expand K and V for GQA if needed
            if self.group_size > 1:
                # k_seq: [kv_len, kv_num_heads, head_dim] -> [kv_len, num_heads, head_dim]
                k_seq_expanded = (
                    k_seq.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([kv_len, num_heads, qk_head_dim])
                )
                v_seq_expanded = (
                    v_seq.unsqueeze(2).tile([1, 1, self.group_size, 1]).reshape([kv_len, num_heads, qk_head_dim])
                )
            else:
                k_seq_expanded = k_seq
                v_seq_expanded = v_seq

            # Run attention for this sequence
            out_seq = self._scaled_dot_product_attention_per_seq(
                q_seq, k_seq_expanded, v_seq_expanded, is_causal=self.causal
            )

            output_list.append(out_seq)
            token_start += q_len

        # Concatenate outputs
        if output_list:
            output = paddle.concat(output_list, axis=0)
            # Reshape to [num_tokens, num_heads * v_head_dim]
            output = output.reshape([-1, num_heads * v_head_dim])
        else:
            output = paddle.empty([0, num_heads * v_head_dim], dtype=q.dtype)

        return output

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

        Uses the same implementation as forward_mixed since V100
        doesn't have optimized decode kernels.
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
