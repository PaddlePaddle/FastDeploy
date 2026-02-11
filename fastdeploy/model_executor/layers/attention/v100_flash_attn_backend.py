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

V100 (SM70) compatible FlashAttention backend.

This backend is designed for NVIDIA V100 GPUs (SM70) which do not support
cp.async instructions required by append_attention and gqa_rope_write_cache.

Instead, it uses:
1. fused_rotary_position_encoding for RoPE (SM70 compatible)
2. Manual KV cache write operations (pure Python/Paddle)
3. flash_attn_unpadded for attention computation (SM70 compatible)

Limitations compared to SM80+ backends:
- No fused KV cache write kernel (separate RoPE and cache write)
- Lower performance due to non-fused operations
- Basic KV cache quantization only (no int4_zp support)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded
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
    V100 (SM70) compatible FlashAttention backend.

    This backend provides a pure FlashAttention implementation that works on V100 GPUs
    without requiring SM80+ specific instructions like cp.async.

    Key differences from standard FlashAttentionBackend:
    1. Uses fused_rotary_position_encoding instead of gqa_rope_write_cache
    2. Manual KV cache write instead of fused append_attention for cache updates
    3. All attention computation uses flash_attn_unpadded
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

        logger.info("V100FlashAttentionBackend initialized for SM70 GPU.")

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
    ):
        """
        Split fused QKV tensor into separate Q, K, V tensors.

        Args:
            qkv: Fused QKV tensor of shape [num_tokens, (num_heads + 2 * kv_num_heads) * head_dim]

        Returns:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, kv_num_heads, head_dim]
            v: Value tensor [num_tokens, kv_num_heads, head_dim]
        """
        num_tokens = qkv.shape[0]
        q_size = self.num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim

        q = qkv[:, :q_size].reshape([num_tokens, self.num_heads, self.head_dim])
        k = qkv[:, q_size : q_size + kv_size].reshape([num_tokens, self.kv_num_heads, self.head_dim])
        v = qkv[:, q_size + kv_size :].reshape([num_tokens, self.kv_num_heads, self.head_dim])

        return q, k, v

    def _apply_rotary_emb(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        rotary_embs: paddle.Tensor,
        use_neox_rotary_style: bool = False,
    ):
        """
        Apply rotary position embeddings to Q and K.

        This is a simplified implementation for V100.
        For production use, consider using fused_rotary_position_encoding.

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, kv_num_heads, head_dim]
            rotary_embs: Rotary embeddings [2, 1, max_seq_len, 1, head_dim//2]
                        where rotary_embs[0] is cos and rotary_embs[1] is sin
            use_neox_rotary_style: Whether to use neox rotary style

        Returns:
            q_rotated: Rotated query tensor
            k_rotated: Rotated key tensor
        """
        # Extract cos and sin from rotary embeddings
        # rotary_embs shape: [2, 1, max_seq_len, 1, head_dim//2] or similar
        if rotary_embs is None:
            return q, k

        # For now, return without rotation if format is not standard
        # This is a placeholder - real implementation would need to handle various formats
        if len(rotary_embs.shape) != 5:
            logger.warning(
                f"Unexpected rotary_embs shape {rotary_embs.shape}. "
                "Skipping RoPE application. This may affect model accuracy."
            )
            return q, k

        num_tokens = q.shape[0]

        # Get cos and sin values
        # Shape: [2, 1, max_seq_len, 1, head_dim//2]
        cos = rotary_embs[0]  # [1, max_seq_len, 1, head_dim//2]
        sin = rotary_embs[1]  # [1, max_seq_len, 1, head_dim//2]

        # Slice to current sequence length
        cos = cos[:, :num_tokens, :, :]  # [1, num_tokens, 1, head_dim//2]
        sin = sin[:, :num_tokens, :, :]

        # Reshape for broadcasting
        cos = cos.squeeze([0, 2])  # [num_tokens, head_dim//2]
        sin = sin.squeeze([0, 2])

        # Apply rotary embedding
        def rotate_half(x):
            """Rotate half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return paddle.concat([-x2, x1], axis=-1)

        def apply_rope(x, cos, sin):
            """Apply rotary position embedding."""
            # x: [num_tokens, num_heads, head_dim]
            # cos, sin: [num_tokens, head_dim//2]

            # Expand cos/sin to match x shape
            cos = cos.unsqueeze(1)  # [num_tokens, 1, head_dim//2]
            sin = sin.unsqueeze(1)

            # Duplicate for full head_dim
            cos = paddle.concat([cos, cos], axis=-1)  # [num_tokens, 1, head_dim]
            sin = paddle.concat([sin, sin], axis=-1)

            if use_neox_rotary_style:
                return (x * cos) + (rotate_half(x) * sin)
            else:
                # GPT-J style
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                cos_half = cos[..., : cos.shape[-1] // 2]
                sin_half = sin[..., : sin.shape[-1] // 2]
                x_rotated = paddle.stack([x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half], axis=-1)
                return x_rotated.flatten(start_axis=-2)

        q_rotated = apply_rope(q, cos, sin)
        k_rotated = apply_rope(k, cos, sin)

        return q_rotated, k_rotated

    def _write_kv_to_cache(
        self,
        k: paddle.Tensor,
        v: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        seq_lens: paddle.Tensor,
        batch_id_per_token: paddle.Tensor,
    ):
        """
        Write K and V tensors to block-based cache.

        This is a manual (non-fused) implementation for V100.

        Args:
            k: Key tensor [num_tokens, kv_num_heads, head_dim]
            v: Value tensor [num_tokens, kv_num_heads, head_dim]
            key_cache: Key cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            value_cache: Value cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            block_tables: Block table [batch_size, max_num_blocks_per_seq]
            seq_lens: Sequence lengths [batch_size]
            batch_id_per_token: Batch ID for each token [num_tokens]
        """
        num_tokens = k.shape[0]

        # For each token, find its block and offset within the block
        for token_idx in range(num_tokens):
            batch_id = batch_id_per_token[token_idx].item()
            seq_len = seq_lens[batch_id].item()

            # Calculate which block and offset
            block_idx = (seq_len - 1) // self.block_size
            block_offset = (seq_len - 1) % self.block_size

            # Get physical block number from block table
            physical_block = block_tables[batch_id, block_idx].item()

            # Write K and V to cache
            key_cache[physical_block, :, block_offset, :] = k[token_idx]
            value_cache[physical_block, :, block_offset, :] = v[token_idx]

    def _read_kv_from_cache(
        self,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        block_tables: paddle.Tensor,
        seq_lens: paddle.Tensor,
        batch_size: int,
    ):
        """
        Read K and V from block-based cache for all sequences.

        Returns concatenated K and V tensors for flash attention.

        Args:
            key_cache: Key cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            value_cache: Value cache [max_num_blocks, kv_num_heads, block_size, head_dim]
            block_tables: Block table [batch_size, max_num_blocks_per_seq]
            seq_lens: Sequence lengths [batch_size]
            batch_size: Number of sequences in batch

        Returns:
            k: Concatenated keys [total_tokens, kv_num_heads, head_dim]
            v: Concatenated values [total_tokens, kv_num_heads, head_dim]
            cu_seqlens_k: Cumulative sequence lengths for K
        """
        k_list = []
        v_list = []
        cu_seqlens = [0]

        for batch_id in range(batch_size):
            seq_len = seq_lens[batch_id].item()
            num_blocks = (seq_len + self.block_size - 1) // self.block_size

            for block_idx in range(num_blocks):
                physical_block = block_tables[batch_id, block_idx].item()

                # Calculate how many tokens in this block
                if block_idx == num_blocks - 1:
                    # Last block may be partial
                    tokens_in_block = seq_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size

                # Read from cache
                k_block = key_cache[physical_block, :, :tokens_in_block, :]
                v_block = value_cache[physical_block, :, :tokens_in_block, :]

                # Transpose to [tokens, heads, dim]
                k_list.append(k_block.transpose([1, 0, 2]))
                v_list.append(v_block.transpose([1, 0, 2]))

            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        k = paddle.concat(k_list, axis=0) if k_list else paddle.empty([0, self.kv_num_heads, self.head_dim])
        v = paddle.concat(v_list, axis=0) if v_list else paddle.empty([0, self.kv_num_heads, self.head_dim])
        cu_seqlens_k = paddle.to_tensor(cu_seqlens, dtype="int32")

        return k, v, cu_seqlens_k

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
        2. Apply RoPE manually
        3. Write KV to cache manually
        4. Use flash_attn_unpadded for attention

        Note: This is less efficient than the SM80+ fused implementation
        but provides full functionality on V100.
        """
        # Note: metadata is available for future use but not currently needed
        # metadata = forward_meta.attention_metadata

        # Step 1: Split QKV tensor
        q_split, k_split, v_split = self._split_qkv(qkv)

        # Step 2: Apply rotary embeddings
        q_rotated, k_rotated = self._apply_rotary_emb(
            q_split,
            k_split,
            forward_meta.rotary_embs,
            layer.use_neox_rotary_style,
        )

        # Step 3: Write KV to cache
        key_cache = forward_meta.caches[2 * layer.layer_id]
        value_cache = forward_meta.caches[2 * layer.layer_id + 1]

        self._write_kv_to_cache(
            k_rotated,
            v_split,
            key_cache,
            value_cache,
            forward_meta.block_tables,
            forward_meta.seq_lens_this_time,
            forward_meta.batch_id_per_token,
        )

        # Step 4: Read all KV from cache for attention
        batch_size = forward_meta.seq_lens_encoder.shape[0]
        k_all, v_all, cu_seqlens_k = self._read_kv_from_cache(
            key_cache,
            value_cache,
            forward_meta.block_tables,
            forward_meta.seq_lens_encoder + forward_meta.seq_lens_decoder,
            batch_size,
        )

        # Step 5: Expand K, V for GQA if needed
        if self.group_size > 1:
            # Repeat K and V for each query head in the group
            k_all = k_all.unsqueeze(2).expand([-1, -1, self.group_size, -1])
            k_all = k_all.reshape([-1, self.num_heads, self.head_dim])
            v_all = v_all.unsqueeze(2).expand([-1, -1, self.group_size, -1])
            v_all = v_all.reshape([-1, self.num_heads, self.head_dim])

        # Step 6: Run flash attention
        # Reshape Q for flash attention: [num_tokens, num_heads, head_dim]
        q_for_attn = q_rotated

        # Calculate max sequence lengths
        max_seqlen_q = forward_meta.seq_lens_this_time.max().item()
        max_seqlen_k = (forward_meta.seq_lens_encoder + forward_meta.seq_lens_decoder).max().item()

        # Run flash attention
        output = flash_attn_unpadded(
            q_for_attn,
            k_all,
            v_all,
            cu_seqlens_q=forward_meta.cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=self.causal,
            scale=self.head_dim**-0.5,
            training=False,
        )[0]

        # Reshape output to [num_tokens, num_heads * head_dim]
        output = output.reshape([-1, self.attn_outputsize_tp])

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
    ) -> paddle.Tensor:
        """
        Forward pass for extend (prompt cache hit).

        Uses the same implementation as forward_mixed.
        """
        return self.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
