# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 Black Forest Labs.
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
Flux DiT (Diffusion Transformer) for image generation.

Adapted from PPDiffusers FluxTransformer2DModel as a standalone PaddlePaddle
implementation — no ppdiffusers or torch dependencies.

Architecture:
  - N double-stream blocks (joint attention on image + text)
  - M single-stream blocks (concatenated image+text self-attention)
  - AdaLayerNorm conditioning from timestep + pooled text embeddings
  - RoPE positional encoding for spatial + text positions
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# ---------------------------------------------------------------------------
# 辅助模块 (Helper modules)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Layer):
    """Root Mean Square Layer Normalization.

    .. todo:: Phase 3 — unify with ``fastdeploy.model_executor.layers.normalization.RMSNorm``
       once the diffusion pipeline carries an ``FDConfig`` instance.  The FD-native
       RMSNorm requires ``FDConfig`` + fused CUDA kernels + batch-invariant dispatch
       which are not yet wired into the diffusion engine.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(shape=[dim], default_initializer=nn.initializer.Constant(1.0))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        rms = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TimestepEmbedding(nn.Layer):
    """Sinusoidal timestep embedding → MLP projection."""

    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim),
            nn.Silu(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: paddle.Tensor) -> paddle.Tensor:
        """Embed scalar timesteps to vector representations.

        Args:
            timestep: [B] tensor of timestep values.

        Returns:
            [B, dim] timestep embeddings.
        """
        half_dim = self.frequency_dim // 2
        freqs = paddle.exp(-math.log(10000.0) * paddle.arange(0, half_dim, dtype=paddle.float32) / half_dim)
        args = timestep.unsqueeze(-1).cast(paddle.float32) * freqs.unsqueeze(0)
        emb = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        return self.mlp(emb.cast(timestep.dtype))


class CombinedTimestepTextEmbedding(nn.Layer):
    """Combine timestep embedding with pooled text projection.

    Used in Flux-schnell (no guidance embedding).
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(embedding_dim)
        self.text_embed = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: paddle.Tensor, pooled_projection: paddle.Tensor) -> paddle.Tensor:
        time_emb = self.time_embed(timestep)
        pooled_emb = self.text_embed(pooled_projection)
        return time_emb + pooled_emb


class CombinedTimestepGuidanceTextEmbedding(nn.Layer):
    """Combine timestep + guidance scale + pooled text embeddings.

    Used in Flux-dev (with guidance embedding).
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(embedding_dim)
        self.guidance_embed = TimestepEmbedding(embedding_dim)
        self.text_embed = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(
        self,
        timestep: paddle.Tensor,
        guidance: paddle.Tensor,
        pooled_projection: paddle.Tensor,
    ) -> paddle.Tensor:
        time_emb = self.time_embed(timestep)
        guidance_emb = self.guidance_embed(guidance)
        pooled_emb = self.text_embed(pooled_projection)
        return time_emb + guidance_emb + pooled_emb


# ---------------------------------------------------------------------------
# RoPE 位置编码 (Rotary Position Embedding)
# ---------------------------------------------------------------------------


class FluxRoPE(nn.Layer):
    """Rotary Position Embedding with multi-axis support for Flux.

    Flux uses 3 axes: (time=16, height=56, width=56) dimensions for RoPE.
    """

    def __init__(self, theta: int = 10000, axes_dim: Tuple[int, ...] = (16, 56, 56)) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute cos/sin RoPE embeddings from position IDs.

        Args:
            ids: [seq_len, n_axes] position indices.

        Returns:
            Tuple of (cos, sin) each of shape [seq_len, total_dim].
        """
        cos_list, sin_list = [], []

        for i, dim in enumerate(self.axes_dim):
            pos = ids[:, i].cast(paddle.float32)
            half_dim = dim // 2
            freqs = 1.0 / (self.theta ** (paddle.arange(0, half_dim, dtype=paddle.float32) / half_dim))
            angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)
            cos_list.append(paddle.cos(angles).repeat_interleave(2, axis=-1))
            sin_list.append(paddle.sin(angles).repeat_interleave(2, axis=-1))

        cos_emb = paddle.concat(cos_list, axis=-1)  # [seq_len, total_dim]
        sin_emb = paddle.concat(sin_list, axis=-1)
        return cos_emb, sin_emb


def apply_rope(x: paddle.Tensor, cos: paddle.Tensor, sin: paddle.Tensor) -> paddle.Tensor:
    """Apply rotary position embedding to query or key tensor.

    Args:
        x: [B, heads, seq_len, head_dim] tensor.
        cos: [seq_len, head_dim] cosine components.
        sin: [seq_len, head_dim] sine components.

    Returns:
        Rotated tensor of the same shape.
    """
    # 交替旋转 (Interleaved rotation: [-x1, x0, -x3, x2, ...])
    x_rotated = paddle.stack([-x[..., 1::2], x[..., ::2]], axis=-1).flatten(-2)
    cos = cos.unsqueeze(0).unsqueeze(0).cast(x.dtype)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0).cast(x.dtype)
    return x * cos + x_rotated * sin


# ---------------------------------------------------------------------------
# AdaLayerNorm 条件化层 (Adaptive Layer Norm for conditioning)
# ---------------------------------------------------------------------------


class AdaLayerNormZero(nn.Layer):
    """Adaptive LayerNorm with zero-init for double-stream blocks.

    Produces 5 modulation parameters: gate_msa, shift_mlp, scale_mlp, gate_mlp
    plus for the norm itself.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.Silu()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)

    def forward(
        self, x: paddle.Tensor, emb: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        emb = self.silu(emb)
        emb = self.linear(emb).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa.squeeze(1), shift_mlp.squeeze(1), scale_mlp.squeeze(1), gate_mlp.squeeze(1)


class AdaLayerNormZeroSingle(nn.Layer):
    """Adaptive LayerNorm for single-stream blocks — produces norm + gate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.Silu()
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)

    def forward(self, x: paddle.Tensor, emb: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        emb = self.silu(emb)
        emb = self.linear(emb).unsqueeze(1)
        shift, scale, gate = emb.chunk(3, axis=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate.squeeze(1)


class AdaLayerNormContinuous(nn.Layer):
    """Continuous adaptive LayerNorm for the output projection."""

    def __init__(self, dim: int, conditioning_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.Silu()
        self.linear = nn.Linear(conditioning_dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, epsilon=eps, weight_attr=False, bias_attr=False)

    def forward(self, x: paddle.Tensor, conditioning: paddle.Tensor) -> paddle.Tensor:
        emb = self.silu(conditioning)
        emb = self.linear(emb).unsqueeze(1)
        scale, shift = emb.chunk(2, axis=-1)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Transformer 模块 (Transformer blocks)
# ---------------------------------------------------------------------------


class FluxDoubleStreamBlock(nn.Layer):
    """Double-stream MMDiT block — joint attention on image + text streams.

    Both streams share the same attention layer but have separate
    FFN and normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        mlp_dim = int(dim * mlp_ratio)

        # Image stream
        self.norm1 = AdaLayerNormZero(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate=True),
            nn.Linear(mlp_dim, dim),
        )

        # Text (context) stream
        self.norm1_context = AdaLayerNormZero(dim)
        self.attn_qkv_context = nn.Linear(dim, 3 * dim)
        self.attn_out_context = nn.Linear(dim, dim)
        self.norm2_context = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
        self.ff_context = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate=True),
            nn.Linear(mlp_dim, dim),
        )

        # QK norm (RMSNorm on each head)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Forward pass for double-stream block.

        Args:
            hidden_states: Image latent features [B, img_seq, dim].
            encoder_hidden_states: Text features [B, txt_seq, dim].
            temb: Timestep + text conditioning embedding [B, dim].
            image_rotary_emb: (cos, sin) RoPE for joint sequence.

        Returns:
            Tuple of (updated encoder_hidden_states, updated hidden_states).
        """
        B = hidden_states.shape[0]

        # --- 图像流 AdaLN (Image stream AdaLN) ---
        norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        # --- 文本流 AdaLN (Text stream AdaLN) ---
        norm_ctx, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        # --- QKV projection ---
        qkv_img = self.attn_qkv(norm_hs).reshape([B, -1, 3, self.num_heads, self.head_dim])
        q_img, k_img, v_img = qkv_img.unbind(axis=2)  # [B, seq, heads, head_dim]

        qkv_ctx = self.attn_qkv_context(norm_ctx).reshape([B, -1, 3, self.num_heads, self.head_dim])
        q_ctx, k_ctx, v_ctx = qkv_ctx.unbind(axis=2)

        # QK norm
        q_img = self.q_norm(q_img)
        k_img = self.k_norm(k_img)
        q_ctx = self.q_norm(q_ctx)
        k_ctx = self.k_norm(k_ctx)

        # 拼接 joint attention (Concatenate for joint attention)
        q = paddle.concat([q_ctx, q_img], axis=1).transpose([0, 2, 1, 3])  # [B, heads, seq, dim]
        k = paddle.concat([k_ctx, k_img], axis=1).transpose([0, 2, 1, 3])
        v = paddle.concat([v_ctx, v_img], axis=1).transpose([0, 2, 1, 3])

        # 应用 RoPE (Apply RoPE)
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # [B, heads, seq, dim]
        attn = attn.transpose([0, 2, 1, 3]).reshape([B, -1, self.num_heads * self.head_dim])

        # Split back into image and text
        txt_len = encoder_hidden_states.shape[1]
        context_attn = attn[:, :txt_len]
        img_attn = attn[:, txt_len:]

        # --- 图像残差 (Image residual) ---
        img_attn = self.attn_out(img_attn)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * img_attn
        norm_hs2 = self.norm2(hidden_states)
        norm_hs2 = norm_hs2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hs2)

        # --- 文本残差 (Text residual) ---
        context_attn = self.attn_out_context(context_attn)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn
        norm_ctx2 = self.norm2_context(encoder_hidden_states)
        norm_ctx2 = norm_ctx2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_ctx2)

        return encoder_hidden_states, hidden_states


class FluxSingleStreamBlock(nn.Layer):
    """Single-stream block — concatenated image+text self-attention.

    After double-stream blocks merge context, single-stream blocks
    process the combined sequence with self-attention + MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate=True)
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        """Forward pass for single-stream block.

        Args:
            hidden_states: Combined image+text features [B, seq, dim].
            temb: Conditioning embedding [B, dim].
            image_rotary_emb: (cos, sin) RoPE embeddings.

        Returns:
            Updated hidden states [B, seq, dim].
        """
        B = hidden_states.shape[0]
        residual = hidden_states

        norm_hs, gate = self.norm(hidden_states, emb=temb)

        # Parallel attention + MLP
        mlp_hidden = self.act_mlp(self.proj_mlp(norm_hs))

        qkv = self.attn_qkv(norm_hs).reshape([B, -1, 3, self.num_heads, self.head_dim])
        q, k, v = qkv.unbind(axis=2)

        q = self.q_norm(q).transpose([0, 2, 1, 3])  # [B, heads, seq, dim]
        k = self.k_norm(k).transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose([0, 2, 1, 3]).reshape([B, -1, self.num_heads * self.head_dim])

        # Merge attention + MLP
        hidden_states = paddle.concat([attn, mlp_hidden], axis=-1)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# 主模型 (Main model)
# ---------------------------------------------------------------------------


class FluxForImageGeneration(nn.Layer):
    """Flux Diffusion Transformer for image generation.

    This is a standalone PaddlePaddle implementation adapted from
    PPDiffusers FluxTransformer2DModel, with no external dependencies.

    Architecture (Flux-dev defaults):
      - 19 double-stream blocks (joint image-text attention)
      - 38 single-stream blocks (concatenated self-attention)
      - 24 attention heads × 128 head_dim = 3072 inner_dim
      - T5 context → 4096-dim projected to inner_dim
      - Pooled CLIP → 768-dim projected as timestep conditioning
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.guidance_embeds = guidance_embeds

        # 位置编码 (Positional encoding)
        self.pos_embed = FluxRoPE(theta=10000, axes_dim=axes_dims_rope)

        # 时间步 + 文本嵌入 (Timestep + text embedding)
        if guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextEmbedding(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
            )
        else:
            self.time_text_embed = CombinedTimestepTextEmbedding(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
            )

        # 输入投影 (Input projections)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        # 双流 blocks (Double-stream blocks)
        self.transformer_blocks = nn.LayerList(
            [
                FluxDoubleStreamBlock(
                    dim=self.inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 单流 blocks (Single-stream blocks)
        self.single_transformer_blocks = nn.LayerList(
            [
                FluxSingleStreamBlock(
                    dim=self.inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 输出层 (Output layers)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        pooled_projections: paddle.Tensor,
        timestep: paddle.Tensor,
        img_ids: paddle.Tensor,
        txt_ids: paddle.Tensor,
        guidance: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Forward pass of the Flux transformer.

        Args:
            hidden_states: Patchified image latents [B, img_seq, in_channels].
            encoder_hidden_states: T5 text encodings [B, txt_seq, joint_attention_dim].
            pooled_projections: CLIP pooled text embeddings [B, pooled_projection_dim].
            timestep: Denoising timestep [B].
            img_ids: Image position IDs [img_seq, 3].
            txt_ids: Text position IDs [txt_seq, 3].
            guidance: Guidance scale embedding [B] (only for Flux-dev).

        Returns:
            Denoised output [B, img_seq, patch_size^2 * out_channels].
        """
        # 输入投影 (Project inputs to inner_dim)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 时间步嵌入 (Timestep embedding — scale to [0, 1000])
        timestep = timestep.cast(hidden_states.dtype) * 1000.0
        if guidance is not None:
            guidance = guidance.cast(hidden_states.dtype) * 1000.0

        if self.guidance_embeds and guidance is not None:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)

        # RoPE 位置编码 (Compute RoPE from position IDs)
        ids = paddle.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        # 双流 blocks (Double-stream: joint attention on image + text)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # 合并流 (Merge streams for single-stream blocks)
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # 单流 blocks (Single-stream: self-attention on combined sequence)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        # 截取图像部分 (Extract image portion — discard text tokens)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # 输出投影 (Output projection with AdaLN)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output
