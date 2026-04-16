# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 Stability AI.
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
SD3 MMDiT (Multi-Modal Diffusion Transformer) for image generation.

Adapted from the Stability AI SD3 paper (arxiv:2403.03206) as a standalone
PaddlePaddle implementation — no ppdiffusers or torch dependencies.

Architecture:
  - N joint transformer blocks with independent img/txt normalization
  - Each block: AdaLN → QKV (separate for img & txt) → joint attention → FFN
  - Learnable positional encoding for spatial positions
  - Timestep + pooled text conditioning via AdaLN-Zero
"""

from __future__ import annotations

import math
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .flux_dit import RMSNorm

# ---------------------------------------------------------------------------
# 辅助模块 (Helper modules)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Layer):
    """2D image to patch embedding via convolution.

    Converts [B, C, H, W] → [B, num_patches, embed_dim].
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Project patches: [B, C, H, W] → [B, num_patches, embed_dim]."""
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W]).transpose([0, 2, 1])  # [B, H*W, C]
        return x


class SD3TimestepEmbedding(nn.Layer):
    """Sinusoidal timestep embedding + MLP for SD3."""

    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, timestep: paddle.Tensor) -> paddle.Tensor:
        """Embed timestep [B] → [B, dim]."""
        half = self.frequency_dim // 2
        freqs = paddle.exp(-math.log(10000.0) * paddle.arange(half, dtype=paddle.float32) / half)
        args = timestep.cast(paddle.float32).unsqueeze(-1) * freqs.unsqueeze(0)
        emb = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        return self.mlp(emb.cast(timestep.dtype))


class SD3CombinedEmbedding(nn.Layer):
    """Combined timestep + pooled text conditioning for SD3.

    SD3 uses CLIP-L (768d) + CLIP-G (1280d) pooled = 2048d projection.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int = 2048) -> None:
        super().__init__()
        self.time_embed = SD3TimestepEmbedding(embedding_dim)
        self.text_proj = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(self, timestep: paddle.Tensor, pooled_projection: paddle.Tensor) -> paddle.Tensor:
        """Combine timestep and pooled text into conditioning vector."""
        temb = self.time_embed(timestep)
        pooled = self.text_proj(pooled_projection)
        return temb + pooled


class SD3AdaLayerNormZero(nn.Layer):
    """Adaptive Layer Norm Zero for SD3 MMDiT blocks.

    Projects the conditioning vector into 6 modulation parameters:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
        self.linear = nn.Linear(dim, 6 * dim)
        self.silu = nn.SiLU()

    def forward(
        self, x: paddle.Tensor, emb: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Apply adaptive normalization.

        Returns:
            (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        params = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = paddle.chunk(params, 6, axis=-1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# SD3 JointTransformerBlock
# ---------------------------------------------------------------------------


class SD3JointTransformerBlock(nn.Layer):
    """SD3 Joint Transformer Block — independent img/txt paths with joint attention.

    Both image and context streams have separate QK norms (matching HuggingFace diffusers):
    - Separate AdaLN for image and context
    - Separate QKV projections with separate QK norms per stream
    - Joint (concatenated) attention
    - Separate output projections and FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        context_pre_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.context_pre_only = context_pre_only
        mlp_dim = int(dim * mlp_ratio)

        # Image stream
        self.norm1 = SD3AdaLayerNormZero(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate=True),
            nn.Linear(mlp_dim, dim),
        )

        # Context (text) stream
        self.norm1_context = SD3AdaLayerNormZero(dim)
        self.attn_qkv_context = nn.Linear(dim, 3 * dim)

        if not context_pre_only:
            self.attn_out_context = nn.Linear(dim, dim)
            self.norm2_context = nn.LayerNorm(dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
            self.ff_context = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(approximate=True),
                nn.Linear(mlp_dim, dim),
            )

        # QK norm — separate norms for image and context streams
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.q_norm_context = RMSNorm(head_dim, eps=1e-6)
        self.k_norm_context = RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Image features [B, img_seq, dim].
            encoder_hidden_states: Text features [B, txt_seq, dim].
            temb: Timestep + text conditioning [B, dim].

        Returns:
            (updated_encoder_hidden_states, updated_hidden_states).
        """
        B = hidden_states.shape[0]

        # --- Image AdaLN ---
        norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        # --- Context AdaLN ---
        norm_ctx, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, temb)

        # --- QKV projection ---
        qkv_img = self.attn_qkv(norm_hs).reshape([B, -1, 3, self.num_heads, self.head_dim])
        q_img, k_img, v_img = qkv_img.unbind(axis=2)

        qkv_ctx = self.attn_qkv_context(norm_ctx).reshape([B, -1, 3, self.num_heads, self.head_dim])
        q_ctx, k_ctx, v_ctx = qkv_ctx.unbind(axis=2)

        # QK norm — separate norms for image vs context
        q_img = self.q_norm(q_img)
        k_img = self.k_norm(k_img)
        q_ctx = self.q_norm_context(q_ctx)
        k_ctx = self.k_norm_context(k_ctx)

        # 拼接 joint attention (Concatenate for joint attention)
        q = paddle.concat([q_ctx, q_img], axis=1).transpose([0, 2, 1, 3])
        k = paddle.concat([k_ctx, k_img], axis=1).transpose([0, 2, 1, 3])
        v = paddle.concat([v_ctx, v_img], axis=1).transpose([0, 2, 1, 3])

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose([0, 2, 1, 3]).reshape([B, -1, self.num_heads * self.head_dim])

        # Split back
        txt_len = encoder_hidden_states.shape[1]
        context_attn = attn[:, :txt_len]
        img_attn = attn[:, txt_len:]

        # --- Image residual ---
        img_attn = self.attn_out(img_attn)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * img_attn
        norm_hs2 = self.norm2(hidden_states)
        norm_hs2 = norm_hs2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hs2)

        # --- Context residual (skip for last block) ---
        if not self.context_pre_only:
            context_attn = self.attn_out_context(context_attn)
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn
            norm_ctx2 = self.norm2_context(encoder_hidden_states)
            norm_ctx2 = norm_ctx2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_ctx2)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# SD3 主模型 (Main model)
# ---------------------------------------------------------------------------


class SD3Transformer2DModel(nn.Layer):
    """Stable Diffusion 3 MMDiT Transformer for image generation.

    Standalone PaddlePaddle implementation based on the SD3 paper
    (arxiv:2403.03206), with no external dependencies.

    Architecture (SD3-medium defaults):
      - 24 joint transformer blocks
      - 24 attention heads × 64 head_dim = 1536 inner_dim
      - 16-channel latent input with 2×2 patch embedding
      - T5 context → 4096-dim projected to inner_dim
      - CLIP-L (768d) + CLIP-G (1280d) pooled = 2048-dim conditioning
      - Sinusoidal positional encoding (not RoPE)
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 2048,
        pos_embed_max_size: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.pos_embed_max_size = pos_embed_max_size

        # 图块嵌入 (Patch embedding)
        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )

        # 位置编码 (Learnable positional embedding)
        self.pos_embed_weight = self.create_parameter(
            shape=[1, pos_embed_max_size * pos_embed_max_size, self.inner_dim],
            default_initializer=nn.initializer.Normal(std=0.02),
        )

        # 时间步 + 文本嵌入 (Timestep + text conditioning)
        self.time_text_embed = SD3CombinedEmbedding(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
        )

        # 文本投影 (Context projection)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)

        # Transformer blocks
        self.joint_transformer_blocks = nn.LayerList(
            [
                SD3JointTransformerBlock(
                    dim=self.inner_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    context_pre_only=(i == num_layers - 1),
                )
                for i in range(num_layers)
            ]
        )

        # 输出层 (Output)
        self.norm_out = nn.LayerNorm(self.inner_dim, epsilon=1e-6, weight_attr=False, bias_attr=False)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)
        self.adaln_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, 2 * self.inner_dim),
        )

    def _get_positional_encoding(self, h: int, w: int) -> paddle.Tensor:
        """Crop positional encoding for the given spatial size (center crop).

        Args:
            h: Number of patches in height.
            w: Number of patches in width.

        Returns:
            Positional encoding [1, h*w, inner_dim].

        Raises:
            ValueError: If h or w exceeds pos_embed_max_size.
        """
        if h > self.pos_embed_max_size or w > self.pos_embed_max_size:
            raise ValueError(
                f"Patch dimensions ({h}, {w}) exceed pos_embed_max_size "
                f"({self.pos_embed_max_size}). Input image is too large."
            )
        # 裁剪 learnable pos embed — 中心裁剪匹配 HF diffusers
        pos = self.pos_embed_weight[:, : self.pos_embed_max_size * self.pos_embed_max_size]
        pos = pos.reshape([1, self.pos_embed_max_size, self.pos_embed_max_size, self.inner_dim])
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        pos = pos[:, top : top + h, left : left + w, :].reshape([1, h * w, self.inner_dim])
        return pos

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        pooled_projections: paddle.Tensor,
        timestep: paddle.Tensor,
    ) -> paddle.Tensor:
        """Forward pass of the SD3 MMDiT.

        Args:
            hidden_states: Image latents [B, C, H, W].
            encoder_hidden_states: T5 text encodings [B, txt_seq, joint_attention_dim].
            pooled_projections: Pooled CLIP embeddings [B, pooled_projection_dim].
            timestep: Denoising timestep [B].

        Returns:
            Denoised output [B, C, H, W].
        """
        B, C, H, W = hidden_states.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # 图块嵌入 + 位置编码 (Patch embed + positional encoding)
        hidden_states = self.pos_embed(hidden_states)  # [B, num_patches, inner_dim]
        hidden_states = hidden_states + self._get_positional_encoding(h_patches, w_patches)

        # 时间步嵌入 (Timestep embedding)
        timestep = timestep.cast(hidden_states.dtype) * 1000.0
        temb = self.time_text_embed(timestep, pooled_projections)

        # 文本投影 (Context projection)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Transformer blocks
        for block in self.joint_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        # 输出投影 with AdaLN (Output projection)
        adaln_params = self.adaln_out(temb)
        shift, scale = paddle.chunk(adaln_params, 2, axis=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out(hidden_states)

        # 反向 patchify: [B, num_patches, p*p*C] → [B, C, H, W]
        output = output.reshape([B, h_patches, w_patches, self.patch_size, self.patch_size, self.out_channels])
        output = output.transpose([0, 5, 1, 3, 2, 4])  # [B, C, h, p, w, p]
        output = output.reshape([B, self.out_channels, H, W])

        return output
