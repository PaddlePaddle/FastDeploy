# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
AutoencoderKL for Flux / SD3 latent-pixel conversion.

Flux uses a 16-channel VAE (in_channels=16) with scaling_factor=0.3611.
SD3 uses a 16-channel VAE with scaling_factor=1.5305 and shift_factor=0.0609.

Architecture: Conv2D encoder/decoder with ResNet blocks and attention,
following the standard KL-VAE design from LDM / Stable Diffusion.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAE building blocks
# ---------------------------------------------------------------------------


class ResnetBlock2D(nn.Layer):
    """ResNet block with GroupNorm for VAE encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2D(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class Downsample2D(nn.Layer):
    """Strided convolution downsample (2×)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2D(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.pad(x, [0, 1, 0, 1], mode="constant", value=0)
        return self.conv(x)


class Upsample2D(nn.Layer):
    """Nearest-neighbor upsample (2×) + Conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2D(channels, channels, 3, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class AttentionBlock(nn.Layer):
    """Single-head self-attention for VAE mid-block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2D(channels, channels, 1)
        self.k = nn.Conv2D(channels, channels, 1)
        self.v = nn.Conv2D(channels, channels, 1)
        self.proj_out = nn.Conv2D(channels, channels, 1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape([B, C, H * W]).transpose([0, 2, 1])  # [B, HW, C]
        k = self.k(h).reshape([B, C, H * W])  # [B, C, HW]
        v = self.v(h).reshape([B, C, H * W]).transpose([0, 2, 1])  # [B, HW, C]

        scale = C**-0.5
        attn = paddle.bmm(q, k) * scale  # [B, HW, HW]
        attn = F.softmax(attn, axis=-1)
        out = paddle.bmm(attn, v)  # [B, HW, C]
        out = out.transpose([0, 2, 1]).reshape([B, C, H, W])
        return x + self.proj_out(out)


class Encoder(nn.Layer):
    """VAE Encoder: [B, 3, H, W] → [B, 2*z_channels, H/8, W/8].

    Standard architecture: input conv → 4 down blocks (each: 2 ResNet + optional
    downsample) → mid block (ResNet + Attention + ResNet) → output norm + conv.
    Channel progression: 128 → 256 → 512 → 512.
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        ch = block_out_channels[0]
        self.conv_in = nn.Conv2D(in_channels, ch, 3, padding=1)

        # Down blocks
        self.down_blocks = nn.LayerList()
        for i, out_ch in enumerate(block_out_channels):
            block = nn.LayerList()
            for j in range(num_res_blocks):
                block.append(ResnetBlock2D(ch, out_ch))
                ch = out_ch
            if i < len(block_out_channels) - 1:
                block.append(Downsample2D(ch))
            self.down_blocks.append(block)

        # Mid block
        self.mid_block = nn.LayerList(
            [
                ResnetBlock2D(ch, ch),
                AttentionBlock(ch),
                ResnetBlock2D(ch, ch),
            ]
        )

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2D(ch, 2 * z_channels, 3, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h = self.conv_in(x)
        for down_block in self.down_blocks:
            for layer in down_block:
                h = layer(h)
        for layer in self.mid_block:
            h = layer(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Layer):
    """VAE Decoder: [B, z_channels, H/8, W/8] → [B, 3, H, W].

    Mirror of the encoder with upsampling instead of downsampling.
    Channel progression (reversed): 512 → 512 → 256 → 128.
    """

    def __init__(
        self,
        out_channels: int = 3,
        z_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        num_res_blocks: int = 3,
    ) -> None:
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        ch = reversed_channels[0]

        self.conv_in = nn.Conv2D(z_channels, ch, 3, padding=1)

        # Mid block
        self.mid_block = nn.LayerList(
            [
                ResnetBlock2D(ch, ch),
                AttentionBlock(ch),
                ResnetBlock2D(ch, ch),
            ]
        )

        # Up blocks
        self.up_blocks = nn.LayerList()
        for i, out_ch in enumerate(reversed_channels):
            block = nn.LayerList()
            for j in range(num_res_blocks):
                block.append(ResnetBlock2D(ch, out_ch))
                ch = out_ch
            if i < len(reversed_channels) - 1:
                block.append(Upsample2D(ch))
            self.up_blocks.append(block)

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2D(ch, out_channels, 3, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        h = self.conv_in(x)
        for layer in self.mid_block:
            h = layer(h)
        for up_block in self.up_blocks:
            for layer in up_block:
                h = layer(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# AutoencoderKL — main VAE class
# ---------------------------------------------------------------------------


class AutoencoderKL(nn.Layer):
    """KL-regularized autoencoder for Flux / SD3 latent-pixel conversion.

    Contains a full encoder/decoder architecture with ResNet blocks,
    attention, and optional quant/post-quant convolutions.

    Attributes:
        scaling_factor: Multiplier applied to latents after encoding (and inverse
            before decoding). Flux VAE uses 0.3611, SD3 uses 1.5305.
        shift_factor: Additive shift for SD3 VAE (0.0 for Flux, 0.0609 for SD3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        scaling_factor: float = 0.3611,
        shift_factor: float = 0.0,
    ) -> None:
        super().__init__()
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
        )
        self.quant_conv = nn.Conv2D(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2D(latent_channels, latent_channels, 1)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: paddle.dtype = paddle.float32,
        subfolder: str = "vae",
    ) -> "AutoencoderKL":
        """Load a pretrained VAE from a model directory.

        Args:
            model_path: Root model directory (e.g. "black-forest-labs/FLUX.1-dev").
            dtype: Weight dtype.
            subfolder: Subfolder containing VAE weights.

        Returns:
            Initialized AutoencoderKL instance.
        """
        vae_path = os.path.join(model_path, subfolder)

        # 读取 VAE 配置 (Read VAE config)
        config_file = os.path.join(vae_path, "config.json")
        scaling_factor = 0.3611
        shift_factor = 0.0
        latent_channels = 16
        block_out_channels = (128, 256, 512, 512)

        if os.path.isfile(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Failed to parse %s, using defaults: %s", config_file, e)
                config = {}
            scaling_factor = config.get("scaling_factor", scaling_factor)
            shift_factor = config.get("shift_factor", shift_factor)
            latent_channels = config.get("latent_channels", latent_channels)
            if "block_out_channels" in config:
                block_out_channels = tuple(config["block_out_channels"])

        vae = cls(
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

        # 加载权重 (Load weights — paddle state dict or safetensors)
        weight_file = os.path.join(vae_path, "model_state.pdparams")
        safetensors_file = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")

        if os.path.isfile(weight_file):
            state_dict = paddle.load(weight_file)
            vae.set_state_dict(state_dict)
            logger.info("Loaded VAE weights from %s", weight_file)
        elif os.path.isfile(safetensors_file):
            from .weight_utils import load_safetensors_to_paddle

            state_dict = load_safetensors_to_paddle(safetensors_file)
            vae.set_state_dict(state_dict)
            logger.info("Loaded VAE weights from %s", safetensors_file)

        vae = vae.to(dtype=dtype)
        vae.eval()
        return vae

    def encode(self, image: paddle.Tensor) -> paddle.Tensor:
        """Encode pixel-space image to latent space.

        Args:
            image: [B, 3, H, W] tensor in [-1, 1] range.

        Returns:
            Latent tensor [B, C, H//8, W//8] scaled by scaling_factor.
        """
        h = self.encoder(image)
        h = self.quant_conv(h)
        # 取 DiagonalGaussian 的 mean (Take mean of DiagonalGaussian posterior)
        mean, _ = paddle.chunk(h, 2, axis=1)
        latents = (mean - self.shift_factor) * self.scaling_factor
        return latents

    def decode(self, latents: paddle.Tensor) -> paddle.Tensor:
        """Decode latent space to pixel-space image.

        Args:
            latents: [B, C, H//8, W//8] latent tensor.

        Returns:
            Image tensor [B, 3, H, W] in [-1, 1] range.
        """
        latents = latents / self.scaling_factor + self.shift_factor
        latents = self.post_quant_conv(latents)
        image = self.decoder(latents)
        return image

    @staticmethod
    def latents_to_pil(latent_image: paddle.Tensor) -> list:
        """Convert decoded image tensor to PIL Images.

        Args:
            latent_image: [B, 3, H, W] tensor in [-1, 1] range.

        Returns:
            List of PIL.Image.Image objects.
        """
        from PIL import Image

        # [-1, 1] → [0, 255]
        images = (latent_image / 2.0 + 0.5).clip(0, 1)
        # Ensure float32 before numpy — bfloat16 has limited numpy support
        if images.dtype == paddle.bfloat16:
            images = images.cast(paddle.float32)
        images = images.transpose([0, 2, 3, 1]).numpy()  # [B, H, W, 3]
        images = (images * 255.0).round().astype(np.uint8)

        pil_images = []
        for img_array in images:
            pil_images.append(Image.fromarray(img_array))
        return pil_images
