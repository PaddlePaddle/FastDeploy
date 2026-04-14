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
DiffusionEngine — orchestrates Flux / SD3 denoising pipelines.

Pipeline stages:
  1. Text encoding (CLIP-L pooled + T5-XXL sequence)
  2. Noise initialization (Gaussian in latent space)
  3. Denoising loop (scheduler + transformer forward)
  4. VAE decode (latents → pixels)
  5. Post-processing (tensor → PIL Image)

Supported model types:
  - "flux": Flux.1-dev / Flux.1-schnell (packed sequence, RoPE, double/single stream)
  - "sd3": Stable Diffusion 3 / 3.5 (spatial latents, learnable pos embed, joint blocks)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import paddle

from .components.text_encoder import TextEncoderPipeline
from .components.vae import AutoencoderKL
from .config import DiffusionConfig
from .models.flux_dit import FluxForImageGeneration
from .models.sd3_dit import SD3Transformer2DModel
from .schedulers.flow_matching import FlowMatchEulerDiscreteScheduler

logger = logging.getLogger(__name__)


class DiffusionEngine:
    """Orchestrates the full Flux / SD3 text-to-image diffusion pipeline.

    Usage (Flux):
        config = DiffusionConfig(model_name_or_path="black-forest-labs/FLUX.1-dev")
        engine = DiffusionEngine(config)
        engine.load()
        images = engine.generate("A cat sitting on a cloud")

    Usage (SD3):
        config = DiffusionConfig(
            model_name_or_path="stabilityai/stable-diffusion-3-medium",
            model_type="sd3",
        )
        engine = DiffusionEngine(config)
        engine.load()
        images = engine.generate("A cat sitting on a cloud")
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        config.validate()

        self.transformer: Optional[Union[FluxForImageGeneration, SD3Transformer2DModel]] = None
        self.vae: Optional[AutoencoderKL] = None
        self.text_encoder: Optional[TextEncoderPipeline] = None
        self.scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None

    def load(self) -> None:
        """Load all pipeline components from the model path."""
        model_path = self.config.model_name_or_path
        dtype = self.config.get_paddle_dtype()
        model_type = self.config.model_type

        logger.info("Loading %s pipeline from %s (dtype=%s)", model_type, model_path, self.config.dtype)

        # 1. 调度器 (Scheduler)
        scheduler_shift = 1.0 if model_type == "flux" else 3.0
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=scheduler_shift,
        )

        # 2. 文本编码器 (Text encoders)
        self.text_encoder = TextEncoderPipeline.from_pretrained(
            model_path,
            dtype=dtype,
            max_sequence_length=self.config.max_sequence_length,
            model_type=model_type,
        )

        # 3. VAE
        vae_path = self.config.vae_path or model_path
        self.vae = AutoencoderKL.from_pretrained(vae_path, dtype=dtype)

        # 4. Transformer — build architecture, load weights
        if model_type == "sd3":
            self.transformer = SD3Transformer2DModel()
        else:
            self.transformer = FluxForImageGeneration(
                guidance_embeds=(self.config.guidance_scale > 0.0),
            )

        # 加载 Transformer 权重 (Load transformer weights from checkpoint)
        from .components.weight_utils import load_model_weights

        load_model_weights(self.transformer, model_path, subfolder="transformer", dtype=dtype)

        self.transformer = self.transformer.to(dtype=dtype)
        self.transformer.eval()

        logger.info("%s pipeline loaded successfully", model_type.upper())

    def _prepare_latent_image_ids(self, height: int, width: int, dtype: paddle.dtype) -> paddle.Tensor:
        """Create position IDs for image latent patches.

        Flux uses 3-axis position IDs: (batch_index=0, row, col).

        Args:
            height: Latent height (image_height // 16).
            width: Latent width (image_width // 16).
            dtype: Tensor dtype.

        Returns:
            Image position IDs [height * width, 3].
        """
        latent_h = height // 2  # Flux packs 2×2 latent patches
        latent_w = width // 2

        img_ids = paddle.zeros([latent_h, latent_w, 3], dtype=dtype)
        row_ids = paddle.arange(latent_h, dtype=dtype).unsqueeze(1).expand([latent_h, latent_w])
        col_ids = paddle.arange(latent_w, dtype=dtype).unsqueeze(0).expand([latent_h, latent_w])
        img_ids[:, :, 1] = row_ids
        img_ids[:, :, 2] = col_ids

        return img_ids.reshape([-1, 3])

    def _prepare_text_ids(self, seq_len: int, dtype: paddle.dtype) -> paddle.Tensor:
        """Create position IDs for text tokens (all zeros for Flux).

        Args:
            seq_len: Text sequence length.
            dtype: Tensor dtype.

        Returns:
            Text position IDs [seq_len, 3].
        """
        return paddle.zeros([seq_len, 3], dtype=dtype)

    @paddle.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list:
        """Generate images from text prompts.

        Dispatches to the appropriate pipeline based on config.model_type.

        Args:
            prompt: Single prompt string or list of prompts.
            num_inference_steps: Override config num_inference_steps.
            guidance_scale: Override config guidance_scale.
            height: Override config image_height.
            width: Override config image_width.
            seed: Random seed for reproducibility.

        Returns:
            List of PIL.Image.Image objects.
        """
        if self.transformer is None:
            raise RuntimeError("Pipeline not loaded. Call engine.load() first.")

        if self.config.model_type == "sd3":
            return self._generate_sd3(prompt, num_inference_steps, guidance_scale, height, width, seed)
        return self._generate_flux(prompt, num_inference_steps, guidance_scale, height, width, seed)

    def _generate_flux(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list:
        """Flux text-to-image generation (packed sequence pipeline)."""
        # 参数解析 (Resolve parameters)
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        num_steps = num_inference_steps or self.config.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        img_h = height or self.config.image_height
        img_w = width or self.config.image_width
        dtype = self.config.get_paddle_dtype()

        # Flux latent dimensions: image / 8 for VAE, then / 2 for patch packing
        latent_h = img_h // 8
        latent_w = img_w // 8

        # 1. 文本编码 (Text encoding)
        text_output = self.text_encoder.encode(prompt, dtype=dtype)
        prompt_embeds = text_output.prompt_embeds  # [B, seq_len, 4096]
        pooled_embeds = text_output.pooled_prompt_embeds  # [B, 768]

        # 2. 位置 ID (Position IDs)
        img_ids = self._prepare_latent_image_ids(latent_h, latent_w, dtype)
        txt_ids = self._prepare_text_ids(prompt_embeds.shape[1], dtype)

        # 3. 噪声初始化 (Initialize noise)
        if seed is not None:
            paddle.seed(seed)

        # Flux 使用 packed latents: [B, (H/2)*(W/2), C*4]
        num_channels = self.transformer.in_channels
        latent_seq_len = (latent_h // 2) * (latent_w // 2)
        latents = paddle.randn([batch_size, latent_seq_len, num_channels], dtype=dtype)

        # 4. 设置调度器 (Set up scheduler)
        self.scheduler.set_timesteps(num_steps, dtype=dtype)

        # 5. Guidance 张量 (Guidance tensor for Flux-dev)
        guidance_tensor = None
        if self.transformer.guidance_embeds and guidance > 0:
            guidance_tensor = paddle.full([batch_size], guidance, dtype=dtype)

        # 6. 去噪循环 (Denoising loop)
        for i, t in enumerate(self.scheduler.timesteps):
            timestep = paddle.full([batch_size], t.item(), dtype=dtype)

            # 模型前向 (Transformer forward)
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=timestep / 1000.0,  # Normalize back to [0, 1]
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance_tensor,
            )

            # 调度器步进 (Scheduler step)
            latents = self.scheduler.step(noise_pred, i, latents)

        # 7. Unpack latents: [B, seq, C] → [B, C, H/8, W/8]
        latents = self._unpack_latents(latents, latent_h, latent_w, num_channels)

        # 8. VAE 解码 (VAE decode)
        images = self.vae.decode(latents)

        # 9. 后处理 (Post-process to PIL)
        return AutoencoderKL.latents_to_pil(images)

    def _generate_sd3(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list:
        """SD3 text-to-image generation (spatial latent pipeline)."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        num_steps = num_inference_steps or self.config.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        img_h = height or self.config.image_height
        img_w = width or self.config.image_width
        dtype = self.config.get_paddle_dtype()

        do_cfg = guidance > 1.0

        # SD3 latent: image / 8 (no extra packing)
        latent_h = img_h // 8
        latent_w = img_w // 8
        latent_channels = 16  # SD3 VAE channels

        # 1. 文本编码 (Text encoding)
        text_output = self.text_encoder.encode(prompt, dtype=dtype)
        prompt_embeds = text_output.prompt_embeds  # [B, seq_len, 4096]
        pooled_embeds = text_output.pooled_prompt_embeds  # [B, 2048] for SD3 (CLIP-L 768 + CLIP-G 1280)

        # 无条件嵌入用于 CFG (Unconditional embeddings for classifier-free guidance)
        if do_cfg:
            uncond_embeds = paddle.zeros_like(prompt_embeds)
            uncond_pooled = paddle.zeros_like(pooled_embeds)

        # 2. 噪声初始化 (Initialize noise — spatial latents for SD3)
        if seed is not None:
            paddle.seed(seed)
        latents = paddle.randn([batch_size, latent_channels, latent_h, latent_w], dtype=dtype)

        # 3. 设置调度器 (Set up scheduler)
        self.scheduler.set_timesteps(num_steps, dtype=dtype)

        # 4. 去噪循环 (Denoising loop)
        for i, t in enumerate(self.scheduler.timesteps):
            timestep = paddle.full([batch_size], t.item(), dtype=dtype)

            # SD3 使用空间 (B,C,H,W) latent 输入
            # SD3 uses spatial [B, C, H, W] latent input
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=timestep / 1000.0,
            )

            # 分类器自由引导 (Classifier-free guidance)
            if do_cfg:
                noise_pred_uncond = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=uncond_embeds,
                    pooled_projections=uncond_pooled,
                    timestep=timestep / 1000.0,
                )
                noise_pred = noise_pred_uncond + guidance * (noise_pred - noise_pred_uncond)

            # 调度器步进 (Scheduler step)
            latents = self.scheduler.step(noise_pred, i, latents)

        # 5. VAE 解码 (VAE decode — latents already spatial)
        images = self.vae.decode(latents)

        # 6. 后处理 (Post-process to PIL)
        return AutoencoderKL.latents_to_pil(images)

    @staticmethod
    def _unpack_latents(
        latents: paddle.Tensor,
        latent_h: int,
        latent_w: int,
        num_channels: int,
    ) -> paddle.Tensor:
        """Unpack Flux packed latents to spatial format.

        Flux packs 2×2 patches into the channel dimension:
        [B, (H/2)*(W/2), C*4] → [B, C, H, W]

        Args:
            latents: Packed latent tensor [B, seq, C].
            latent_h: Spatial latent height (H/8 from image).
            latent_w: Spatial latent width (W/8 from image).
            num_channels: Number of latent channels (before packing).

        Returns:
            Spatial latent tensor [B, C//4, H, W].
        """
        B = latents.shape[0]
        h_half = latent_h // 2
        w_half = latent_w // 2
        c_per_patch = num_channels // 4  # 64 // 4 = 16 channels

        # [B, h*w, C] → [B, h, w, C] → [B, h, w, 2, 2, c] → [B, c, H, W]
        latents = latents.reshape([B, h_half, w_half, 2, 2, c_per_patch])
        latents = latents.transpose([0, 5, 1, 3, 2, 4])  # [B, c, h, 2, w, 2]
        latents = latents.reshape([B, c_per_patch, latent_h, latent_w])

        return latents
