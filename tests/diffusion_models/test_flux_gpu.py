#!/usr/bin/env python3
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
GPU validation for Flux diffusion pipeline.

Tests architecture correctness with random weights on A800 GPU.
Kept tests: small transformer (CPU), full synthetic pipeline (CPU/GPU),
full-size model (GPU only, ~24GB VRAM).

Run on CPU (CI):
    cd FastDeploy && pytest tests/diffusion_models/test_flux_gpu.py -v -x \\
        --override-ini="confcutdir=tests/diffusion_models" -k "not full_size"

Run on AI Studio A800:
    ssh aistudio
    cd ~/FastDeploy && pytest tests/diffusion_models/test_flux_gpu.py -v \\
        --override-ini="confcutdir=tests/diffusion_models"
"""

from __future__ import annotations

import paddle
import pytest

HAS_CUDA = paddle.is_compiled_with_cuda()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA available")

# ---------------------------------------------------------------------------
# Shared tiny configs
# ---------------------------------------------------------------------------

TINY_FLUX_KWARGS = dict(
    in_channels=64,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=128,
    num_attention_heads=4,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    guidance_embeds=True,
    axes_dims_rope=(16, 56, 56),
)


def _flux_img_ids(seq_len, h, w, dtype=paddle.float32):
    """Build spatial image IDs for Flux."""
    img_ids = paddle.zeros([seq_len, 3], dtype=dtype)
    for i in range(h):
        for j in range(w):
            img_ids[i * w + j, 1] = float(i)
            img_ids[i * w + j, 2] = float(j)
    return img_ids


# ===================================================================
# 1. Small Transformer (CPU, fast)
# ===================================================================


class TestFluxTransformerSmall:
    """Flux DiT forward pass with tiny config — tests all plumbing."""

    def test_dev_mode(self):
        """Flux-dev with guidance embedding produces correct output shape."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        model = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        model.eval()

        B, img_seq, txt_seq = 1, 64, 16

        with paddle.no_grad():
            output = model(
                hidden_states=paddle.randn([B, img_seq, 64]),
                encoder_hidden_states=paddle.randn([B, txt_seq, 4096]),
                pooled_projections=paddle.randn([B, 768]),
                timestep=paddle.to_tensor([0.5]),
                img_ids=_flux_img_ids(img_seq, 8, 8),
                txt_ids=paddle.zeros([txt_seq, 3]),
                guidance=paddle.to_tensor([3.5]),
            )

        assert output.shape == [B, img_seq, 64]
        assert paddle.isfinite(output).all()

    def test_schnell_mode(self):
        """Flux-schnell (no guidance) produces correct output shape."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        kwargs = {**TINY_FLUX_KWARGS, "guidance_embeds": False}
        model = FluxForImageGeneration(**kwargs)
        model.eval()

        B, img_seq, txt_seq = 1, 64, 16

        with paddle.no_grad():
            output = model(
                hidden_states=paddle.randn([B, img_seq, 64]),
                encoder_hidden_states=paddle.randn([B, txt_seq, 4096]),
                pooled_projections=paddle.randn([B, 768]),
                timestep=paddle.to_tensor([0.5]),
                img_ids=_flux_img_ids(img_seq, 8, 8),
                txt_ids=paddle.zeros([txt_seq, 3]),
                guidance=None,
            )

        assert output.shape == [B, img_seq, 64]
        assert paddle.isfinite(output).all()


# ===================================================================
# 2. Full Synthetic Pipeline (CPU or GPU)
# ===================================================================


class TestFullPipelineSynthetic:
    """End-to-end: scheduler + transformer + unpack + VAE decode."""

    def test_pipeline_produces_valid_decoded_image(self):
        """Denoising loop → unpack → VAE decode yields finite spatial tensor."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        model = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        model.eval()

        vae = AutoencoderKL(scaling_factor=0.3611)
        vae.eval()

        sched = FlowMatchEulerDiscreteScheduler(shift=1.0)

        B = 1
        img_h, img_w = 256, 256
        num_steps = 3
        latent_h, latent_w = img_h // 8, img_w // 8
        h_half, w_half = latent_h // 2, latent_w // 2
        latent_seq = h_half * w_half
        txt_seq = 32

        prompt_embeds = paddle.randn([B, txt_seq, 4096])
        pooled_embeds = paddle.randn([B, 768])
        img_ids = _flux_img_ids(latent_seq, h_half, w_half)
        txt_ids = paddle.zeros([txt_seq, 3])
        guidance = paddle.to_tensor([3.5])

        latents = paddle.randn([B, latent_seq, 64])
        sched.set_timesteps(num_steps, dtype=paddle.float32)

        with paddle.no_grad():
            for i, t in enumerate(sched.timesteps):
                ts = paddle.full([B], t.item())
                noise_pred = model(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    timestep=ts / 1000.0,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                )
                latents = sched.step(noise_pred, i, latents)

        # Unpack + decode
        spatial = DiffusionEngine._unpack_latents(latents, latent_h, latent_w, 64)
        assert spatial.shape == [B, 16, latent_h, latent_w]

        decoded = vae.decode(spatial.cast(paddle.float32))
        assert decoded.shape == [B, 3, latent_h * 8, latent_w * 8]
        assert paddle.isfinite(decoded).all()


# ===================================================================
# 3. Full-Size Flux on GPU (A800, ~24GB VRAM)
# ===================================================================


class TestFluxFullSizeGPU:
    """Large Flux model on GPU — validates multi-layer forward at scale."""

    @skip_no_cuda
    def test_large_forward(self):
        """Large Flux forward produces finite bf16 outputs (10+20 layers)."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        paddle.set_device("gpu:0")
        import gc

        gc.collect()
        paddle.device.cuda.empty_cache()

        # Build directly in bf16 to fit in 80GB A800
        paddle.set_default_dtype("bfloat16")
        try:
            model = FluxForImageGeneration(
                in_channels=64,
                num_layers=10,
                num_single_layers=20,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=4096,
                pooled_projection_dim=768,
                guidance_embeds=True,
            )
        finally:
            paddle.set_default_dtype("float32")
        model.eval()

        B = 1
        img_seq = 256  # 128×128 → 16×16 packed
        txt_seq = 128
        dtype = paddle.bfloat16

        with paddle.no_grad():
            output = model(
                hidden_states=paddle.randn([B, img_seq, 64], dtype=dtype),
                encoder_hidden_states=paddle.randn([B, txt_seq, 4096], dtype=dtype),
                pooled_projections=paddle.randn([B, 768], dtype=dtype),
                timestep=paddle.to_tensor([0.5], dtype=dtype),
                img_ids=_flux_img_ids(img_seq, 16, 16, dtype=dtype),
                txt_ids=paddle.zeros([txt_seq, 3], dtype=dtype),
                guidance=paddle.to_tensor([3.5], dtype=dtype),
            )
        paddle.device.synchronize()

        assert output.shape == [B, img_seq, 64]
        assert output.dtype == paddle.bfloat16
        assert not paddle.isnan(output).any()
        assert not paddle.isinf(output).any()

        del model, output
        gc.collect()
        paddle.device.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
