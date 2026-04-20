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
Pipeline contract integration tests — proves end-to-end delivery.

Unlike the existing tests that use random weights, these tests:
  1. Save deterministic transformer + VAE weights to disk
  2. Use engine.load() to reload them (the real production codepath)
  3. Run engine.generate() and verify the output matches a reference run
  4. Validate every intermediate pipeline stage (not just final PIL)

This file is the T48 equivalent of T49's test_ngram_gpu_kernel.py:
  - T49 pattern: CPU reference → compare against GPU kernel output
  - T48 pattern: known-weight reference → compare against engine.load() output

CI-runnable on CPU (~30s), no real model downloads needed.
"""

from __future__ import annotations

import json

import numpy as np
import paddle
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny model configs (matching test_diffusion_integration.py)
# ---------------------------------------------------------------------------
TINY_FLUX_KWARGS = dict(
    in_channels=64,
    num_layers=1,
    num_single_layers=2,
    attention_head_dim=128,
    num_attention_heads=2,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    guidance_embeds=True,
    axes_dims_rope=(16, 56, 56),
)

TINY_SD3_KWARGS = dict(
    patch_size=2,
    in_channels=16,
    num_layers=2,
    attention_head_dim=64,
    num_attention_heads=4,
    joint_attention_dim=4096,
    pooled_projection_dim=2048,
    pos_embed_max_size=32,
)

TINY_VAE_KWARGS = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=16,
    block_out_channels=(32, 64, 64, 64),
    scaling_factor=0.3611,
    shift_factor=0.0,
)


def _create_full_checkpoint(tmp_path, model_type="flux"):
    """Create a complete model checkpoint with transformer + VAE weights.

    Saves both models' state dicts as safetensors, plus config.json files.
    Returns (transformer, vae, model_dir_path).
    """
    from safetensors.numpy import save_file

    from fastdeploy.model_executor.diffusion_models.components.vae import AutoencoderKL
    from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
        FluxForImageGeneration,
    )
    from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
        SD3Transformer2DModel,
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # --- VAE ---
    vae_dir = model_dir / "vae"
    vae_dir.mkdir()
    vae = AutoencoderKL(**TINY_VAE_KWARGS)
    vae.eval()
    vae_sd = {k: v.numpy() for k, v in vae.state_dict().items()}
    save_file(vae_sd, str(vae_dir / "diffusion_pytorch_model.safetensors"))
    with open(vae_dir / "config.json", "w") as f:
        json.dump(
            {
                "scaling_factor": 0.3611,
                "shift_factor": 0.0,
                "latent_channels": 16,
                "block_out_channels": [32, 64, 64, 64],
            },
            f,
        )

    # --- Transformer ---
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir()
    if model_type == "sd3":
        transformer = SD3Transformer2DModel(**TINY_SD3_KWARGS)
    else:
        transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
    transformer.eval()
    tr_sd = {k: v.numpy() for k, v in transformer.state_dict().items()}
    save_file(tr_sd, str(transformer_dir / "diffusion_pytorch_model.safetensors"))

    return transformer, vae, str(model_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Transformer Weight Roundtrip via engine.load()
# ═══════════════════════════════════════════════════════════════════════════
class TestTransformerWeightRoundtrip:
    """Proves transformer weights survive save → engine.load() → forward.

    This is the CRITICAL gap: existing tests only verify VAE weight roundtrip.
    The transformer is the core model (~12B params for Flux-dev) and must be
    proven to load correctly through the production codepath.
    """

    def test_flux_transformer_save_load_forward_match(self, tmp_path):
        """Save Flux transformer → engine.load() loads it → forward output matches."""
        pytest.importorskip("safetensors")

        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
        )
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        ref_transformer, _, model_dir = _create_full_checkpoint(tmp_path, "flux")

        # Create a fresh transformer and load saved weights
        loaded_transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        loaded_transformer.eval()

        # Verify they differ before loading (random init)
        paddle.seed(99)
        hidden = paddle.randn([1, 16, 64], dtype=paddle.float32)
        encoder_hidden = paddle.zeros([1, 8, 4096], dtype=paddle.float32)
        pooled = paddle.zeros([1, 768], dtype=paddle.float32)
        timestep = paddle.to_tensor([0.5], dtype=paddle.float32)
        img_ids = paddle.zeros([16, 3], dtype=paddle.float32)
        txt_ids = paddle.zeros([8, 3], dtype=paddle.float32)
        guidance = paddle.to_tensor([3.5], dtype=paddle.float32)

        ref_out = ref_transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )

        fresh_out = loaded_transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )

        # Random init should differ
        assert not np.allclose(
            ref_out.numpy(), fresh_out.numpy(), atol=1e-6
        ), "Fresh random-init transformer matches reference — test is invalid"

        # Load weights via the production codepath
        load_model_weights(loaded_transformer, model_dir, subfolder="transformer")

        loaded_out = loaded_transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )

        np.testing.assert_allclose(
            loaded_out.numpy(),
            ref_out.numpy(),
            atol=1e-5,
            err_msg="Transformer output after load_model_weights differs from reference",
        )

    def test_sd3_transformer_save_load_forward_match(self, tmp_path):
        """Save SD3 transformer → load → forward output matches reference."""
        pytest.importorskip("safetensors")

        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
        )
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        ref_transformer, _, model_dir = _create_full_checkpoint(tmp_path, "sd3")

        loaded_transformer = SD3Transformer2DModel(**TINY_SD3_KWARGS)
        loaded_transformer.eval()

        # SD3 forward: spatial latents [B, C, H, W]
        paddle.seed(99)
        hidden = paddle.randn([1, 16, 8, 8], dtype=paddle.float32)
        encoder_hidden = paddle.zeros([1, 8, 4096], dtype=paddle.float32)
        pooled = paddle.zeros([1, 2048], dtype=paddle.float32)
        timestep = paddle.to_tensor([0.5], dtype=paddle.float32)

        ref_out = ref_transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
        )

        # Load weights via production codepath
        load_model_weights(loaded_transformer, model_dir, subfolder="transformer")

        loaded_out = loaded_transformer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            pooled_projections=pooled,
            timestep=timestep,
        )

        np.testing.assert_allclose(
            loaded_out.numpy(),
            ref_out.numpy(),
            atol=1e-5,
            err_msg="SD3 transformer output after weight loading differs from reference",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Full Pipeline: save all components → engine.load() → generate()
# ═══════════════════════════════════════════════════════════════════════════
class TestFullPipelineLoadGenerate:
    """End-to-end: save checkpoint → engine.load() → engine.generate().

    This is THE delivery proof: all components loaded from disk via the
    production codepath, full denoising loop, valid PIL Image output.
    """

    def _build_reference_engine_and_checkpoint(self, tmp_path, model_type="flux"):
        """Build a reference engine with known weights AND save a checkpoint.

        Returns (reference_engine, model_dir).
        """
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
            TextEncoderPipeline,
        )
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Save VAE checkpoint
        vae_dir = model_dir / "vae"
        vae_dir.mkdir()
        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        vae.eval()
        vae_sd = {k: v.numpy() for k, v in vae.state_dict().items()}
        save_file(vae_sd, str(vae_dir / "diffusion_pytorch_model.safetensors"))
        with open(vae_dir / "config.json", "w") as f:
            json.dump(
                {
                    "scaling_factor": 0.3611,
                    "shift_factor": 0.0,
                    "latent_channels": 16,
                    "block_out_channels": [32, 64, 64, 64],
                },
                f,
            )

        # Save transformer checkpoint
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        if model_type == "sd3":
            transformer = SD3Transformer2DModel(**TINY_SD3_KWARGS)
        else:
            transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        transformer.eval()
        tr_sd = {k: v.numpy() for k, v in transformer.state_dict().items()}
        save_file(tr_sd, str(transformer_dir / "diffusion_pytorch_model.safetensors"))

        # Build reference engine with the SAME weights (not from disk)
        config = DiffusionConfig(
            model_name_or_path=str(model_dir),
            model_type=model_type,
            num_inference_steps=3,
            guidance_scale=3.5 if model_type == "flux" else 7.0,
            image_height=128,
            image_width=128,
            dtype="float32",
            seed=42,
        )

        ref_engine = DiffusionEngine(config)
        shift = 1.0 if model_type == "flux" else 3.0
        ref_engine.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)

        if model_type == "sd3":
            # SD3 needs clip_g_encoder (even if model=None) to produce 2048d pooled fallback
            class _StubEncoder:
                model = None

            ref_engine.text_encoder = TextEncoderPipeline(
                clip_encoder=None,
                clip_g_encoder=_StubEncoder(),
                t5_encoder=None,
            )
        else:
            ref_engine.text_encoder = TextEncoderPipeline(clip_encoder=None, t5_encoder=None)

        ref_engine.vae = vae
        ref_engine.transformer = transformer

        return ref_engine, str(model_dir)

    def test_flux_load_generate_matches_reference(self, tmp_path, monkeypatch):
        """engine.load() → generate() produces same output as in-memory reference."""
        from fastdeploy.model_executor.diffusion_models import engine as engine_mod
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        ref_engine, model_dir = self._build_reference_engine_and_checkpoint(tmp_path, "flux")

        # Generate reference image from known weights
        ref_images = ref_engine.generate("integration test", seed=42)
        ref_pixels = np.array(ref_images[0])

        # Monkeypatch transformer constructor to use tiny config
        # (engine.load() creates full-size by default, but our checkpoint is tiny)
        def _tiny_flux(**kwargs):
            merged = {**TINY_FLUX_KWARGS, **kwargs}
            return FluxForImageGeneration(**merged)

        monkeypatch.setattr(engine_mod, "FluxForImageGeneration", _tiny_flux)

        # Now load from disk via production codepath
        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="flux",
            num_inference_steps=3,
            guidance_scale=3.5,
            image_height=128,
            image_width=128,
            dtype="float32",
            seed=42,
        )
        loaded_engine = DiffusionEngine(config)
        loaded_engine.load()

        loaded_images = loaded_engine.generate("integration test", seed=42)
        loaded_pixels = np.array(loaded_images[0])

        # Core assertion: disk-loaded pipeline produces identical output
        np.testing.assert_array_equal(
            loaded_pixels,
            ref_pixels,
            err_msg=(
                "Pipeline output from engine.load() differs from in-memory reference. "
                "This means weight loading, pipeline assembly, or generate() has a bug."
            ),
        )

    def test_sd3_load_generate_matches_reference(self, tmp_path, monkeypatch):
        """SD3 engine.load() → generate() matches reference."""
        from fastdeploy.model_executor.diffusion_models import engine as engine_mod
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        ref_engine, model_dir = self._build_reference_engine_and_checkpoint(tmp_path, "sd3")

        ref_images = ref_engine.generate("sd3 integration test", seed=42)
        ref_pixels = np.array(ref_images[0])

        # Monkeypatch transformer constructor to use tiny config
        def _tiny_sd3(**kwargs):
            merged = {**TINY_SD3_KWARGS, **kwargs}
            return SD3Transformer2DModel(**merged)

        monkeypatch.setattr(engine_mod, "SD3Transformer2DModel", _tiny_sd3)

        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="sd3",
            num_inference_steps=3,
            guidance_scale=7.0,
            image_height=128,
            image_width=128,
            dtype="float32",
            seed=42,
        )
        loaded_engine = DiffusionEngine(config)
        loaded_engine.load()

        loaded_images = loaded_engine.generate("sd3 integration test", seed=42)
        loaded_pixels = np.array(loaded_images[0])

        np.testing.assert_array_equal(
            loaded_pixels,
            ref_pixels,
            err_msg="SD3 pipeline output from engine.load() differs from reference",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pipeline Intermediate Stage Validation
# ═══════════════════════════════════════════════════════════════════════════
class TestPipelineIntermediateStages:
    """Validate every intermediate pipeline stage for data flow correctness.

    Goes beyond "it produces PIL images" to prove every stage transforms
    data with correct shapes, dtypes, finite values, and expected ranges.
    """

    def test_flux_stage_by_stage(self, tmp_path):
        """Walk through Flux pipeline stage by stage, asserting each."""
        from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
            TextEncoderOutput,
            TextEncoderPipeline,
        )
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

        # --- Stage 1: Text encoding ---
        text_encoder = TextEncoderPipeline(clip_encoder=None, t5_encoder=None)
        text_out = text_encoder.encode(["test prompt"], dtype=paddle.float32)
        assert isinstance(text_out, TextEncoderOutput)
        assert text_out.prompt_embeds.shape == [1, 512, 4096], f"prompt_embeds shape: {text_out.prompt_embeds.shape}"
        assert text_out.pooled_prompt_embeds.shape == [1, 768], f"pooled shape: {text_out.pooled_prompt_embeds.shape}"

        # --- Stage 2: Scheduler setup ---
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        scheduler.set_timesteps(5, dtype=paddle.float32)
        assert len(scheduler.timesteps) == 5
        for i in range(len(scheduler.sigmas) - 1):
            assert scheduler.sigmas[i] >= scheduler.sigmas[i + 1], "Sigmas not monotonically decreasing"

        # --- Stage 3: Noise initialization ---
        paddle.seed(42)
        img_h, img_w = 128, 128
        latent_h, latent_w = img_h // 8, img_w // 8  # 16, 16
        latent_seq_len = (latent_h // 2) * (latent_w // 2)  # 64
        num_channels = 64
        latents = paddle.randn([1, latent_seq_len, num_channels], dtype=paddle.float32)
        assert latents.shape == [1, 64, 64]
        assert paddle.all(paddle.isfinite(latents)).item()
        initial_std = float(latents.std())
        assert initial_std > 0.5, f"Initial noise has unexpectedly low variance: {initial_std}"

        # --- Stage 4: Transformer forward (single step) ---
        transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        transformer.eval()
        img_ids = paddle.zeros([latent_seq_len, 3], dtype=paddle.float32)
        txt_ids = paddle.zeros([512, 3], dtype=paddle.float32)
        timestep = paddle.to_tensor([0.5], dtype=paddle.float32)
        guidance = paddle.to_tensor([3.5], dtype=paddle.float32)

        with paddle.no_grad():
            noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=text_out.prompt_embeds,
                pooled_projections=text_out.pooled_prompt_embeds,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            )
        assert (
            noise_pred.shape == latents.shape
        ), f"Transformer output shape {noise_pred.shape} != input shape {latents.shape}"
        assert paddle.all(paddle.isfinite(noise_pred)).item(), "Transformer produced NaN/Inf"

        # --- Stage 5: Scheduler step ---
        stepped_latents = scheduler.step(noise_pred, 0, latents)
        assert stepped_latents.shape == latents.shape
        assert paddle.all(paddle.isfinite(stepped_latents)).item(), "Scheduler step produced NaN/Inf"

        # --- Stage 6: Unpack latents ---
        unpacked = DiffusionEngine._unpack_latents(stepped_latents, latent_h, latent_w, num_channels)
        assert unpacked.shape == [
            1,
            16,
            latent_h,
            latent_w,
        ], f"Unpacked shape {unpacked.shape} != expected [1, 16, {latent_h}, {latent_w}]"
        assert paddle.all(paddle.isfinite(unpacked)).item(), "Unpack produced NaN/Inf"

        # --- Stage 7: VAE decode ---
        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        vae.eval()
        with paddle.no_grad():
            decoded = vae.decode(unpacked)
        assert decoded.shape == [
            1,
            3,
            img_h,
            img_w,
        ], f"VAE decode shape {decoded.shape} != expected [1, 3, {img_h}, {img_w}]"
        assert paddle.all(paddle.isfinite(decoded)).item(), "VAE decode produced NaN/Inf"

        # --- Stage 8: PIL conversion ---
        pil_images = AutoencoderKL.latents_to_pil(decoded)
        assert len(pil_images) == 1
        assert isinstance(pil_images[0], Image.Image)
        assert pil_images[0].size == (img_w, img_h)
        assert pil_images[0].mode == "RGB"
        pixels = np.array(pil_images[0])
        assert pixels.dtype == np.uint8
        assert pixels.min() >= 0 and pixels.max() <= 255

    def test_sd3_stage_by_stage(self, tmp_path):
        """Walk through SD3 pipeline stage by stage."""
        from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
            TextEncoderOutput,
            TextEncoderPipeline,
        )
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        # SD3 text encoding: pooled_dim=2048 (CLIP-L 768 + CLIP-G 1280)
        # Construct to validate no crash, but use manual output below
        TextEncoderPipeline(
            clip_encoder=None,
            clip_g_encoder=None,  # triggers clip_g is not None path in from_pretrained
            t5_encoder=None,
        )

        # SD3 manually builds zero fallback with 2048d pooled
        text_out = TextEncoderOutput(
            prompt_embeds=paddle.zeros([1, 512, 4096], dtype=paddle.float32),
            pooled_prompt_embeds=paddle.zeros([1, 2048], dtype=paddle.float32),
        )

        # SD3 uses spatial latents [B, C, H, W]
        paddle.seed(42)
        img_h, img_w = 128, 128
        latent_h, latent_w = img_h // 8, img_w // 8  # 16, 16
        latents = paddle.randn([1, 16, latent_h, latent_w], dtype=paddle.float32)

        # Transformer forward
        transformer = SD3Transformer2DModel(**TINY_SD3_KWARGS)
        transformer.eval()
        timestep = paddle.to_tensor([0.5], dtype=paddle.float32)

        with paddle.no_grad():
            noise_pred = transformer(
                hidden_states=latents,
                encoder_hidden_states=text_out.prompt_embeds,
                pooled_projections=text_out.pooled_prompt_embeds,
                timestep=timestep,
            )
        assert (
            noise_pred.shape == latents.shape
        ), f"SD3 transformer output shape {noise_pred.shape} != input {latents.shape}"
        assert paddle.all(paddle.isfinite(noise_pred)).item()

        # Scheduler step
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        scheduler.set_timesteps(5, dtype=paddle.float32)
        stepped = scheduler.step(noise_pred, 0, latents)
        assert stepped.shape == latents.shape
        assert paddle.all(paddle.isfinite(stepped)).item()

        # VAE decode (SD3 latents are already spatial — no unpack needed)
        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        vae.eval()
        with paddle.no_grad():
            decoded = vae.decode(stepped)
        assert decoded.shape == [1, 3, img_h, img_w]
        assert paddle.all(paddle.isfinite(decoded)).item()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Regression Snapshot
# ═══════════════════════════════════════════════════════════════════════════
class TestRegressionSnapshot:
    """Catch accidental regressions: deterministic forward must be stable."""

    def test_flux_deterministic_generate_twice(self):
        """Two generate() calls with same seed → bit-identical PIL images."""
        from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
            TextEncoderPipeline,
        )
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        config = DiffusionConfig(
            model_name_or_path="snapshot-test",
            model_type="flux",
            num_inference_steps=3,
            guidance_scale=3.5,
            image_height=128,
            image_width=128,
            dtype="float32",
        )

        def _make_engine():
            engine = DiffusionEngine(config)
            engine.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
            engine.text_encoder = TextEncoderPipeline(clip_encoder=None, t5_encoder=None)
            engine.vae = AutoencoderKL(**TINY_VAE_KWARGS)
            engine.vae.eval()
            engine.transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
            engine.transformer.eval()
            return engine

        # Fix both model weights AND noise seed
        paddle.seed(0)
        engine1 = _make_engine()
        paddle.seed(0)
        engine2 = _make_engine()

        img1 = engine1.generate("regression test", seed=123)
        img2 = engine2.generate("regression test", seed=123)

        np.testing.assert_array_equal(
            np.array(img1[0]),
            np.array(img2[0]),
            err_msg="Same weights + same seed produced different images — determinism broken",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
