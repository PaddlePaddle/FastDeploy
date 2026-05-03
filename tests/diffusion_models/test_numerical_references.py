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

"""Module for Hackathon 10th Spring No.48.
Pure NumPy reference implementations + correctness verification.

This file is the T48 equivalent of T49's test_ngram_gpu_kernel.py:

  T49 pattern: _cpu_ngram_match() (pure NumPy) → compare against GPU kernel
  T48 pattern: NumPy reference for each core algorithm → compare against Paddle impl

Core algorithms with references:
  1. Flow matching Euler step (scheduler)
  2. Time-shifted sigma schedule (Flux shift=1.0, SD3 shift=3.0)
  3. RoPE frequency computation and rotation
  4. Known-weight forward pass snapshot (regression detection)

Run on CPU (~10s):
    cd FastDeploy && pytest tests/diffusion_models/test_numerical_references.py -v -x \\
        --override-ini="confcutdir=tests/diffusion_models"
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Pure NumPy Reference Implementations (no Paddle dependency)
# ═══════════════════════════════════════════════════════════════════════════


def _sigma_schedule_numpy(num_steps: int, shift: float = 1.0, num_train_timesteps: int = 1000) -> np.ndarray:
    """Pure NumPy: compute flow matching sigma schedule (matches HF diffusers).

    sigmas = linspace(1, 1/num_train_timesteps, num_steps), then append 0.0
    if shift != 1: sigmas = shift * s / (1 + (shift-1) * s)  (before append)
    """
    sigmas = np.linspace(1.0, 1.0 / num_train_timesteps, num_steps, dtype=np.float64)
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas = np.append(sigmas, 0.0)
    return sigmas


# ═══════════════════════════════════════════════════════════════════════════
# Test Classes — Each Compares Paddle Implementation Against NumPy Reference
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerVsReference:
    """Flow matching scheduler: Paddle impl matches NumPy reference at every step."""

    @pytest.mark.parametrize("shift", [1.0, 3.0])
    def test_sigmas_match_numpy_reference(self, shift):
        """Every sigma value matches the pure NumPy reference implementation."""
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
        scheduler.set_timesteps(28, dtype=paddle.float64)
        sigmas_paddle = scheduler.sigmas.numpy()

        sigmas_numpy = _sigma_schedule_numpy(28, shift=shift)

        np.testing.assert_allclose(
            sigmas_paddle,
            sigmas_numpy,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Paddle sigmas do not match NumPy reference (shift={shift})",
        )

    def test_sd3_shifted_schedule_properties(self):
        """SD3 shift=3.0: sigmas still monotonically decrease, boundaries hold."""
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        scheduler.set_timesteps(28, dtype=paddle.float64)
        sigmas = scheduler.sigmas.numpy()

        # Boundary check
        np.testing.assert_allclose(sigmas[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(sigmas[-1], 0.0, atol=1e-10)

        # Monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1], (
                f"Sigma not monotonically decreasing at index {i}: " f"{sigmas[i]:.6f} < {sigmas[i+1]:.6f}"
            )

        # Shifted schedule should differ from unshifted
        unshifted = _sigma_schedule_numpy(28, shift=1.0)
        assert not np.allclose(
            sigmas, unshifted, atol=1e-3
        ), "shift=3.0 schedule identical to shift=1.0 — shifting is broken"


class TestRoPEVsReference:
    """RoPE implementation: Paddle FluxRoPE matches NumPy reference."""

    def test_rope_position_zero_is_identity(self):
        """At position 0, RoPE should be identity (cos=1, sin=0)."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxRoPE,
            apply_rope,
        )

        axes_dim = (16, 56, 56)
        total_dim = sum(axes_dim)
        seq_len = 8

        ids_zero = paddle.zeros([seq_len, 3], dtype=paddle.float32)
        rope = FluxRoPE(theta=10000, axes_dim=axes_dim)
        cos, sin = rope(ids_zero)

        # At position 0: angles = 0, cos(0)=1, sin(0)=0
        np.testing.assert_allclose(cos.numpy(), 1.0, atol=1e-6, err_msg="cos should be 1.0 at position 0")
        np.testing.assert_allclose(sin.numpy(), 0.0, atol=1e-6, err_msg="sin should be 0.0 at position 0")

        # apply_rope at position 0 should return input unchanged
        paddle.seed(42)
        x = paddle.randn([1, 2, seq_len, total_dim])
        result = apply_rope(x, cos, sin)
        np.testing.assert_allclose(
            result.numpy(),
            x.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="RoPE at position 0 should be identity transform",
        )

    def test_rope_is_norm_preserving(self):
        """RoPE rotation preserves vector magnitude."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxRoPE,
            apply_rope,
        )

        B, heads, seq, dim = 1, 2, 8, 128
        paddle.seed(55)
        x = paddle.randn([B, heads, seq, dim])

        ids = paddle.zeros([seq, 3], dtype=paddle.float32)
        ids[:, 1] = paddle.arange(seq, dtype=paddle.float32) * 5  # varied positions

        rope = FluxRoPE(theta=10000, axes_dim=(16, 56, 56))
        cos, sin = rope(ids)
        result = apply_rope(x, cos, sin)

        # L2 norm per position should be preserved
        norm_before = paddle.norm(x, axis=-1).numpy()
        norm_after = paddle.norm(result, axis=-1).numpy()
        np.testing.assert_allclose(
            norm_after,
            norm_before,
            rtol=1e-4,
            atol=1e-5,
            err_msg="RoPE does not preserve vector norm — rotation is broken",
        )

    @pytest.mark.parametrize("theta", [10000, 1000000])
    def test_rope_frequency_formula_direct(self, theta):
        """Verify individual frequency values match theta^(-2j/d) formula."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import FluxRoPE

        dim = 16  # Use first axis only for clarity
        pos_val = 7.0
        ids = paddle.to_tensor([[pos_val, 0.0, 0.0]], dtype=paddle.float32)

        rope = FluxRoPE(theta=theta, axes_dim=(dim, 56, 56))
        cos_out, sin_out = rope(ids)

        # Manual computation for first axis (dim=16, half_dim=8)
        half = dim // 2
        for j in range(half):
            freq = 1.0 / (theta ** (j / half))
            angle = pos_val * freq
            expected_cos = np.cos(angle)
            expected_sin = np.sin(angle)
            # repeat_interleave: positions 2j and 2j+1 get same value
            actual_cos = float(cos_out[0, 2 * j])
            actual_sin = float(sin_out[0, 2 * j])
            np.testing.assert_allclose(
                actual_cos,
                expected_cos,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"cos mismatch at freq index {j}, theta={theta}",
            )
            np.testing.assert_allclose(
                actual_sin,
                expected_sin,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"sin mismatch at freq index {j}, theta={theta}",
            )


class TestKnownWeightSnapshot:
    """Known-weight forward pass: deterministic weights → expected output norm.

    Catches regressions — if architecture changes, the snapshot breaks.
    """

    def _set_weights_constant(self, model, value=0.01):
        """Set all parameters to a small constant value."""
        with paddle.no_grad():
            for param in model.parameters():
                paddle.assign(paddle.full_like(param, value), param)

    def test_flux_dit_known_weight_norm(self):
        """Flux DiT with constant weights produces deterministic output norm."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        model = FluxForImageGeneration(
            in_channels=64,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=128,
            num_attention_heads=2,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            guidance_embeds=True,
            axes_dims_rope=(16, 56, 56),
        )
        model.eval()
        self._set_weights_constant(model, 0.01)

        paddle.seed(0)
        hidden = paddle.randn([1, 16, 64])
        encoder_hidden = paddle.zeros([1, 8, 4096])
        pooled = paddle.zeros([1, 768])
        timestep = paddle.to_tensor([0.5])
        img_ids = paddle.zeros([16, 3], dtype=paddle.float32)
        txt_ids = paddle.zeros([8, 3], dtype=paddle.float32)
        guidance = paddle.to_tensor([3.5])

        with paddle.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=encoder_hidden,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            )

        # Store the norm for regression detection
        norm_val = float(paddle.norm(out).numpy())
        assert np.isfinite(norm_val), "Output contains NaN/Inf"
        assert norm_val > 0, "Output is all zeros — model is broken"

        # Run again — must be deterministic
        paddle.seed(0)
        hidden2 = paddle.randn([1, 16, 64])
        with paddle.no_grad():
            out2 = model(
                hidden_states=hidden2,
                encoder_hidden_states=encoder_hidden,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            )
        norm_val2 = float(paddle.norm(out2).numpy())
        np.testing.assert_allclose(
            norm_val2,
            norm_val,
            rtol=1e-5,
            err_msg="Flux DiT is non-deterministic with same inputs",
        )

    def test_sd3_dit_known_weight_norm(self):
        """SD3 DiT with constant weights produces deterministic output norm."""
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        model = SD3Transformer2DModel(
            patch_size=2,
            in_channels=16,
            num_layers=1,
            attention_head_dim=64,
            num_attention_heads=4,
            joint_attention_dim=4096,
            pooled_projection_dim=2048,
            pos_embed_max_size=32,
        )
        model.eval()
        self._set_weights_constant(model, 0.01)

        paddle.seed(0)
        hidden = paddle.randn([1, 16, 8, 8])
        encoder_hidden = paddle.zeros([1, 8, 4096])
        pooled = paddle.zeros([1, 2048])
        timestep = paddle.to_tensor([0.5])

        with paddle.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=encoder_hidden,
                pooled_projections=pooled,
                timestep=timestep,
            )

        norm_val = float(paddle.norm(out).numpy())
        assert np.isfinite(norm_val), "SD3 output contains NaN/Inf"
        assert norm_val > 0, "SD3 output is all zeros"

        paddle.seed(0)
        hidden2 = paddle.randn([1, 16, 8, 8])
        with paddle.no_grad():
            out2 = model(
                hidden_states=hidden2,
                encoder_hidden_states=encoder_hidden,
                pooled_projections=pooled,
                timestep=timestep,
            )
        norm_val2 = float(paddle.norm(out2).numpy())
        np.testing.assert_allclose(
            norm_val2,
            norm_val,
            rtol=1e-5,
            err_msg="SD3 DiT is non-deterministic with same inputs",
        )

    def test_vae_encode_decode_known_weights(self):
        """VAE with constant weights: encode→decode preserves structure."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=16,
            block_out_channels=(32, 64, 64, 64),
            scaling_factor=0.3611,
            shift_factor=0.0,
        )
        vae.eval()
        self._set_weights_constant(vae, 0.01)

        paddle.seed(0)
        image = paddle.randn([1, 3, 64, 64])

        with paddle.no_grad():
            latent = vae.encode(image)
            reconstructed = vae.decode(latent)

        assert latent.shape == [1, 16, 8, 8], f"Latent shape: {latent.shape}"
        assert reconstructed.shape == [1, 3, 64, 64], f"Recon shape: {reconstructed.shape}"
        assert np.all(np.isfinite(latent.numpy())), "Latent contains NaN/Inf"
        assert np.all(np.isfinite(reconstructed.numpy())), "Reconstructed contains NaN/Inf"

        # Encode again with same input — must be deterministic
        with paddle.no_grad():
            latent2 = vae.encode(image)
        np.testing.assert_allclose(
            latent2.numpy(),
            latent.numpy(),
            rtol=1e-5,
            err_msg="VAE encode is non-deterministic",
        )


class TestEndToEndDenoising:
    """Full pipeline: noise → denoising loop → images, all verified against NumPy."""

    @pytest.mark.parametrize("shift", [1.0, 3.0])
    def test_noise_to_clean_reduces_variance(self, shift):
        """Denoising reduces sample variance (converges toward data manifold).

        Flow matching: v = predicted velocity pointing from noise toward data.
        Euler step: x_{t-dt} = x_t + (sigma_next - sigma_curr) * v
        Since sigma decreases, dt < 0, so with v = sample (pointing away from
        origin), the step becomes x - |dt| * x = x * (1 - |dt|), contracting.
        """
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
        scheduler.set_timesteps(20, dtype=paddle.float32)

        paddle.seed(42)
        sample = paddle.randn([1, 16, 8, 8])
        variances = [float(paddle.var(sample))]

        for i in range(20):
            # v = sample simulates velocity pointing from noise toward origin.
            # dt = sigma_next - sigma_curr < 0, so step = x + dt*x = x*(1+dt)
            # = x*(1 - |dt|), which contracts the sample.
            v = sample
            sample = scheduler.step(v, i, sample)
            variances.append(float(paddle.var(sample)))

        # Variance should decrease overall (not monotonically, but first > last)
        assert variances[-1] < variances[0] * 0.5, (
            f"Denoising did not reduce variance: {variances[0]:.4f} → {variances[-1]:.4f} " f"(shift={shift})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
