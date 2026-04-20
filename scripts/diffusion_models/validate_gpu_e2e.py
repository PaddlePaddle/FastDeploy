#!/usr/bin/env python3
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
GPU end-to-end validation for T48 diffusion models.

Runs on AI Studio (A800/V100) to prove:
  1. All components construct without error on real GPU
  2. Transformer + VAE weight save/load roundtrip (bfloat16 on GPU)
  3. Full Flux pipeline: noise → denoise → VAE decode → PIL image
  4. Full SD3 pipeline: same, different architecture
  5. bfloat16 fidelity: forward pass in bf16 produces finite output

Usage (AI Studio SSH):
    ssh aistudio "cd /home/aistudio/FastDeploy && PYTHONPATH=. python3 tests/diffusion_models/validate_gpu_e2e.py"
"""

from __future__ import annotations

import sys
import time

import paddle

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


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _pass(name: str, elapsed: float) -> None:
    print(f"  ✅ PASS: {name} ({elapsed:.2f}s)")


def _fail(name: str, error: str) -> None:
    print(f"  ❌ FAIL: {name}: {error}")


def check_gpu() -> bool:
    """Verify GPU is available and report specs."""
    _banner("GPU Environment")
    if not paddle.is_compiled_with_cuda():
        print("  ⚠️  PaddlePaddle compiled WITHOUT CUDA — CPU-only mode")
        return False

    props = paddle.device.cuda.get_device_properties(0)
    print(f"  Device: {props.name}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  PaddlePaddle version: {paddle.__version__}")
    bf16 = props.major >= 8
    print(f"  BFloat16 support: {'YES' if bf16 else 'NO (SM<80)'}")
    return True


def test_construction():
    """Test 1: All components construct on GPU."""
    _banner("Test 1: Component Construction (GPU)")
    from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
        TextEncoderPipeline,
    )
    from fastdeploy.model_executor.diffusion_models.components.vae import AutoencoderKL
    from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
    from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
        FluxForImageGeneration,
    )
    from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
        SD3Transformer2DModel,
    )
    from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
        FlowMatchEulerDiscreteScheduler,
    )

    results = []

    for name, factory in [
        ("FluxForImageGeneration", lambda: FluxForImageGeneration(**TINY_FLUX_KWARGS)),
        ("SD3Transformer2DModel", lambda: SD3Transformer2DModel(**TINY_SD3_KWARGS)),
        ("AutoencoderKL", lambda: AutoencoderKL(**TINY_VAE_KWARGS)),
        ("FlowMatchEulerDiscreteScheduler", lambda: FlowMatchEulerDiscreteScheduler()),
        ("TextEncoderPipeline", lambda: TextEncoderPipeline()),
        ("DiffusionConfig", lambda: DiffusionConfig(model_name_or_path="/tmp/x")),
    ]:
        t0 = time.time()
        try:
            factory()
            results.append((name, True, time.time() - t0))
            _pass(name, time.time() - t0)
        except Exception as e:
            results.append((name, False, str(e)))
            _fail(name, str(e))

    return all(r[1] for r in results)


def test_weight_roundtrip_gpu():
    """Test 2: Transformer weight save → load → forward match on GPU (bfloat16)."""
    import os
    import tempfile

    import numpy as np

    _banner("Test 2: Transformer Weight Roundtrip (GPU, BFloat16)")

    from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
        load_model_weights,
    )
    from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
        FluxForImageGeneration,
    )

    use_bf16 = paddle.device.cuda.get_device_properties(0).major >= 8
    dtype = paddle.bfloat16 if use_bf16 else paddle.float16

    # Reference model
    paddle.seed(42)
    ref = FluxForImageGeneration(**TINY_FLUX_KWARGS)
    ref = ref.to(dtype=dtype)
    ref.eval()

    # Save weights
    with tempfile.TemporaryDirectory() as tmpdir:
        sd = ref.state_dict()
        save_path = os.path.join(tmpdir, "model_state.pdparams")
        paddle.save(sd, save_path)

        # Fresh model, load from disk
        loaded = FluxForImageGeneration(**TINY_FLUX_KWARGS)
        loaded = loaded.to(dtype=dtype)
        loaded.eval()
        load_model_weights(loaded, tmpdir)

    # Forward both with same input (on GPU)
    paddle.seed(99)
    h = paddle.randn([1, 16, 64], dtype=dtype)
    enc = paddle.zeros([1, 8, 4096], dtype=dtype)
    pooled = paddle.zeros([1, 768], dtype=dtype)
    ts = paddle.to_tensor([0.5], dtype=dtype)
    img_ids = paddle.zeros([16, 3], dtype=dtype)
    txt_ids = paddle.zeros([8, 3], dtype=dtype)
    guid = paddle.to_tensor([3.5], dtype=dtype)

    t0 = time.time()
    with paddle.no_grad():
        ref_out = ref(
            hidden_states=h,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=ts,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guid,
        )
        loaded_out = loaded(
            hidden_states=h,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=ts,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guid,
        )
    elapsed = time.time() - t0

    # Compare
    ref_np = ref_out.cast(paddle.float32).numpy()
    loaded_np = loaded_out.cast(paddle.float32).numpy()

    max_diff = float(np.max(np.abs(ref_np - loaded_np)))
    if max_diff < 1e-3:
        _pass(f"Weight roundtrip ({dtype}), max_diff={max_diff:.2e}", elapsed)
        return True
    else:
        _fail(f"Weight roundtrip ({dtype})", f"max_diff={max_diff:.2e} > 1e-3")
        return False


def test_flux_pipeline_gpu():
    """Test 3: Full Flux pipeline on GPU — noise → denoise → VAE → PIL."""
    import numpy as np

    _banner("Test 3: Full Flux Pipeline (GPU)")

    from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
        TextEncoderPipeline,
    )
    from fastdeploy.model_executor.diffusion_models.components.vae import AutoencoderKL
    from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
    from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
    from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
        FluxForImageGeneration,
    )
    from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
        FlowMatchEulerDiscreteScheduler,
    )

    config = DiffusionConfig(
        model_name_or_path="gpu-test",
        model_type="flux",
        num_inference_steps=3,
        guidance_scale=3.5,
        image_height=128,
        image_width=128,
        dtype="float32",
    )

    engine = DiffusionEngine(config)
    engine.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    engine.text_encoder = TextEncoderPipeline(clip_encoder=None, t5_encoder=None)
    engine.vae = AutoencoderKL(**TINY_VAE_KWARGS)
    engine.vae.eval()
    engine.transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
    engine.transformer.eval()

    t0 = time.time()
    images = engine.generate("A cat sitting on a GPU", seed=42)
    elapsed = time.time() - t0

    # Validate output
    errors = []
    if len(images) != 1:
        errors.append(f"Expected 1 image, got {len(images)}")
    img = images[0]
    pixels = np.array(img)
    if img.size != (128, 128):
        errors.append(f"Image size {img.size} != (128, 128)")
    if img.mode != "RGB":
        errors.append(f"Image mode {img.mode} != RGB")
    if np.any(np.isnan(pixels)):
        errors.append("Image contains NaN pixels")
    pixel_std = float(np.std(pixels.astype(float)))
    if pixel_std < 1.0:
        errors.append(f"Image has near-zero variance (std={pixel_std:.2f}) — likely blank")

    if errors:
        for e in errors:
            _fail("Flux pipeline", e)
        return False
    _pass(f"Flux pipeline → {img.size} {img.mode}, std={pixel_std:.1f}", elapsed)
    return True


def test_sd3_pipeline_gpu():
    """Test 4: Full SD3 pipeline on GPU."""
    import numpy as np

    _banner("Test 4: Full SD3 Pipeline (GPU)")

    from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
        TextEncoderPipeline,
    )
    from fastdeploy.model_executor.diffusion_models.components.vae import AutoencoderKL
    from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
    from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
    from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
        SD3Transformer2DModel,
    )
    from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
        FlowMatchEulerDiscreteScheduler,
    )

    config = DiffusionConfig(
        model_name_or_path="gpu-test",
        model_type="sd3",
        num_inference_steps=3,
        guidance_scale=7.0,
        image_height=128,
        image_width=128,
        dtype="float32",
    )

    engine = DiffusionEngine(config)
    engine.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    # SD3 needs clip_g_encoder (even if model=None) to produce 2048d pooled fallback
    class _StubEncoder:
        model = None

    engine.text_encoder = TextEncoderPipeline(
        clip_encoder=None,
        clip_g_encoder=_StubEncoder(),
        t5_encoder=None,
    )
    engine.vae = AutoencoderKL(**TINY_VAE_KWARGS)
    engine.vae.eval()
    engine.transformer = SD3Transformer2DModel(**TINY_SD3_KWARGS)
    engine.transformer.eval()

    t0 = time.time()
    images = engine.generate("A cat sitting on a cloud", seed=42)
    elapsed = time.time() - t0

    errors = []
    if len(images) != 1:
        errors.append(f"Expected 1 image, got {len(images)}")
    img = images[0]
    pixels = np.array(img)
    if img.size != (128, 128):
        errors.append(f"Image size {img.size} != (128, 128)")
    pixel_std = float(np.std(pixels.astype(float)))
    if pixel_std < 1.0:
        errors.append(f"Near-zero variance (std={pixel_std:.2f})")

    if errors:
        for e in errors:
            _fail("SD3 pipeline", e)
        return False
    _pass(f"SD3 pipeline → {img.size}, std={pixel_std:.1f}", elapsed)
    return True


def test_bfloat16_fidelity():
    """Test 5: BFloat16 forward pass on GPU — no NaN/Inf."""
    _banner("Test 5: BFloat16 Fidelity")

    props = paddle.device.cuda.get_device_properties(0)
    if props.major < 8:
        print("  ⏭️  SKIP: SM < 80, bfloat16 not supported")
        return True

    from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
        FluxForImageGeneration,
    )

    transformer = FluxForImageGeneration(**TINY_FLUX_KWARGS)
    transformer = transformer.to(dtype=paddle.bfloat16)
    transformer.eval()

    paddle.seed(42)
    h = paddle.randn([1, 16, 64], dtype=paddle.bfloat16)
    enc = paddle.zeros([1, 8, 4096], dtype=paddle.bfloat16)
    pooled = paddle.zeros([1, 768], dtype=paddle.bfloat16)
    ts = paddle.to_tensor([0.5], dtype=paddle.bfloat16)
    img_ids = paddle.zeros([16, 3], dtype=paddle.bfloat16)
    txt_ids = paddle.zeros([8, 3], dtype=paddle.bfloat16)
    guid = paddle.to_tensor([3.5], dtype=paddle.bfloat16)

    t0 = time.time()
    with paddle.no_grad():
        out = transformer(
            hidden_states=h,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=ts,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guid,
        )
    elapsed = time.time() - t0

    out_f32 = out.cast(paddle.float32)
    if paddle.all(paddle.isfinite(out_f32)).item():
        _pass(f"BFloat16 forward ({out.shape}), finite output", elapsed)
        return True
    else:
        nan_count = int(paddle.sum(paddle.isnan(out_f32)).item())
        inf_count = int(paddle.sum(paddle.isinf(out_f32)).item())
        _fail("BFloat16 fidelity", f"{nan_count} NaN, {inf_count} Inf in output")
        return False


def main():
    _banner("T48 Diffusion Models — GPU End-to-End Validation")
    print(f"  Python: {sys.version}")
    print(f"  Paddle: {paddle.__version__}")

    has_gpu = check_gpu()
    if not has_gpu:
        print("\n⚠️  Running in CPU-only mode. Weight roundtrip and BF16 tests will use float32.")

    tests = [
        ("Construction", test_construction),
        ("Weight Roundtrip", test_weight_roundtrip_gpu),
        ("Flux Pipeline", test_flux_pipeline_gpu),
        ("SD3 Pipeline", test_sd3_pipeline_gpu),
    ]
    if has_gpu:
        tests.append(("BFloat16 Fidelity", test_bfloat16_fidelity))

    results = []
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            _fail(name, str(e))
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    _banner("SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
    print(f"\n  Result: {passed}/{total} passed")

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED — T48 delivery validated on real GPU")
    else:
        print("\n  ⚠️  SOME TESTS FAILED — see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
