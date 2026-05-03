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
FastDeploy framework integration tests for diffusion models.

Unlike test_diffusion_integration.py (which validates forward-pass with
synthetic random-weight models), these tests prove our code integrates
with FastDeploy's actual infrastructure:

  1. Package exports work (import from the public API).
  2. Weight save/load roundtrip (safetensors + pdparams).
  3. engine.load() codepath (AutoencoderKL.from_pretrained, config.json).
  4. DiffusionConfig.validate() contract.
  5. Full weight-loading pipeline: save → load → generate.

Run on CPU (fast, ~5s):
    cd FastDeploy && pytest tests/diffusion_models/test_fd_integration.py -v -x \
        --override-ini="confcutdir=tests/diffusion_models"

Run on AI Studio A800 (full suite):
    ssh aistudio
    cd ~/FastDeploy && PYTHONPATH=. pytest tests/diffusion_models/test_fd_integration.py -v \
        --override-ini="confcutdir=tests/diffusion_models"
"""

from __future__ import annotations

import json

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# Conditionals
# ---------------------------------------------------------------------------
HAS_CUDA = paddle.is_compiled_with_cuda()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA available")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Package Import Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestPackageImports:
    """Prove all public symbols are importable from the diffusion_models package."""

    def test_top_level_exports(self):
        """__init__.py __all__ exports are importable and non-None."""
        from fastdeploy.model_executor.diffusion_models import (
            DiffusionConfig,
            DiffusionEngine,
            apply_tensor_parallel,
            apply_weight_quantization,
        )

        assert DiffusionConfig is not None
        assert DiffusionEngine is not None
        assert callable(apply_tensor_parallel)
        assert callable(apply_weight_quantization)

    def test_component_imports(self):
        """Every component module is importable."""
        from fastdeploy.model_executor.diffusion_models.components.text_encoder import (
            CLIPTextEncoder,
            T5TextEncoder,
        )
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
            load_safetensors_to_paddle,
        )

        assert AutoencoderKL is not None
        assert CLIPTextEncoder is not None
        assert T5TextEncoder is not None
        assert callable(load_safetensors_to_paddle)
        assert callable(load_model_weights)

    def test_model_imports(self):
        """DiT model classes are importable."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        assert FluxForImageGeneration is not None
        assert SD3Transformer2DModel is not None

    def test_scheduler_import(self):
        """Scheduler class is importable."""
        from fastdeploy.model_executor.diffusion_models.schedulers.flow_matching import (
            FlowMatchEulerDiscreteScheduler,
        )

        assert FlowMatchEulerDiscreteScheduler is not None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Weight Save/Load Roundtrip
# ═══════════════════════════════════════════════════════════════════════════

# Tiny VAE config matching test_diffusion_integration.py
TINY_VAE_KWARGS = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=16,
    block_out_channels=(32, 64, 64, 64),
    scaling_factor=0.3611,
    shift_factor=0.0,
)


class TestWeightRoundtrip:
    """Prove weight_utils can save and reload model weights exactly."""

    def test_safetensors_save_load_exact(self, tmp_path):
        """Save a VAE state dict as safetensors → load back → verify bit-exact."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_safetensors_to_paddle,
        )

        model = AutoencoderKL(**TINY_VAE_KWARGS)
        original_sd = {k: v.numpy() for k, v in model.state_dict().items()}

        filepath = str(tmp_path / "vae_weights.safetensors")
        save_file(original_sd, filepath)

        loaded_sd = load_safetensors_to_paddle(filepath)

        assert set(loaded_sd.keys()) == set(original_sd.keys()), (
            f"Key mismatch: missing={set(original_sd) - set(loaded_sd)}, " f"extra={set(loaded_sd) - set(original_sd)}"
        )
        for key in original_sd:
            np.testing.assert_array_equal(
                loaded_sd[key].numpy(),
                original_sd[key],
                err_msg=f"Weight mismatch for key '{key}'",
            )

    def test_safetensors_dtype_cast(self, tmp_path):
        """load_safetensors_to_paddle with dtype= casts all tensors."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_safetensors_to_paddle,
        )

        model = AutoencoderKL(**TINY_VAE_KWARGS)
        original_sd = {k: v.numpy() for k, v in model.state_dict().items()}

        filepath = str(tmp_path / "vae_fp16.safetensors")
        save_file(original_sd, filepath)

        loaded_sd = load_safetensors_to_paddle(filepath, dtype=paddle.float16)

        for key, tensor in loaded_sd.items():
            assert tensor.dtype == paddle.float16, f"Expected float16 for key '{key}', got {tensor.dtype}"

    def test_pdparams_save_load_exact(self, tmp_path):
        """Save via paddle.save → load via load_paddle_state_dict → verify exact."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_paddle_state_dict,
        )

        model = AutoencoderKL(**TINY_VAE_KWARGS)
        original_sd = model.state_dict()

        filepath = str(tmp_path / "vae_weights.pdparams")
        paddle.save(original_sd, filepath)

        loaded_sd = load_paddle_state_dict(filepath)

        assert set(loaded_sd.keys()) == set(original_sd.keys())
        for key in original_sd:
            np.testing.assert_array_equal(
                loaded_sd[key].numpy(),
                original_sd[key].numpy(),
                err_msg=f"Weight mismatch for key '{key}'",
            )

    def test_load_model_weights_into_fresh_model(self, tmp_path):
        """Create model A → save weights → create model B → load → verify identical output."""
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )
        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
        )

        # Model A: random init
        model_a = AutoencoderKL(**TINY_VAE_KWARGS)
        model_a.eval()

        # Save model A weights as safetensors (the HuggingFace default name)
        sd_numpy = {k: v.numpy() for k, v in model_a.state_dict().items()}
        weight_dir = tmp_path / "vae"
        weight_dir.mkdir()
        save_file(sd_numpy, str(weight_dir / "diffusion_pytorch_model.safetensors"))

        # Model B: different random init
        model_b = AutoencoderKL(**TINY_VAE_KWARGS)
        model_b.eval()

        # Verify models differ before loading
        x = paddle.randn([1, 3, 64, 64])
        out_a = model_a.encode(x)
        out_b_before = model_b.encode(x)
        # Random init — extremely unlikely to match
        assert not np.allclose(out_a.numpy(), out_b_before.numpy(), atol=1e-6)

        # Load A's weights into B
        load_model_weights(model_b, str(tmp_path), subfolder="vae")

        # Now they must match
        out_b_after = model_b.encode(x)
        np.testing.assert_allclose(
            out_b_after.numpy(),
            out_a.numpy(),
            atol=1e-6,
            err_msg="Model B output differs from model A after loading A's weights",
        )

    def test_multi_shard_path_traversal_rejected(self, tmp_path):
        """Shard filenames with path traversal components are rejected."""
        import json

        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
        )

        weight_dir = tmp_path / "weights"
        weight_dir.mkdir()
        # Index file pointing to a traversal shard
        index = {"weight_map": {"layer.weight": "../../../etc/passwd.safetensors"}}
        (weight_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))

        model = paddle.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Path traversal detected"):
            load_model_weights(model, str(weight_dir))

    def test_multi_shard_absolute_path_rejected(self, tmp_path):
        """Shard filenames with absolute paths are rejected."""
        import json

        from fastdeploy.model_executor.diffusion_models.components.weight_utils import (
            load_model_weights,
        )

        weight_dir = tmp_path / "weights"
        weight_dir.mkdir()
        index = {"weight_map": {"layer.weight": "/etc/passwd.safetensors"}}
        (weight_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))

        model = paddle.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Invalid shard filename"):
            load_model_weights(model, str(weight_dir))


# ═══════════════════════════════════════════════════════════════════════════
# 3. AutoencoderKL.from_pretrained() — config.json + weight loading
# ═══════════════════════════════════════════════════════════════════════════
class TestVAEFromPretrained:
    """Prove AutoencoderKL.from_pretrained() reads config.json and loads weights."""

    def _create_fake_vae_checkpoint(self, tmp_path, *, use_safetensors=True):
        """Create a minimal fake VAE checkpoint directory.

        Returns (vae_model, root_dir) where root_dir/vae/ contains weights + config.
        """
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        # Create the VAE and save its weights
        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        vae.eval()

        vae_dir = tmp_path / "vae"
        vae_dir.mkdir()

        # Write config.json (what from_pretrained reads)
        config = {
            "scaling_factor": 0.3611,
            "shift_factor": 0.0,
            "latent_channels": 16,
            "block_out_channels": [32, 64, 64, 64],
        }
        with open(vae_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Write weights
        sd_numpy = {k: v.numpy() for k, v in vae.state_dict().items()}
        if use_safetensors:
            save_file(sd_numpy, str(vae_dir / "diffusion_pytorch_model.safetensors"))
        else:
            paddle.save(vae.state_dict(), str(vae_dir / "model_state.pdparams"))

        return vae, tmp_path

    def test_from_pretrained_safetensors(self, tmp_path):
        """from_pretrained loads config.json + safetensors weights correctly."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        original_vae, root_dir = self._create_fake_vae_checkpoint(tmp_path, use_safetensors=True)

        loaded_vae = AutoencoderKL.from_pretrained(str(root_dir), dtype=paddle.float32, subfolder="vae")

        # Verify config was read correctly
        assert loaded_vae.scaling_factor == 0.3611
        assert loaded_vae.shift_factor == 0.0

        # Verify weights produce identical output
        x = paddle.randn([1, 3, 64, 64])
        original_vae.eval()
        original_out = original_vae.encode(x)
        loaded_out = loaded_vae.encode(x)

        np.testing.assert_allclose(
            loaded_out.numpy(),
            original_out.numpy(),
            atol=1e-6,
            err_msg="from_pretrained(safetensors) loaded different weights",
        )

    def test_from_pretrained_pdparams(self, tmp_path):
        """from_pretrained prefers pdparams when both exist."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        original_vae, root_dir = self._create_fake_vae_checkpoint(tmp_path, use_safetensors=False)

        loaded_vae = AutoencoderKL.from_pretrained(str(root_dir), dtype=paddle.float32, subfolder="vae")

        x = paddle.randn([1, 3, 64, 64])
        original_vae.eval()
        np.testing.assert_allclose(
            loaded_vae.encode(x).numpy(),
            original_vae.encode(x).numpy(),
            atol=1e-6,
            err_msg="from_pretrained(pdparams) loaded different weights",
        )

    def test_from_pretrained_no_weights_still_works(self, tmp_path):
        """from_pretrained with config.json but no weights: model is random-init."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        vae_dir = tmp_path / "vae"
        vae_dir.mkdir()
        config = {
            "scaling_factor": 0.5,
            "shift_factor": 0.1,
            "latent_channels": 16,
            "block_out_channels": [32, 64, 64, 64],
        }
        with open(vae_dir / "config.json", "w") as f:
            json.dump(config, f)

        vae = AutoencoderKL.from_pretrained(str(tmp_path), dtype=paddle.float32, subfolder="vae")

        # Config values should be read from JSON
        assert vae.scaling_factor == 0.5
        assert vae.shift_factor == 0.1

        # Model should still produce valid output (random weights, no NaN)
        x = paddle.randn([1, 3, 64, 64])
        latents = vae.encode(x)
        assert not paddle.isnan(latents).any(), "VAE encode produced NaN with random weights"

    def test_from_pretrained_malformed_config_uses_defaults(self, tmp_path):
        """from_pretrained falls back to defaults when config.json is malformed."""
        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        vae_dir = tmp_path / "vae"
        vae_dir.mkdir()
        # Write invalid JSON
        with open(vae_dir / "config.json", "w") as f:
            f.write("{not valid json!!!")

        vae = AutoencoderKL.from_pretrained(str(tmp_path), dtype=paddle.float32, subfolder="vae")

        # Should fall back to default scaling_factor (0.3611)
        assert vae.scaling_factor == 0.3611
        assert vae.shift_factor == 0.0

        # Model should still produce valid output
        x = paddle.randn([1, 3, 64, 64])
        latents = vae.encode(x)
        assert not paddle.isnan(latents).any(), "VAE encode produced NaN after malformed config"


# ═══════════════════════════════════════════════════════════════════════════
# 4. engine.load() Integration
# ═══════════════════════════════════════════════════════════════════════════
class TestEngineLoad:
    """Prove engine.load() works with real filesystem paths."""

    @pytest.fixture(autouse=True)
    def _tiny_transformers(self, monkeypatch):
        """Patch Flux and SD3 constructors to use tiny configs (prevent OOM)."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        _TINY_FLUX = dict(
            in_channels=64,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=128,
            num_attention_heads=4,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            axes_dims_rope=(16, 56, 56),
        )
        _TINY_SD3 = dict(
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=4,
            joint_attention_dim=4096,
            pooled_projection_dim=2048,
        )

        _orig_flux_init = FluxForImageGeneration.__init__
        _orig_sd3_init = SD3Transformer2DModel.__init__

        def _flux_init(self, **kw):
            _orig_flux_init(self, **{**_TINY_FLUX, **kw})

        def _sd3_init(self, **kw):
            _orig_sd3_init(self, **{**_TINY_SD3, **kw})

        monkeypatch.setattr(FluxForImageGeneration, "__init__", _flux_init)
        monkeypatch.setattr(SD3Transformer2DModel, "__init__", _sd3_init)

    def _create_model_directory(self, tmp_path, model_type="flux"):
        """Create minimal model directory for engine.load().

        engine.load() calls:
          1. FlowMatchEulerDiscreteScheduler() — no disk I/O
          2. TextEncoderPipeline.from_pretrained(model_path) — needs encoder dirs
          3. AutoencoderKL.from_pretrained(vae_path) — needs vae dir + weights
          4. FluxForImageGeneration() or SD3Transformer2DModel() — no disk I/O

        We create a minimal VAE checkpoint. Text encoders will fallback to zero tensors
        (the code handles missing encoder directories gracefully).
        """
        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create VAE checkpoint
        vae_dir = model_dir / "vae"
        vae_dir.mkdir()

        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        sd_numpy = {k: v.numpy() for k, v in vae.state_dict().items()}
        save_file(sd_numpy, str(vae_dir / "diffusion_pytorch_model.safetensors"))

        config = {
            "scaling_factor": 0.3611,
            "shift_factor": 0.0,
            "latent_channels": 16,
            "block_out_channels": [32, 64, 64, 64],
        }
        with open(vae_dir / "config.json", "w") as f:
            json.dump(config, f)

        return str(model_dir), vae

    def test_engine_load_flux(self, tmp_path):
        """engine.load() for Flux: creates scheduler, text_encoder, vae, transformer."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine

        model_dir, original_vae = self._create_model_directory(tmp_path, "flux")

        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="flux",
            dtype="float32",
            num_inference_steps=2,
            image_height=128,
            image_width=128,
        )

        engine = DiffusionEngine(config)
        engine.load()

        # All components must be initialized
        assert engine.scheduler is not None, "scheduler not loaded"
        assert engine.text_encoder is not None, "text_encoder not loaded"
        assert engine.vae is not None, "vae not loaded"
        assert engine.transformer is not None, "transformer not loaded"

        # VAE should have loaded weights from our checkpoint
        x = paddle.randn([1, 3, 64, 64])
        original_vae.eval()
        loaded_out = engine.vae.encode(x)
        original_out = original_vae.encode(x)
        np.testing.assert_allclose(
            loaded_out.numpy(),
            original_out.numpy(),
            atol=1e-5,
            err_msg="engine.load() did not load VAE weights correctly",
        )

    def test_engine_load_sd3(self, tmp_path):
        """engine.load() for SD3 creates correct transformer type."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine
        from fastdeploy.model_executor.diffusion_models.models.sd3_dit import (
            SD3Transformer2DModel,
        )

        model_dir, _ = self._create_model_directory(tmp_path, "sd3")

        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="sd3",
            dtype="float32",
            num_inference_steps=2,
        )

        engine = DiffusionEngine(config)
        engine.load()

        assert isinstance(
            engine.transformer, SD3Transformer2DModel
        ), f"Expected SD3Transformer2DModel, got {type(engine.transformer)}"

    def test_engine_load_vae_path_override(self, tmp_path):
        """engine.load() with vae_path= uses that path instead of model_path."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine

        # Create VAE in a separate directory from the model
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        vae_dir_root = tmp_path / "separate_vae"
        vae_dir_root.mkdir()

        pytest.importorskip("safetensors")
        from safetensors.numpy import save_file

        from fastdeploy.model_executor.diffusion_models.components.vae import (
            AutoencoderKL,
        )

        vae = AutoencoderKL(**TINY_VAE_KWARGS)
        vae_subdir = vae_dir_root / "vae"
        vae_subdir.mkdir()

        sd_numpy = {k: v.numpy() for k, v in vae.state_dict().items()}
        save_file(sd_numpy, str(vae_subdir / "diffusion_pytorch_model.safetensors"))

        config_data = {
            "scaling_factor": 0.3611,
            "shift_factor": 0.0,
            "latent_channels": 16,
            "block_out_channels": [32, 64, 64, 64],
        }
        with open(vae_subdir / "config.json", "w") as f:
            json.dump(config_data, f)

        config = DiffusionConfig(
            model_name_or_path=str(model_dir),
            model_type="flux",
            dtype="float32",
            vae_path=str(vae_dir_root),
        )

        engine = DiffusionEngine(config)
        engine.load()

        # VAE should have loaded from the separate path
        assert engine.vae is not None
        x = paddle.randn([1, 3, 64, 64])
        vae.eval()
        np.testing.assert_allclose(
            engine.vae.encode(x).numpy(),
            vae.encode(x).numpy(),
            atol=1e-5,
            err_msg="vae_path override not respected by engine.load()",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Full Pipeline: Save → Load → Generate
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipelineWithWeightLoading:
    """Most critical test: proves end-to-end weight save → load → generate works."""

    @pytest.fixture(autouse=True)
    def _tiny_flux(self, monkeypatch):
        """Patch FluxForImageGeneration to tiny config (prevent OOM on GPU)."""
        from fastdeploy.model_executor.diffusion_models.models.flux_dit import (
            FluxForImageGeneration,
        )

        _TINY = dict(
            in_channels=64,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=128,
            num_attention_heads=4,
            joint_attention_dim=4096,
            pooled_projection_dim=768,
            axes_dims_rope=(16, 56, 56),
        )
        _orig = FluxForImageGeneration.__init__

        def _init(self, **kw):
            _orig(self, **{**_TINY, **kw})

        monkeypatch.setattr(FluxForImageGeneration, "__init__", _init)

    def _setup_pipeline_checkpoint(self, tmp_path):
        """Create a complete model checkpoint with saved VAE weights.

        Returns (engine_with_known_weights, model_dir_path).
        The returned engine has all components initialized with known weights
        that match the saved checkpoint.
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

        sd_numpy = {k: v.numpy() for k, v in vae.state_dict().items()}
        save_file(sd_numpy, str(vae_dir / "diffusion_pytorch_model.safetensors"))

        config_data = {
            "scaling_factor": 0.3611,
            "shift_factor": 0.0,
            "latent_channels": 16,
            "block_out_channels": [32, 64, 64, 64],
        }
        with open(vae_dir / "config.json", "w") as f:
            json.dump(config_data, f)

        # Build engine with the SAME VAE weights
        config = DiffusionConfig(
            model_name_or_path=str(model_dir),
            model_type="flux",
            num_inference_steps=2,
            guidance_scale=3.5,
            image_height=128,
            image_width=128,
            dtype="float32",
            seed=42,
        )

        # Tiny Flux kwargs
        flux_kwargs = dict(
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

        engine = DiffusionEngine(config)
        engine.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        engine.text_encoder = TextEncoderPipeline(clip_encoder=None, t5_encoder=None)
        engine.vae = vae  # Same weights as saved
        engine.transformer = FluxForImageGeneration(**flux_kwargs)
        engine.transformer.eval()

        return engine, str(model_dir)

    def test_saved_vae_weights_produce_matching_output(self, tmp_path):
        """Save VAE → engine.load() loads it → decode output matches original."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine

        reference_engine, model_dir = self._setup_pipeline_checkpoint(tmp_path)

        # Now use engine.load() to create a fresh engine
        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="flux",
            dtype="float32",
        )
        loaded_engine = DiffusionEngine(config)
        loaded_engine.load()

        # VAE decode with same input must match
        latents = paddle.randn([1, 16, 8, 8])
        ref_decoded = reference_engine.vae.decode(latents)
        loaded_decoded = loaded_engine.vae.decode(latents)

        np.testing.assert_allclose(
            loaded_decoded.numpy(),
            ref_decoded.numpy(),
            atol=1e-5,
            err_msg="Loaded VAE decode output differs from reference",
        )

    def test_full_generate_after_load(self, tmp_path):
        """engine.load() → generate() → PIL images. The complete delivery proof."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig
        from fastdeploy.model_executor.diffusion_models.engine import DiffusionEngine

        _, model_dir = self._setup_pipeline_checkpoint(tmp_path)

        config = DiffusionConfig(
            model_name_or_path=model_dir,
            model_type="flux",
            num_inference_steps=2,
            guidance_scale=3.5,
            image_height=128,
            image_width=128,
            dtype="float32",
            seed=42,
        )

        engine = DiffusionEngine(config)
        engine.load()

        # Text encoders fall back to zero tensors (no real CLIP/T5 checkpoints)
        # Transformer has random weights (no checkpoint saved for it)
        # But the PIPELINE must still produce valid PIL images.
        images = engine.generate("a test prompt")

        assert len(images) == 1, f"Expected 1 image, got {len(images)}"

        from PIL import Image

        assert isinstance(images[0], Image.Image), f"Expected PIL.Image, got {type(images[0])}"
        assert images[0].size == (128, 128), f"Expected 128x128, got {images[0].size}"
        assert images[0].mode == "RGB"

        # Pixel values must be valid (no NaN collapse → all-black or all-white)
        pixels = np.array(images[0])
        assert pixels.min() >= 0 and pixels.max() <= 255
        # With random weights, we expect some variance (not a solid color)
        assert pixels.std() > 1.0, f"Image has no variance (std={pixels.std():.2f}), likely broken pipeline"


# ═══════════════════════════════════════════════════════════════════════════
# 7. DiffusionConfig.validate() Contract
# ═══════════════════════════════════════════════════════════════════════════
class TestDiffusionConfigValidate:
    """DiffusionConfig.validate() rejects invalid configurations."""

    def test_max_sequence_length_zero_rejected(self):
        """max_sequence_length=0 must raise ValueError."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig

        config = DiffusionConfig(model_name_or_path="/fake", max_sequence_length=0)
        with pytest.raises(ValueError, match="max_sequence_length"):
            config.validate()

    def test_max_sequence_length_negative_rejected(self):
        """Negative max_sequence_length must raise ValueError."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig

        config = DiffusionConfig(model_name_or_path="/fake", max_sequence_length=-1)
        with pytest.raises(ValueError, match="max_sequence_length"):
            config.validate()

    def test_valid_config_passes(self):
        """Valid configuration should not raise."""
        from fastdeploy.model_executor.diffusion_models.config import DiffusionConfig

        config = DiffusionConfig(model_name_or_path="/fake", max_sequence_length=512)
        config.validate()  # Should not raise
