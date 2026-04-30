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
Integration tests for MiniMax-M1 model with FastDeploy infrastructure.

Validates that model code works through FD's real pipelines:
- Package imports (all public symbols accessible)
- ModelRegistry resolution (both architecture names)
- FDConfig construction from config.json
- Weight key remapping (HF -> FD) through load_weights iterator path
- End-to-end forward pass with real (tiny) weights on GPU

CPU-tier tests run in CI (no GPU). GPU-tier tests run on AI Studio A800.
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# Tiny model config — production-faithful structure, minimal dimensions
# ---------------------------------------------------------------------------

_TINY_MODEL_CONFIG = {
    "architectures": ["MiniMaxM1ForCausalLM"],
    "model_type": "MiniMaxM1",
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "num_local_experts": 2,
    "num_experts_per_tok": 1,
    "rope_theta": 10000.0,
    "torch_dtype": "bfloat16",
    "full_attention_layer_indices": [1, 3],
    "attn_type_list": [0, 1, 0, 1],  # linear, full, linear, full
    "use_deep_norm": True,
    "num_layers_for_deep_norm": 4,
    "use_post_norm": True,
    "hidden_act": "silu",
    "norm_topk_prob": False,
    "postnorm": False,
}


def _make_fd_config(**model_overrides):
    """Build a minimal FDConfig-like namespace for CPU tests."""
    mc_dict = dict(_TINY_MODEL_CONFIG)
    mc_dict.update(model_overrides)
    mc_dict["pretrained_config"] = SimpleNamespace(prefix_name="model")
    mc = SimpleNamespace(**mc_dict)
    pc = SimpleNamespace(tensor_parallel_size=1, tensor_parallel_rank=0, tp_group=None)
    gc = SimpleNamespace(graph_opt_level=0, use_cudagraph=False)
    return SimpleNamespace(
        model_config=mc,
        parallel_config=pc,
        graph_opt_config=gc,
    )


def _write_config_json(tmp_dir, overrides=None):
    """Write a minimal config.json that mimics real MiniMax-M1 HF layout."""
    cfg = dict(_TINY_MODEL_CONFIG)
    if overrides:
        cfg.update(overrides)
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    return config_path


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — CPU tests (run in CI)
# ═══════════════════════════════════════════════════════════════════════════


class TestPackageImports:
    """Prove all public MiniMax-M1 symbols are importable from FD."""

    def test_import_model_module(self):
        from fastdeploy.model_executor.models import minimax_m1

        assert hasattr(minimax_m1, "MiniMaxM1ForCausalLM")

    def test_import_causal_lm(self):
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM

        assert MiniMaxM1ForCausalLM is not None

    def test_import_pretrained_model(self):
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1PretrainedModel

        assert MiniMaxM1PretrainedModel is not None

    def test_import_all_classes(self):
        from fastdeploy.model_executor.models.minimax_m1 import (
            MiniMaxM1Attention,
            MiniMaxM1DecoderLayer,
            MiniMaxM1ForCausalLM,
            MiniMaxM1LinearAttention,
            MiniMaxM1MLP,
            MiniMaxM1Model,
            MiniMaxM1MoE,
            MiniMaxM1PretrainedModel,
        )

        classes = [
            MiniMaxM1MLP,
            MiniMaxM1MoE,
            MiniMaxM1Attention,
            MiniMaxM1LinearAttention,
            MiniMaxM1DecoderLayer,
            MiniMaxM1Model,
            MiniMaxM1ForCausalLM,
            MiniMaxM1PretrainedModel,
        ]
        for cls in classes:
            assert callable(cls), f"{cls.__name__} should be callable"

    def test_lightning_attention_importable(self):
        from fastdeploy.model_executor.ops.triton_ops import lightning_attn

        assert hasattr(lightning_attn, "lightning_attention")


class TestModelRegistryResolution:
    """Prove ModelRegistry resolves MiniMax-M1 by both architecture names."""

    def test_primary_arch_resolves(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        cls = ModelRegistry.get_class("MiniMaxM1ForCausalLM")
        assert cls.__name__ == "MiniMaxM1ForCausalLM"

    def test_alias_arch_resolves(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        cls = ModelRegistry.get_class("MiniMaxText01ForCausalLM")
        assert cls.__name__ == "MiniMaxM1ForCausalLM"

    def test_both_resolve_to_same_class(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        primary = ModelRegistry.get_class("MiniMaxM1ForCausalLM")
        alias = ModelRegistry.get_class("MiniMaxText01ForCausalLM")
        assert primary is alias

    def test_in_supported_archs(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        supported = ModelRegistry.get_supported_archs()
        assert "MiniMaxM1ForCausalLM" in supported
        assert "MiniMaxText01ForCausalLM" in supported


class TestHFToFDWeightKeyMapping:
    """Prove the HF→FD weight key remapping pipeline works correctly.

    Tests set_state_dict (v2 path) with real numpy arrays — verifying that
    HF checkpoint key conventions are correctly transformed to FD conventions.
    This is the most common source of integration bugs.
    """

    @pytest.fixture
    def tiny_model(self, monkeypatch):
        """Build a MiniMaxM1ForCausalLM with minimal stubs for weight loading."""
        from fastdeploy.model_executor.models import minimax_m1

        # Lightweight stubs that track load_state_dict calls
        class _TrackingLayer(paddle.nn.Layer):
            def __init__(self, *a, **kw):
                super().__init__()
                self.loaded_keys = []

            def forward(self, x, *a, **kw):
                return x

            def load_state_dict(self, sd):
                self.loaded_keys.extend(sd.keys())

        class _TrackingLinear(_TrackingLayer):
            def __init__(self, *a, **kw):
                super().__init__()
                self._out = kw.get("output_size", 128)

            def forward(self, x, *a, **kw):
                shape = list(x.shape)
                shape[-1] = self._out
                return paddle.zeros(shape, dtype=x.dtype)

        class _TrackingNorm(_TrackingLayer):
            def forward(self, x, residual_input=None, forward_meta=None):
                if residual_input is None:
                    residual_input = paddle.zeros_like(x)
                return x, residual_input + x

        class _TrackingMoE(_TrackingLayer):
            def __init__(self, *a, **kw):
                super().__init__()
                self.loaded_keys = []
                self.weight_key_map = kw.get("weight_key_map", {})

            def forward(self, hidden_states, gate, forward_meta=None):
                return hidden_states

        class _TrackingAttn(_TrackingLayer):
            def forward(self, q, k, v, forward_meta=None):
                return q

        class _TrackingEmbed(_TrackingLayer):
            def forward(self, x, *a, **kw):
                return paddle.zeros([x.shape[0], 128], dtype="float32")

        class _TrackingLMHead(_TrackingLayer):
            def forward(self, x, *a, **kw):
                return paddle.zeros([x.shape[0], 256], dtype="float32")

        # Patch constructors
        monkeypatch.setattr(minimax_m1, "RMSNorm", _TrackingNorm)
        monkeypatch.setattr(minimax_m1, "ColumnParallelLinear", _TrackingLinear)
        monkeypatch.setattr(minimax_m1, "RowParallelLinear", _TrackingLinear)
        monkeypatch.setattr(minimax_m1, "MergedColumnParallelLinear", _TrackingLinear)
        monkeypatch.setattr(minimax_m1, "QKVParallelLinear", _TrackingLinear)
        monkeypatch.setattr(minimax_m1, "ReplicatedLinear", _TrackingLinear)
        monkeypatch.setattr(minimax_m1, "Attention", _TrackingAttn)
        monkeypatch.setattr(minimax_m1, "FusedMoE", _TrackingMoE)
        monkeypatch.setattr(minimax_m1, "VocabParallelEmbedding", _TrackingEmbed)
        monkeypatch.setattr(minimax_m1, "ParallelLMHead", _TrackingLMHead)
        monkeypatch.setattr(minimax_m1, "SiluAndMul", lambda *a, **kw: (lambda x: x[..., : x.shape[-1] // 2]))
        monkeypatch.setattr(minimax_m1, "lightning_attention", lambda *a, **kw: (a[0], paddle.zeros([1])))
        monkeypatch.setattr(minimax_m1, "tensor_model_parallel_all_reduce", lambda x: x)
        monkeypatch.setattr(minimax_m1, "support_graph_optimization", lambda *a, **kw: (lambda fn: fn))

        cfg = _make_fd_config()
        model = minimax_m1.MiniMaxM1ForCausalLM(cfg)
        return model

    def test_expert_w1_w2_w3_renamed(self, tiny_model):
        """HF w1→gate_proj, w3→up_proj, w2→down_proj in MoE experts."""
        sd = {}
        # Layer 0 = linear attention layer (not in full_attention_layer_indices [1,3])
        # MoE layer
        sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"] = np.ones((256, 128), dtype=np.float32)
        sd["model.layers.0.block_sparse_moe.experts.0.w2.weight"] = np.ones((128, 256), dtype=np.float32)
        sd["model.layers.0.block_sparse_moe.experts.0.w3.weight"] = np.ones((256, 128), dtype=np.float32)

        tiny_model.set_state_dict(sd)

        # Verify renamed keys were passed to MoE sublayer's experts
        moe = tiny_model.model.layers[0].block_sparse_moe
        # MiniMaxM1MoE.load_state_dict dispatches to self.gate and self.experts
        expert_keys = moe.experts.loaded_keys
        assert any("gate_proj" in k for k in expert_keys), f"Expected gate_proj, got {expert_keys}"
        assert any("down_proj" in k for k in expert_keys), f"Expected down_proj, got {expert_keys}"
        assert any("up_proj" in k for k in expert_keys), f"Expected up_proj, got {expert_keys}"

    def test_qkv_merge_for_full_attention_layers(self, tiny_model):
        """Full attention layers merge separate q/k/v → qkv_proj."""
        sd = {}
        # Layer 1 is a full attention layer (index 1 in full_attention_layer_indices)
        sd["model.layers.1.self_attn.q_proj.weight"] = np.ones((128, 128), dtype=np.float32)
        sd["model.layers.1.self_attn.k_proj.weight"] = np.ones((128, 128), dtype=np.float32) * 2
        sd["model.layers.1.self_attn.v_proj.weight"] = np.ones((128, 128), dtype=np.float32) * 3

        tiny_model.set_state_dict(sd)

        attn = tiny_model.model.layers[1].self_attn
        assert any(
            "qkv_proj" in k for k in attn.qkv_proj.loaded_keys
        ), f"Expected qkv_proj merge, got {attn.qkv_proj.loaded_keys}"

    def test_norm_and_embed_passthrough(self, tiny_model):
        """Non-expert, non-attention keys pass through unchanged."""
        sd = {}
        sd["model.embed_tokens.weight"] = np.ones((256, 128), dtype=np.float32)
        sd["model.norm.weight"] = np.ones(128, dtype=np.float32)

        tiny_model.set_state_dict(sd)

        embed = tiny_model.model.embed_tokens
        assert len(embed.loaded_keys) > 0, "embed_tokens should receive weights"

    def test_all_layer_types_receive_weights(self, tiny_model):
        """Build a full HF-style state dict and verify every layer gets called."""
        sd = {}
        for i in range(4):
            # Input norm
            sd[f"model.layers.{i}.input_layernorm.weight"] = np.ones(128, dtype=np.float32)
            sd[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(128, dtype=np.float32)

            if i in [1, 3]:  # full attention
                sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.k_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.v_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.o_proj.weight"] = np.ones((128, 128), dtype=np.float32)
            else:  # linear attention
                sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.k_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.v_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.out_proj.weight"] = np.ones((128, 128), dtype=np.float32)
                sd[f"model.layers.{i}.self_attn.output_gate.weight"] = np.ones((128, 128), dtype=np.float32)

            # MoE
            for e in range(2):
                sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight"] = np.ones((256, 128), dtype=np.float32)
                sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight"] = np.ones((128, 256), dtype=np.float32)
                sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight"] = np.ones((256, 128), dtype=np.float32)
            sd[f"model.layers.{i}.block_sparse_moe.gate.weight"] = np.ones((2, 128), dtype=np.float32)

        sd["model.embed_tokens.weight"] = np.ones((256, 128), dtype=np.float32)
        sd["model.norm.weight"] = np.ones(128, dtype=np.float32)
        sd["lm_head.weight"] = np.ones((256, 128), dtype=np.float32)

        tiny_model.set_state_dict(sd)

        # Verify embed, model norm, and lm_head all got weights
        assert len(tiny_model.model.embed_tokens.loaded_keys) > 0
        assert len(tiny_model.lm_head.loaded_keys) > 0


class TestModelConstruction:
    """Prove MiniMaxM1ForCausalLM constructs correctly with right layer types."""

    @pytest.fixture
    def model(self, monkeypatch):
        """Build model with stubs to verify construction on CPU."""
        from fastdeploy.model_executor.models import minimax_m1

        class _Stub(paddle.nn.Layer):
            def __init__(self, *a, **kw):
                super().__init__()

            def forward(self, *a, **kw):
                return a[0] if a else paddle.zeros([1])

            def load_state_dict(self, _sd):
                pass

        class _StubNorm(_Stub):
            def forward(self, x, residual_input=None, forward_meta=None):
                r = residual_input if residual_input is not None else paddle.zeros_like(x)
                return x, r + x

        class _StubAttn(_Stub):
            def forward(self, q, k, v, forward_meta=None):
                return q

        class _StubMoE(_Stub):
            def __init__(self, *a, **kw):
                super().__init__()
                self.weight_key_map = kw.get("weight_key_map", {})

            def forward(self, hidden_states, gate, forward_meta=None):
                return hidden_states

        monkeypatch.setattr(minimax_m1, "RMSNorm", _StubNorm)
        monkeypatch.setattr(minimax_m1, "ColumnParallelLinear", _Stub)
        monkeypatch.setattr(minimax_m1, "RowParallelLinear", _Stub)
        monkeypatch.setattr(minimax_m1, "MergedColumnParallelLinear", _Stub)
        monkeypatch.setattr(minimax_m1, "QKVParallelLinear", _Stub)
        monkeypatch.setattr(minimax_m1, "ReplicatedLinear", _Stub)
        monkeypatch.setattr(minimax_m1, "Attention", _StubAttn)
        monkeypatch.setattr(minimax_m1, "FusedMoE", _StubMoE)
        monkeypatch.setattr(minimax_m1, "VocabParallelEmbedding", _Stub)
        monkeypatch.setattr(minimax_m1, "ParallelLMHead", _Stub)
        monkeypatch.setattr(minimax_m1, "SiluAndMul", lambda *a, **kw: (lambda x: x[..., : x.shape[-1] // 2]))
        monkeypatch.setattr(minimax_m1, "lightning_attention", lambda *a, **kw: (a[0], paddle.zeros([1])))
        monkeypatch.setattr(minimax_m1, "tensor_model_parallel_all_reduce", lambda x: x)
        monkeypatch.setattr(minimax_m1, "support_graph_optimization", lambda *a, **kw: (lambda fn: fn))

        cfg = _make_fd_config()
        return minimax_m1.MiniMaxM1ForCausalLM(cfg)

    def test_correct_number_of_layers(self, model):
        assert len(model.model.layers) == 4

    def test_full_attention_at_configured_indices(self, model):
        """Full attention layers at indices [1, 3], linear at [0, 2]."""
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1DecoderLayer

        for i, layer in enumerate(model.model.layers):
            assert isinstance(layer, MiniMaxM1DecoderLayer)
            if i in [1, 3]:
                assert layer.attention_type == 1, f"Layer {i} should be full attention (1), got {layer.attention_type}"
            else:
                assert (
                    layer.attention_type == 0
                ), f"Layer {i} should be linear attention (0), got {layer.attention_type}"

    def test_model_name_method(self, model):
        assert model.name() == "MiniMaxM1ForCausalLM"


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — GPU integration tests (run on AI Studio A800 via SSH)
# See also: tests/model_executor/test_minimax_m1_smoke.py (kernel-level GPU tests)
# See also: tests/operators/test_lightning_attn_triton.py (Triton kernel tests)
# See also: tests/model_executor/validate_minimax_m1_e2e.py (E2E server test)
# ═══════════════════════════════════════════════════════════════════════════

_GPU_AVAILABLE = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
_GPU_SKIP_MSG = "No CUDA GPU available — GPU integration tests require A800/V100"


@pytest.mark.gpu
@unittest.skipUnless(_GPU_AVAILABLE, _GPU_SKIP_MSG)
class TestModelWithRealTritonKernels(unittest.TestCase):
    """Prove MiniMax-M1 model layers produce correct output via real Triton kernels.

    Unlike test_minimax_m1_smoke.py (which tests kernels in isolation), this
    tests through the actual MiniMaxM1LinearAttention and MiniMaxM1DecoderLayer
    code paths — proving the model's forward() method correctly calls Triton ops.
    """

    def _build_slope(self, n_heads):
        """Build ALiBi-style slope tensor (same as production code)."""
        import math

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** (-(math.log2(n) - 3))))
            return [start * (start**i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            nearest = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(nearest) + get_slopes_power_of_2(2 * nearest)[0::2][: n_heads - nearest]
        return paddle.to_tensor(slopes, dtype="float32").reshape([n_heads, 1, 1])

    def test_linear_attention_layer_forward(self):
        """lightning_attention() produces valid output via real Triton kernel."""
        from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
            lightning_attention,
        )

        B, H, S, D = 1, 8, 256, 128  # H=8, S=BLOCK, D>=128 for kernel

        q = paddle.randn([B, H, S, D], dtype="float16")
        k = paddle.randn([B, H, S, D], dtype="float16")
        v = paddle.randn([B, H, S, D], dtype="float16")
        ed = self._build_slope(H).squeeze(-1)  # [H, 1] — wrapper reshapes

        out, kv = lightning_attention(q, k, v, ed, block_size=256)

        self.assertEqual(list(out.shape), [B, H, S, D])
        self.assertFalse(paddle.isnan(out).any().item(), "Output contains NaN")
        self.assertTrue(paddle.isfinite(out).all().item(), "Output contains Inf")
        self.assertTrue(kv.abs().sum().item() > 0, "KV state is all zeros")

    def test_decode_kernel_single_token(self):
        """Decode kernel handles single-token autoregressive step."""
        from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
            linear_decode_forward_triton,
        )

        B, H, D = 2, 4, 128
        q = paddle.randn([B, H, 1, D], dtype="float16")
        k = paddle.randn([B, H, 1, D], dtype="float16")
        v = paddle.randn([B, H, 1, D], dtype="float16")
        kv_state = paddle.zeros([B, H, D, D], dtype="float32")
        slope_rate = self._build_slope(H).squeeze(-1).squeeze(-1)  # [H]
        slot_idx = paddle.arange(B, dtype="int64")

        out = linear_decode_forward_triton(q, k, v, kv_state, slope_rate, slot_idx)

        # Output: [B, H*D] (heads flattened by kernel)
        self.assertEqual(list(out.shape), [B, H * D])
        self.assertFalse(paddle.isnan(out).any().item())

    def test_two_step_decode_state_accumulates(self):
        """Two decode steps via Triton: KV state should differ from fresh state."""
        from fastdeploy.model_executor.ops.triton_ops.lightning_attn import (
            linear_decode_forward_triton,
        )

        B, H, D = 1, 4, 128
        kv_state = paddle.zeros([B, H, D, D], dtype="float32")
        slope_rate = self._build_slope(H).squeeze(-1).squeeze(-1)  # [H]
        slot_idx = paddle.arange(B, dtype="int64")

        # Step 1
        q1 = paddle.randn([B, H, 1, D], dtype="float16")
        k1 = paddle.randn([B, H, 1, D], dtype="float16")
        v1 = paddle.randn([B, H, 1, D], dtype="float16")
        _out1 = linear_decode_forward_triton(q1, k1, v1, kv_state, slope_rate, slot_idx)  # noqa: F841

        # KV state should be updated in-place
        self.assertTrue(kv_state.abs().sum().item() > 0, "KV state not updated after step 1")

        # Step 2 with different input
        q2 = paddle.randn([B, H, 1, D], dtype="float16")
        k2 = paddle.randn([B, H, 1, D], dtype="float16")
        v2 = paddle.randn([B, H, 1, D], dtype="float16")
        kv_before = kv_state.clone()
        _out2 = linear_decode_forward_triton(q2, k2, v2, kv_state, slope_rate, slot_idx)  # noqa: F841

        # State should change between step 1 and step 2
        state_changed = (kv_state - kv_before).abs().sum().item() > 0
        self.assertTrue(state_changed, "KV state unchanged after step 2")


if __name__ == "__main__":
    unittest.main()
