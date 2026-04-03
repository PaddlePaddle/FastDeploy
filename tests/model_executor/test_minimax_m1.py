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
Tests for MiniMax-M1 model scaffold and Lightning Attention reference.
Validates architecture dispatch, slope construction, registration, forward paths,
and Lightning Attention correctness via a pure-Python/NumPy reference implementation.

Uses importlib to load minimax_m1.py directly, bypassing fastdeploy/__init__.py
which pulls in the full inference engine (etcd, Redis, GPU ops, etc.).
All heavy submodules are replaced with lightweight stubs so tests run on CPU.
"""

import importlib
import importlib.util
import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import paddle
import pytest

# ---------------------------------------------------------------------------
# Module-level setup: load minimax_m1 with stub dependencies
# ---------------------------------------------------------------------------

# 1) paddleformers stubs
_PretrainedModel = type("PretrainedModel", (), {})


class _PretrainedConfig:
    prefix_name = ""

    @classmethod
    def get_config_dict(cls, model_path, **kw):
        import json as _j
        import os as _o

        with open(_o.path.join(model_path, "config.json")) as f:
            return _j.load(f), {}

    @classmethod
    def from_dict(cls, d):
        ns = SimpleNamespace(**d)
        ns.prefix_name = ""
        return ns


_cfg_mod = MagicMock()
_cfg_mod.PretrainedConfig = _PretrainedConfig
_transf = MagicMock()
_transf.PretrainedModel = _PretrainedModel
_transf.configuration_utils = _cfg_mod
_transf.PretrainedConfig = _PretrainedConfig

sys.modules.setdefault("paddleformers", MagicMock())
sys.modules["paddleformers.transformers"] = _transf
sys.modules["paddleformers.transformers.configuration_utils"] = _cfg_mod
sys.modules.setdefault("paddleformers.utils", MagicMock())
sys.modules["paddleformers.utils.log"] = MagicMock()

# 2) Lightweight fastdeploy namespace (bypass __init__.py)
_fd_ns = type(sys)("fastdeploy")
_fd_ns.__path__ = ["fastdeploy"]
_fd_ns.__file__ = "fastdeploy/__init__.py"
sys.modules["fastdeploy"] = _fd_ns

for _pkg, _path in [
    ("fastdeploy.model_executor", "fastdeploy/model_executor"),
    ("fastdeploy.model_executor.models", "fastdeploy/model_executor/models"),
]:
    _m = type(sys)(_pkg)
    _m.__path__ = [_path]
    sys.modules[_pkg] = _m

# 3) Mock all heavy fastdeploy submodules
for _mod_name in [
    "fastdeploy.config",
    "fastdeploy.distributed",
    "fastdeploy.distributed.communication",
    "fastdeploy.model_executor.forward_meta",
    "fastdeploy.model_executor.graph_optimization",
    "fastdeploy.model_executor.graph_optimization.decorator",
    "fastdeploy.model_executor.layers",
    "fastdeploy.model_executor.layers.activation",
    "fastdeploy.model_executor.layers.attention",
    "fastdeploy.model_executor.layers.attention.attention",
    "fastdeploy.model_executor.layers.embeddings",
    "fastdeploy.model_executor.layers.linear",
    "fastdeploy.model_executor.layers.lm_head",
    "fastdeploy.model_executor.layers.moe",
    "fastdeploy.model_executor.layers.moe.moe",
    "fastdeploy.model_executor.layers.normalization",
    "fastdeploy.model_executor.models.model_base",
    "fastdeploy.model_executor.ops",
    "fastdeploy.model_executor.ops.triton_ops",
    "fastdeploy.model_executor.ops.triton_ops.lightning_attn",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()


# 4) Real ModelRegistry so @register_model_class works
class _ModelCategory:
    TEXT_GENERATION = "text_generation"


class _ModelRegistry:
    _arch_to_model_cls = {}
    _enhanced_models = {}

    @classmethod
    def register_model_class(cls, model_class=None, **kw):
        def _register(mc):
            arch = kw.get("architecture", mc.name())
            cls._arch_to_model_cls[arch] = mc
            return mc

        return _register(model_class) if model_class is not None else _register


_ModelForCasualLM = type("ModelForCasualLM", (), {"name": classmethod(lambda cls: "base")})

_mb = sys.modules["fastdeploy.model_executor.models.model_base"]
_mb.ModelCategory = _ModelCategory
_mb.ModelRegistry = _ModelRegistry
_mb.ModelForCasualLM = _ModelForCasualLM

# support_graph_optimization → identity
sys.modules["fastdeploy.model_executor.graph_optimization.decorator"].support_graph_optimization = lambda cls: cls

# 5) Load minimax_m1.py via importlib
_spec = importlib.util.spec_from_file_location(
    "fastdeploy.model_executor.models.minimax_m1",
    "fastdeploy/model_executor/models/minimax_m1.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

# Import symbols
MiniMaxM1DecoderLayer = _mod.MiniMaxM1DecoderLayer
MiniMaxM1LinearAttention = _mod.MiniMaxM1LinearAttention
MiniMaxM1Attention = _mod.MiniMaxM1Attention
MiniMaxM1MoE = _mod.MiniMaxM1MoE
MiniMaxM1MLP = _mod.MiniMaxM1MLP
MiniMaxM1ForCausalLM = _mod.MiniMaxM1ForCausalLM
MiniMaxM1PretrainedModel = _mod.MiniMaxM1PretrainedModel
MiniMaxM1Model = _mod.MiniMaxM1Model
ModelRegistry = _ModelRegistry


# ===================================================================
# 1. Pure-logic tests
# ===================================================================


class TestBuildAttnTypeList:

    def test_80_layers_has_10_full_attention(self):
        attn_list = MiniMaxM1DecoderLayer._build_attn_type_list(80)
        assert len(attn_list) == 80
        full_indices = [i for i, t in enumerate(attn_list) if t == 1]
        assert full_indices == [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]

    def test_short_model_clips_indices(self):
        attn_list = MiniMaxM1DecoderLayer._build_attn_type_list(10)
        assert len(attn_list) == 10
        assert attn_list[7] == 1
        assert sum(attn_list) == 1

    def test_single_layer_all_linear(self):
        assert MiniMaxM1DecoderLayer._build_attn_type_list(1) == [0]

    def test_all_linear_below_first_full_index(self):
        assert all(t == 0 for t in MiniMaxM1DecoderLayer._build_attn_type_list(7))


class TestBuildSlopeTensor:

    def test_power_of_two_heads(self):
        slopes = MiniMaxM1LinearAttention._build_slope_tensor(8)
        assert slopes.shape == [8, 1, 1]
        assert (slopes.flatten().numpy() > 0).all()

    def test_non_power_of_two_heads(self):
        slopes = MiniMaxM1LinearAttention._build_slope_tensor(12)
        assert slopes.shape == [12, 1, 1]
        assert (slopes.flatten().numpy() > 0).all()

    def test_64_heads_first_slope(self):
        slopes = MiniMaxM1LinearAttention._build_slope_tensor(64)
        assert slopes.shape == [64, 1, 1]
        expected_start = 2 ** (-(2 ** (-(math.log2(64) - 3))))
        np.testing.assert_allclose(slopes.flatten().numpy()[0], expected_start, rtol=1e-5)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
    def test_slopes_all_positive(self, n):
        slopes = MiniMaxM1LinearAttention._build_slope_tensor(n)
        assert (slopes.flatten().numpy() > 0).all()


# ===================================================================
# 2. Model registration
# ===================================================================


class TestModelRegistration:

    def test_primary_architecture_registered(self):
        assert "MiniMaxM1ForCausalLM" in ModelRegistry._arch_to_model_cls

    def test_alias_architecture_registered(self):
        assert "MiniMaxText01ForCausalLM" in ModelRegistry._arch_to_model_cls

    def test_registered_class(self):
        assert ModelRegistry._arch_to_model_cls["MiniMaxM1ForCausalLM"] is MiniMaxM1ForCausalLM

    def test_name_method(self):
        assert MiniMaxM1ForCausalLM.name() == "MiniMaxM1ForCausalLM"

    def test_pretrained_name(self):
        assert MiniMaxM1PretrainedModel.arch_name() == "MiniMaxM1ForCausalLM"
        assert MiniMaxM1PretrainedModel.name() == "MiniMaxM1ForCausalLM"


# ===================================================================
# 3. Layer construction (lightweight fd_config mock)
# ===================================================================


def _make_fd_config(num_layers=4, attn_type_list=None, num_local_experts=4):
    if attn_type_list is None:
        attn_type_list = [0, 0, 0, 1][:num_layers]
    mc = SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        num_local_experts=num_local_experts,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        postnorm=False,
        attn_type_list=attn_type_list,
        layernorm_full_attention_alpha=3.556,
        layernorm_full_attention_beta=1.0,
        layernorm_linear_attention_alpha=3.556,
        layernorm_linear_attention_beta=1.0,
        layernorm_mlp_alpha=3.556,
        layernorm_mlp_beta=1.0,
        pretrained_config=SimpleNamespace(prefix_name="model"),
    )
    pc = SimpleNamespace(tensor_parallel_size=1, tensor_parallel_rank=0, tp_group=None)
    return SimpleNamespace(model_config=mc, parallel_config=pc)


class TestDecoderLayerConstruction:

    def test_linear_attention_layer(self):
        fd = _make_fd_config()
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert layer.attention_type == 0
        assert isinstance(layer.self_attn, MiniMaxM1LinearAttention)
        assert hasattr(layer.self_attn, "slope_rate")
        assert hasattr(layer.self_attn, "output_gate")
        assert hasattr(layer.self_attn, "norm")
        assert hasattr(layer.self_attn, "out_proj")

    def test_full_attention_layer(self):
        fd = _make_fd_config()
        layer = MiniMaxM1DecoderLayer(fd, layer_id=3, prefix="model.layers.3")
        assert layer.attention_type == 1
        assert isinstance(layer.self_attn, MiniMaxM1Attention)

    def test_deepnorm_defaults(self):
        fd = _make_fd_config()
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert layer.layernorm_attention_alpha == 3.556
        assert layer.layernorm_mlp_alpha == 3.556

    def test_moe_when_experts_gt_1(self):
        fd = _make_fd_config(num_local_experts=4)
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert isinstance(layer.block_sparse_moe, MiniMaxM1MoE)

    def test_dense_mlp_when_single_expert(self):
        fd = _make_fd_config(num_local_experts=1)
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert isinstance(layer.block_sparse_moe, MiniMaxM1MLP)

    def test_fallback_attn_type_when_no_config(self):
        fd = _make_fd_config(num_layers=80)
        delattr(fd.model_config, "attn_type_list")
        layer = MiniMaxM1DecoderLayer(fd, layer_id=7, prefix="model.layers.7")
        assert layer.attention_type == 1

    def test_moe_default_weight_key_map(self):
        """Unquantized config → weight_key_map has plain .weight keys."""
        fd = _make_fd_config(num_local_experts=4)
        FusedMoE = sys.modules["fastdeploy.model_executor.layers.moe.moe"].FusedMoE
        FusedMoE.reset_mock()
        MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
        wkm = FusedMoE.call_args[1]["weight_key_map"]
        assert "gate_weight_key" in wkm
        assert wkm["up_gate_proj_expert_weight_key"].endswith(".up_gate_proj.weight")
        assert "weight_scale" not in str(wkm)

    def test_moe_w4a8_weight_key_map(self):
        """w4a8 quant config → weight_key_map has .quant_weight + scales."""
        fd = _make_fd_config(num_local_experts=4)
        fd.quant_config = SimpleNamespace(moe_quant_type="w4a8")
        fd.model_config.is_quantized = True
        FusedMoE = sys.modules["fastdeploy.model_executor.layers.moe.moe"].FusedMoE
        FusedMoE.reset_mock()
        MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
        wkm = FusedMoE.call_args[1]["weight_key_map"]
        assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
        assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
        assert "activation_scale" in wkm["up_gate_proj_expert_in_scale_key"]

    def test_moe_w4afp8_dynamic_weight_key_map(self):
        """Dynamic w4afp8 → quant_weight + weight_scale but no activation_scale."""
        fd = _make_fd_config(num_local_experts=4)
        fd.quant_config = SimpleNamespace(moe_quant_type="w4afp8", moe_dynamic_quant=True)
        fd.model_config.is_quantized = True
        FusedMoE = sys.modules["fastdeploy.model_executor.layers.moe.moe"].FusedMoE
        FusedMoE.reset_mock()
        MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
        wkm = FusedMoE.call_args[1]["weight_key_map"]
        assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
        assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
        assert "in_scale_key" not in str(wkm)


# ===================================================================
# 4. Forward-pass smoke tests
# ===================================================================


class TestDecoderLayerForward:

    @staticmethod
    def _patch_layer(layer, hidden_size):
        def _norm_fn(x, residual_input=None, forward_meta=None):
            if residual_input is None:
                residual_input = paddle.zeros_like(x)
            return x, residual_input + x

        object.__setattr__(layer, "input_layernorm", MagicMock(side_effect=_norm_fn))
        object.__setattr__(layer, "post_attention_layernorm", MagicMock(side_effect=_norm_fn))
        object.__setattr__(layer, "self_attn", MagicMock(return_value=paddle.randn([4, hidden_size])))
        object.__setattr__(layer, "block_sparse_moe", MagicMock(return_value=paddle.randn([4, hidden_size])))

    def test_linear_layer_returns_tuple(self):
        fd = _make_fd_config()
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        self._patch_layer(layer, 256)

        out = layer(forward_meta=SimpleNamespace(), hidden_states=paddle.randn([4, 256]))
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == [4, 256]

    def test_full_attn_layer_returns_tuple(self):
        fd = _make_fd_config()
        layer = MiniMaxM1DecoderLayer(fd, layer_id=3, prefix="model.layers.3")
        self._patch_layer(layer, 256)

        out = layer(forward_meta=SimpleNamespace(), hidden_states=paddle.randn([4, 256]))
        assert isinstance(out, tuple) and len(out) == 2

    def test_deepnorm_scaling_applied(self):
        fd = _make_fd_config()
        fd.model_config.layernorm_linear_attention_alpha = 2.0
        fd.model_config.layernorm_mlp_alpha = 3.0
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert layer.layernorm_attention_alpha == 2.0
        assert layer.layernorm_mlp_alpha == 3.0

    def test_postnorm_forward(self):
        fd = _make_fd_config()
        fd.model_config.postnorm = True
        layer = MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        self._patch_layer(layer, 256)

        out = layer(forward_meta=SimpleNamespace(), hidden_states=paddle.randn([4, 256]))
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == [4, 256]


# ===================================================================
# 5. Lightning Attention — Pure-Python reference
# ===================================================================


def _lightning_attention_numpy_ref(q, k, v, slope, kv_history=None):
    """
    Pure NumPy reference implementation of linear attention with exponential decay.

    Args:
        q: [batch, heads, seq_len, dim]
        k: [batch, heads, seq_len, dim]
        v: [batch, heads, seq_len, dim_v]
        slope: [heads] decay rates
        kv_history: [batch, heads, dim, dim_v] or None

    Returns:
        output: [batch, heads, seq_len, dim_v]
        kv_state: [batch, heads, dim, dim_v] updated state
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    output = np.zeros((b, h, n, e), dtype=np.float64)

    if kv_history is None:
        kv_state = np.zeros((b, h, d, e), dtype=np.float64)
    else:
        kv_state = kv_history.copy()

    for t in range(n):
        # Decay factor: exp(-slope) broadcast to [b, h, 1, 1] for kv_state
        decay = np.exp(-slope)[np.newaxis, :, np.newaxis, np.newaxis]  # [1, h, 1, 1]
        kv_state = kv_state * decay
        # Add new key-value outer product: k[t] ⊗ v[t]
        kt = k[:, :, t, :]  # [b, h, d]
        vt = v[:, :, t, :]  # [b, h, e]
        kv_state += kt[:, :, :, np.newaxis] * vt[:, :, np.newaxis, :]
        # Query the state
        qt = q[:, :, t, :]  # [b, h, d]
        output[:, :, t, :] = np.einsum("bhd,bhde->bhe", qt, kv_state)

    return output, kv_state


class TestLightningAttentionPurePython:
    """Validate Lightning Attention algorithm correctness via NumPy reference."""

    def test_single_token_output_shape(self):
        """Single token: output shape must match [b, h, 1, e]."""
        b, h, n, d = 1, 4, 1, 16
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.abs(np.random.randn(h)) * 0.1

        output, kv = _lightning_attention_numpy_ref(q, k, v, slope)
        assert output.shape == (b, h, n, d)
        assert kv.shape == (b, h, d, d)

    def test_multi_token_causal(self):
        """With slope → 0, approaches standard dot-product attention (cumulative)."""
        b, h, n, d = 1, 2, 4, 8
        np.random.seed(42)
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.full(h, 1e-8)

        output, _ = _lightning_attention_numpy_ref(q, k, v, slope)

        # With near-zero decay, position 0 sees only token 0,
        # position 1 sees tokens 0+1, etc.  (causal linear attention)
        for t in range(n):
            # Reference: sum of q[t] @ (k[j] ⊗ v[j]) for j=0..t
            ref = np.zeros((b, h, d))
            for j in range(t + 1):
                kv_outer = k[:, :, j, :, np.newaxis] * v[:, :, j, np.newaxis, :]
                ref += np.einsum("bhd,bhde->bhe", q[:, :, t, :], kv_outer)
            np.testing.assert_allclose(output[:, :, t, :], ref, rtol=1e-5, atol=1e-7)

    def test_kv_history_persistence(self):
        """KV state from one call persists to the next (recurrent property)."""
        b, h, n, d = 2, 4, 3, 16
        np.random.seed(123)
        q1 = np.random.randn(b, h, n, d)
        k1 = np.random.randn(b, h, n, d)
        v1 = np.random.randn(b, h, n, d)
        q2 = np.random.randn(b, h, 1, d)
        k2 = np.random.randn(b, h, 1, d)
        v2 = np.random.randn(b, h, 1, d)
        slope = np.abs(np.random.randn(h)) * 0.05

        # Two-step: process seq1, then continue with seq2 using returned state
        _, kv_after_1 = _lightning_attention_numpy_ref(q1, k1, v1, slope)
        out2, _ = _lightning_attention_numpy_ref(q2, k2, v2, slope, kv_history=kv_after_1)

        # One-shot: process full concatenated sequence
        q_full = np.concatenate([q1, q2], axis=2)
        k_full = np.concatenate([k1, k2], axis=2)
        v_full = np.concatenate([v1, v2], axis=2)
        out_full, _ = _lightning_attention_numpy_ref(q_full, k_full, v_full, slope)

        # The last token output should match
        np.testing.assert_allclose(out2[:, :, 0, :], out_full[:, :, n, :], rtol=1e-5, atol=1e-7)

    def test_multi_head_independent(self):
        """Heads are computed independently — zeroing one head's Q zeros its output."""
        b, h, n, d = 1, 8, 4, 16
        np.random.seed(7)
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.abs(np.random.randn(h)) * 0.1

        # Zero out head 3's query
        q_masked = q.copy()
        q_masked[:, 3, :, :] = 0.0

        output, _ = _lightning_attention_numpy_ref(q_masked, k, v, slope)
        np.testing.assert_allclose(output[:, 3, :, :], 0.0, atol=1e-12)
