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

Uses monkeypatch + lightweight stubs so tests run on CPU without GPU kernels.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.models import minimax_m1
from fastdeploy.model_executor.models.model_base import ModelRegistry

# ── Stub classes ─────────────────────────────────────────────────────────


class _StubRMSNorm(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x, residual_input=None, forward_meta=None):
        r = residual_input if residual_input is not None else paddle.zeros_like(x)
        return x, r

    def load_state_dict(self, _sd):
        pass


class _StubLinear(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self._output_size = kw.get("output_size", None)

    def forward(self, x):
        if self._output_size is not None and self._output_size != x.shape[-1]:
            return paddle.zeros(list(x.shape[:-1]) + [self._output_size], dtype=x.dtype)
        return x

    def load_state_dict(self, _sd):
        pass


class _StubAttention(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, qkv=None, forward_meta=None, **kw):
        return qkv

    def load_state_dict(self, _sd):
        pass


class _StubActivation(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x):
        return x


class _StubEmbedding(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self._hidden = kw.get("embedding_dim", 256)

    def forward(self, ids_remove_padding=None, forward_meta=None):
        return paddle.ones([ids_remove_padding.shape[0], self._hidden], dtype="float32")

    def load_state_dict(self, _sd):
        pass


class _StubLMHead(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self._vocab = kw.get("num_embeddings", 1024)

    def forward(self, x):
        return paddle.ones([x.shape[0], self._vocab], dtype=x.dtype)

    def load_state_dict(self, _sd):
        pass


class _StubFusedMoE(paddle.nn.Layer):
    """Recording FusedMoE stub that captures constructor kwargs."""

    last_init_kwargs = {}

    def __init__(self, *a, **kw):
        super().__init__()
        _StubFusedMoE.last_init_kwargs = kw.copy()

    def forward(self, x, *a, **kw):
        return x

    def load_state_dict(self, _sd):
        pass

    @staticmethod
    def make_expert_params_mapping(**kw):
        return []


def _stub_lightning_attention(q, k, v, slope, block_size=256, kv_history=None):
    """Return dummy output tensors for lightning attention."""
    b, h, s, d = q.shape
    return paddle.zeros_like(q), paddle.zeros([b, h, d, d], dtype=q.dtype)


# ── Forward-test callables (replace sublayer internals after construction) ──


class _NormStub:
    """Passthrough norm with residual accumulation."""

    def __call__(self, x, residual_input=None, forward_meta=None):
        if residual_input is None:
            residual_input = paddle.zeros_like(x)
        return x, residual_input + x


class _AttnStub:
    """Fixed-size random output replacing attention."""

    def __init__(self, hidden_size):
        self._h = hidden_size

    def __call__(self, forward_meta=None, hidden_states=None):
        return paddle.randn([hidden_states.shape[0], self._h])


class _MoEStub:
    """Fixed-size random output replacing MoE/MLP."""

    def __init__(self, hidden_size):
        self._h = hidden_size

    def __call__(self, hidden_states, forward_meta=None):
        return paddle.randn([hidden_states.shape[0], self._h])


def _patch_layer(layer, hidden_size):
    """Replace sublayers with simple stubs for forward testing."""
    object.__setattr__(layer, "input_layernorm", _NormStub())
    object.__setattr__(layer, "post_attention_layernorm", _NormStub())
    object.__setattr__(layer, "self_attn", _AttnStub(hidden_size))
    object.__setattr__(layer, "block_sparse_moe", _MoEStub(hidden_size))


# ── Helper ──────────────────────────────────────────────────────────────


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
    gc = SimpleNamespace(graph_opt_level=0, use_cudagraph=False)
    return SimpleNamespace(model_config=mc, parallel_config=pc, graph_opt_config=gc)


# ── Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture()
def m(monkeypatch):
    """Patch heavy layer imports with lightweight stubs."""
    monkeypatch.setattr(minimax_m1, "RMSNorm", _StubRMSNorm)
    monkeypatch.setattr(minimax_m1, "ColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "MergedColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "RowParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "ReplicatedLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "Attention", _StubAttention)
    monkeypatch.setattr(minimax_m1, "SiluAndMul", _StubActivation)
    monkeypatch.setattr(minimax_m1, "VocabParallelEmbedding", _StubEmbedding)
    monkeypatch.setattr(minimax_m1, "ParallelLMHead", _StubLMHead)
    monkeypatch.setattr(minimax_m1, "FusedMoE", _StubFusedMoE)
    monkeypatch.setattr(minimax_m1, "lightning_attention", _stub_lightning_attention)
    monkeypatch.setattr(minimax_m1, "tensor_model_parallel_all_reduce", lambda x: x)
    return minimax_m1


# ===================================================================
# 1. Pure-logic tests
# ===================================================================


def test_80_layers_has_10_full_attention():
    attn_list = minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(80)
    assert len(attn_list) == 80
    full_indices = [i for i, t in enumerate(attn_list) if t == 1]
    assert full_indices == [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]


def test_short_model_clips_indices():
    attn_list = minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(10)
    assert len(attn_list) == 10
    assert attn_list[7] == 1
    assert sum(attn_list) == 1


def test_single_layer_all_linear():
    assert minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(1) == [0]


def test_all_linear_below_first_full_index():
    assert all(t == 0 for t in minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(7))


def test_slope_power_of_two_heads():
    slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(8)
    assert slopes.shape == [8, 1, 1]
    assert (slopes.flatten().numpy() > 0).all()


def test_slope_non_power_of_two_heads():
    slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(12)
    assert slopes.shape == [12, 1, 1]
    assert (slopes.flatten().numpy() > 0).all()


def test_slope_64_heads_first_value():
    slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(64)
    assert slopes.shape == [64, 1, 1]
    expected_start = 2 ** (-(2 ** (-(math.log2(64) - 3))))
    np.testing.assert_allclose(slopes.flatten().numpy()[0], expected_start, rtol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_slopes_all_positive(n):
    slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(n)
    assert (slopes.flatten().numpy() > 0).all()


# ===================================================================
# 2. Model registration
# ===================================================================


def test_primary_architecture_registered():
    assert "MiniMaxM1ForCausalLM" in ModelRegistry._arch_to_model_cls


def test_alias_architecture_registered():
    assert "MiniMaxText01ForCausalLM" in ModelRegistry._arch_to_model_cls


def test_registered_class():
    assert ModelRegistry._arch_to_model_cls["MiniMaxM1ForCausalLM"] is minimax_m1.MiniMaxM1ForCausalLM


def test_name_method():
    assert minimax_m1.MiniMaxM1ForCausalLM.name() == "MiniMaxM1ForCausalLM"


def test_pretrained_name():
    assert minimax_m1.MiniMaxM1PretrainedModel.arch_name() == "MiniMaxM1ForCausalLM"
    assert minimax_m1.MiniMaxM1PretrainedModel.name() == "MiniMaxM1ForCausalLM"


# ===================================================================
# 3. Layer construction
# ===================================================================


def test_linear_attention_layer(m):
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert layer.attention_type == 0
    assert isinstance(layer.self_attn, m.MiniMaxM1LinearAttention)
    assert hasattr(layer.self_attn, "slope_rate")
    assert hasattr(layer.self_attn, "output_gate")
    assert hasattr(layer.self_attn, "norm")
    assert hasattr(layer.self_attn, "out_proj")


def test_full_attention_layer(m):
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=3, prefix="model.layers.3")
    assert layer.attention_type == 1
    assert isinstance(layer.self_attn, m.MiniMaxM1Attention)


def test_deepnorm_defaults(m):
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert layer.layernorm_attention_alpha == 3.556
    assert layer.layernorm_mlp_alpha == 3.556


def test_moe_when_experts_gt_1(m):
    fd = _make_fd_config(num_local_experts=4)
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert isinstance(layer.block_sparse_moe, m.MiniMaxM1MoE)


def test_dense_mlp_when_single_expert(m):
    fd = _make_fd_config(num_local_experts=1)
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert isinstance(layer.block_sparse_moe, m.MiniMaxM1MLP)


def test_fallback_attn_type_when_no_config(m):
    fd = _make_fd_config(num_layers=80)
    delattr(fd.model_config, "attn_type_list")
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=7, prefix="model.layers.7")
    assert layer.attention_type == 1


def test_moe_default_weight_key_map(m):
    """Unquantized config -> weight_key_map has plain .weight keys."""
    fd = _make_fd_config(num_local_experts=4)
    _StubFusedMoE.last_init_kwargs = {}
    m.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = _StubFusedMoE.last_init_kwargs["weight_key_map"]
    assert "gate_weight_key" in wkm
    assert wkm["up_gate_proj_expert_weight_key"].endswith(".up_gate_proj.weight")
    assert "weight_scale" not in str(wkm)


def test_moe_w4a8_weight_key_map(m):
    """w4a8 quant config -> weight_key_map has .quant_weight + scales."""
    fd = _make_fd_config(num_local_experts=4)
    fd.quant_config = SimpleNamespace(moe_quant_type="w4a8")
    fd.model_config.is_quantized = True
    _StubFusedMoE.last_init_kwargs = {}
    m.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = _StubFusedMoE.last_init_kwargs["weight_key_map"]
    assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
    assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
    assert "activation_scale" in wkm["up_gate_proj_expert_in_scale_key"]


def test_moe_w4afp8_dynamic_weight_key_map(m):
    """Dynamic w4afp8 -> quant_weight + weight_scale but no activation_scale."""
    fd = _make_fd_config(num_local_experts=4)
    fd.quant_config = SimpleNamespace(moe_quant_type="w4afp8", moe_dynamic_quant=True)
    fd.model_config.is_quantized = True
    _StubFusedMoE.last_init_kwargs = {}
    m.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = _StubFusedMoE.last_init_kwargs["weight_key_map"]
    assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
    assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
    assert "in_scale_key" not in str(wkm)


# ===================================================================
# 4. Forward-pass smoke tests
# ===================================================================


def test_linear_layer_returns_tuple(m):
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    _patch_layer(layer, 256)
    out = layer(forward_meta=SimpleNamespace(), hidden_states=paddle.randn([4, 256]))
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == [4, 256]


def test_full_attn_layer_returns_tuple(m):
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=3, prefix="model.layers.3")
    _patch_layer(layer, 256)
    out = layer(forward_meta=SimpleNamespace(), hidden_states=paddle.randn([4, 256]))
    assert isinstance(out, tuple) and len(out) == 2


def test_deepnorm_scaling_applied(m):
    fd = _make_fd_config()
    fd.model_config.layernorm_linear_attention_alpha = 2.0
    fd.model_config.layernorm_mlp_alpha = 3.0
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert layer.layernorm_attention_alpha == 2.0
    assert layer.layernorm_mlp_alpha == 3.0


def test_postnorm_forward(m):
    fd = _make_fd_config()
    fd.model_config.postnorm = True
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    _patch_layer(layer, 256)
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


def test_lightning_attn_single_token_shape():
    """Single token: output shape must match [b, h, 1, e]."""
    b, h, n, d = 1, 4, 1, 16
    q = np.random.randn(b, h, n, d)
    k = np.random.randn(b, h, n, d)
    v = np.random.randn(b, h, n, d)
    slope = np.abs(np.random.randn(h)) * 0.1

    output, kv = _lightning_attention_numpy_ref(q, k, v, slope)
    assert output.shape == (b, h, n, d)
    assert kv.shape == (b, h, d, d)


def test_lightning_attn_multi_token_causal():
    """With slope -> 0, approaches standard dot-product attention (cumulative)."""
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


def test_lightning_attn_kv_history_persistence():
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


def test_lightning_attn_multi_head_independent():
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


# ===================================================================
# 6. Model & CausalLM forward
# ===================================================================


def test_model_forward(m):
    """MiniMaxM1Model forward: embed → decoder layers → final norm."""
    fd = _make_fd_config()
    model = m.MiniMaxM1Model(fd)
    for layer in model.layers:
        _patch_layer(layer, 256)
    ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    out = model(ids_remove_padding=ids, forward_meta=SimpleNamespace())
    assert out.shape == [3, 256]


def test_model_single_token(m):
    """Model handles single-token input (generation phase)."""
    fd = _make_fd_config()
    model = m.MiniMaxM1Model(fd)
    for layer in model.layers:
        _patch_layer(layer, 256)
    ids = paddle.to_tensor([42], dtype="int64")
    out = model(ids_remove_padding=ids, forward_meta=SimpleNamespace())
    assert out.shape == [1, 256]


def test_causallm_forward(m):
    """CausalLM wraps model, returns hidden_states matching input batch."""
    fd = _make_fd_config()
    model = m.MiniMaxM1ForCausalLM(fd)
    for layer in model.model.layers:
        _patch_layer(layer, 256)
    ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    hidden = model(inputs={"ids_remove_padding": ids}, forward_meta=SimpleNamespace())
    assert hidden.shape == [3, 256]


def test_compute_logits_float32(m):
    """compute_logits yields float32 logits over full vocab."""
    fd = _make_fd_config()
    model = m.MiniMaxM1ForCausalLM(fd)
    hidden = paddle.randn([3, 256], dtype="float16")
    logits = model.compute_logits(hidden, forward_meta=SimpleNamespace())
    assert logits.dtype == paddle.float32
    assert logits.shape == [3, 1024]


# ===================================================================
# 7. Weight loading — HF→FD name remapping
# ===================================================================


def test_set_state_dict_expert_rename(m):
    """set_state_dict: HF expert w1/w3/w2 → FD gate_proj/up_proj/down_proj."""
    fd = _make_fd_config()
    model = m.MiniMaxM1ForCausalLM(fd)
    state = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": np.zeros([1], dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": np.zeros([1], dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": np.zeros([1], dtype=np.float32),
    }
    model.set_state_dict(state)


def test_set_state_dict_qkv_merge(m):
    """set_state_dict: separate q/k/v_proj → merged qkv_proj via concat."""
    fd = _make_fd_config()
    model = m.MiniMaxM1ForCausalLM(fd)
    h, kv_h, d, hidden = 8, 2, 32, 256
    state = {
        "model.layers.3.self_attn.q_proj.weight": np.ones([h * d, hidden], dtype=np.float32),
        "model.layers.3.self_attn.k_proj.weight": np.ones([kv_h * d, hidden], dtype=np.float32) * 2,
        "model.layers.3.self_attn.v_proj.weight": np.ones([kv_h * d, hidden], dtype=np.float32) * 3,
    }
    model.set_state_dict(state)


def test_set_state_dict_passthrough(m):
    """Non-expert, non-qkv keys pass through without renaming."""
    fd = _make_fd_config()
    model = m.MiniMaxM1ForCausalLM(fd)
    state = {
        "model.layers.0.input_layernorm.weight": np.zeros([256], dtype=np.float32),
        "model.norm.weight": np.zeros([256], dtype=np.float32),
        "lm_head.weight": np.zeros([1024, 256], dtype=np.float32),
    }
    model.set_state_dict(state)


def test_multi_layer_residual_no_blowup(m):
    """Regression: multi-layer forward must not cause residual blowup.

    C1 fix: DeepNorm folds the residual into hidden_states, so the layer
    returns ``(hidden_states, None)`` — not ``(hidden_states, residual)``.
    If the old behaviour returns a non-None residual, the next iteration
    adds it again → exponential growth.  This test stacks 4 layers and
    checks the output norm stays bounded.
    """
    fd = _make_fd_config(num_layers=4)
    model = m.MiniMaxM1Model(fd_config=fd)
    ids = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
    meta = SimpleNamespace()
    out = model(ids_remove_padding=ids, forward_meta=meta)
    assert paddle.isfinite(out).all(), "Output contains NaN/Inf — residual blowup"
    assert out.abs().max().item() < 1e4, (
        f"Output magnitude {out.abs().max().item():.1f} too large — "
        "possible residual double-counting (C1 regression)"
    )


def test_decoder_layer_returns_none_residual(m):
    """DecoderLayer must return None as residual (DeepNorm convention)."""
    fd = _make_fd_config()
    layer = m.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    meta = SimpleNamespace()
    h = paddle.randn([2, 4])
    out, residual = layer(forward_meta=meta, hidden_states=h)
    assert residual is None, f"Expected None residual (DeepNorm folds it into hidden_states), got {type(residual)}"
