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
Tests for MiniMax-M1 model: architecture dispatch, weight loading, forward paths,
and Lightning Attention algorithm correctness.

Follows H10 gold standard (test_ernie4_5_mtp.py pattern):
- Direct import of fastdeploy module
- Real paddle.nn.Layer stubs (not MagicMock)
- monkeypatch.setattr for surgical replacement
- Tests exercise actual FD code paths
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.models import minimax_m1

# ── Lightweight stubs (real nn.Layer subclasses) ────────────────────────────


class _StubRMSNorm(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x, residual_input=None, forward_meta=None):
        if residual_input is None:
            residual_input = paddle.zeros_like(x)
        return x, residual_input + x

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubLinear(paddle.nn.Layer):
    """Stub for ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear, ReplicatedLinear."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False
        self._out = kw.get("output_size", None)

    def forward(self, x, *a, **kw):
        if self._out is not None:
            shape = list(x.shape)
            shape[-1] = self._out
            return paddle.zeros(shape, dtype=x.dtype)
        return x

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubAttention(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, q=None, k=None, v=None, qkv=None, forward_meta=None, **kw):
        if qkv is not None:
            return qkv
        return q

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubSiluAndMul(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x):
        return x[..., : x.shape[-1] // 2]


class _StubFusedMoE(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.weight_key_map = kw.get("weight_key_map", {})
        self.load_state_dict_called = False

    def forward(self, hidden_states, gate, forward_meta=None):
        return hidden_states

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True

    @staticmethod
    def make_expert_params_mapping(**kw):
        return []


class _StubEmbedding(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.hidden_size = kw.get("embedding_dim", 4)
        self.load_state_dict_called = False

    def forward(self, ids_remove_padding=None, forward_meta=None):
        return paddle.zeros([ids_remove_padding.shape[0], self.hidden_size], "float32")

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubLMHead(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return x

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


def _stub_lightning_attention(q, k, v, slope, block_size=256, kv_history=None):
    """Stub: return zeros matching shapes."""
    b, h, seq_len, d = q.shape
    out = paddle.zeros_like(q)
    if kv_history is None:
        kv_history = paddle.zeros([b, h, d, d], dtype=q.dtype)
    return out, kv_history


def _stub_all_reduce(x):
    return x


def _stub_graph_opt(cls):
    return cls


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_fd_config(
    hidden_size=4,
    num_layers=2,
    num_local_experts=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=2,
    postnorm=False,
):
    mc = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=8,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        num_local_experts=num_local_experts,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        postnorm=postnorm,
        attn_type_list=[0, 1][:num_layers],
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


@pytest.fixture()
def mm1(monkeypatch):
    """Patch heavy GPU deps in minimax_m1 module with lightweight stubs."""
    monkeypatch.setattr(minimax_m1, "RMSNorm", _StubRMSNorm)
    monkeypatch.setattr(minimax_m1, "ColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "MergedColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "QKVParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "RowParallelLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "ReplicatedLinear", _StubLinear)
    monkeypatch.setattr(minimax_m1, "Attention", _StubAttention)
    monkeypatch.setattr(minimax_m1, "SiluAndMul", _StubSiluAndMul)
    monkeypatch.setattr(minimax_m1, "FusedMoE", _StubFusedMoE)
    monkeypatch.setattr(minimax_m1, "VocabParallelEmbedding", _StubEmbedding)
    monkeypatch.setattr(minimax_m1, "ParallelLMHead", _StubLMHead)
    monkeypatch.setattr(minimax_m1, "lightning_attention", _stub_lightning_attention)
    monkeypatch.setattr(minimax_m1, "tensor_model_parallel_all_reduce", _stub_all_reduce)
    monkeypatch.setattr(minimax_m1, "support_graph_optimization", _stub_graph_opt)
    return minimax_m1


# ===================================================================
# 1. Pure-logic tests (static methods — no stubs needed)
# ===================================================================


class TestBuildAttnTypeList:

    def test_80_layers_has_10_full_attention(self):
        attn_list = minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(80)
        assert len(attn_list) == 80
        full_indices = [i for i, t in enumerate(attn_list) if t == 1]
        assert full_indices == [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]

    def test_short_model_clips_indices(self):
        attn_list = minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(10)
        assert len(attn_list) == 10
        assert attn_list[7] == 1
        assert sum(attn_list) == 1

    def test_single_layer_all_linear(self):
        assert minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(1) == [0]

    def test_all_linear_below_first_full_index(self):
        assert all(t == 0 for t in minimax_m1.MiniMaxM1DecoderLayer._build_attn_type_list(7))


class TestBuildSlopeTensor:

    def test_power_of_two_heads(self):
        slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(8)
        assert slopes.shape == [8, 1, 1]
        assert (slopes.flatten().numpy() > 0).all()

    def test_non_power_of_two_heads(self):
        slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(12)
        assert slopes.shape == [12, 1, 1]
        assert (slopes.flatten().numpy() > 0).all()

    def test_64_heads_first_slope(self):
        slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(64)
        assert slopes.shape == [64, 1, 1]
        expected_start = 2 ** (-(2 ** (-(math.log2(64) - 3))))
        np.testing.assert_allclose(slopes.flatten().numpy()[0], expected_start, rtol=1e-5)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
    def test_slopes_all_positive(self, n):
        slopes = minimax_m1.MiniMaxM1LinearAttention._build_slope_tensor(n)
        assert (slopes.flatten().numpy() > 0).all()


# ===================================================================
# 2. Model registration (uses real ModelRegistry)
# ===================================================================


class TestModelRegistration:

    def test_primary_architecture_registered(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        assert "MiniMaxM1ForCausalLM" in ModelRegistry._arch_to_model_cls

    def test_alias_architecture_registered(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        assert "MiniMaxText01ForCausalLM" in ModelRegistry._arch_to_model_cls

    def test_registered_class(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        assert ModelRegistry._arch_to_model_cls["MiniMaxM1ForCausalLM"] is minimax_m1.MiniMaxM1ForCausalLM

    def test_name_method(self):
        assert minimax_m1.MiniMaxM1ForCausalLM.name() == "MiniMaxM1ForCausalLM"

    def test_pretrained_model_names(self):
        assert minimax_m1.MiniMaxM1PretrainedModel.arch_name() == "MiniMaxM1ForCausalLM"
        assert minimax_m1.MiniMaxM1PretrainedModel.name() == "MiniMaxM1ForCausalLM"


# ===================================================================
# 3. Layer construction (exercises real FD code with stubs)
# ===================================================================


class TestDecoderLayerConstruction:

    def test_linear_attention_layer(self, mm1):
        fd = _make_fd_config()
        layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert layer.attention_type == 0
        assert isinstance(layer.self_attn, mm1.MiniMaxM1LinearAttention)
        assert hasattr(layer.self_attn, "slope_rate")
        assert hasattr(layer.self_attn, "output_gate")

    def test_full_attention_layer(self, mm1):
        fd = _make_fd_config()
        layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=1, prefix="model.layers.1")
        assert layer.attention_type == 1
        assert isinstance(layer.self_attn, mm1.MiniMaxM1Attention)

    def test_moe_when_experts_gt_1(self, mm1):
        fd = _make_fd_config(num_local_experts=4)
        layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert isinstance(layer.block_sparse_moe, mm1.MiniMaxM1MoE)

    def test_dense_mlp_when_single_expert(self, mm1):
        fd = _make_fd_config(num_local_experts=1)
        layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
        assert isinstance(layer.block_sparse_moe, mm1.MiniMaxM1MLP)

    def test_fallback_attn_type_when_no_config(self, mm1):
        fd = _make_fd_config(num_layers=80)
        delattr(fd.model_config, "attn_type_list")
        layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=7, prefix="model.layers.7")
        assert layer.attention_type == 1


# ===================================================================
# 4. Forward pass tests (exercises real FD forward code)
# ===================================================================


def test_decoder_layer_forward_prenorm(mm1):
    """Pre-norm forward: exercises real DecoderLayer.forward code path."""
    fd = _make_fd_config(postnorm=False)
    layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    meta = SimpleNamespace()
    h = paddle.randn([2, 4])
    out, residual = layer(forward_meta=meta, hidden_states=h)
    assert out.shape[-1] == 4 and out.shape[0] == 2
    assert residual.shape[-1] == 4 and residual.shape[0] == 2


def test_decoder_layer_forward_postnorm(mm1):
    """Post-norm forward: exercises the postnorm=True branch."""
    fd = _make_fd_config(postnorm=True)
    layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    meta = SimpleNamespace()
    h = paddle.randn([2, 4])
    out, residual = layer(forward_meta=meta, hidden_states=h)
    assert out.shape[-1] == 4 and out.shape[0] == 2
    assert residual.shape[-1] == 4 and residual.shape[0] == 2


def test_decoder_layer_forward_full_attn(mm1):
    """Full attention layer forward."""
    fd = _make_fd_config()
    layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=1, prefix="model.layers.1")
    meta = SimpleNamespace()
    h = paddle.randn([2, 4])
    out, residual = layer(forward_meta=meta, hidden_states=h)
    assert out.shape[-1] == 4 and out.shape[0] == 2


def test_deepnorm_scaling(mm1):
    """Verify DeepNorm alpha/beta are read from config."""
    fd = _make_fd_config()
    fd.model_config.layernorm_linear_attention_alpha = 2.0
    fd.model_config.layernorm_mlp_alpha = 3.0
    layer = mm1.MiniMaxM1DecoderLayer(fd, layer_id=0, prefix="model.layers.0")
    assert layer.layernorm_attention_alpha == 2.0
    assert layer.layernorm_mlp_alpha == 3.0


def test_model_forward(mm1):
    """MiniMaxM1Model forward: exercises embed -> layers -> norm chain."""
    fd = _make_fd_config(hidden_size=4, num_layers=2)
    model = mm1.MiniMaxM1Model(fd_config=fd)
    ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    meta = SimpleNamespace()
    out = model(ids_remove_padding=ids, forward_meta=meta)
    assert out.shape[-1] == 4 and out.shape[0] == 3


def test_model_load_state_dict(mm1):
    """Verify load_state_dict delegates to all sublayers."""
    fd = _make_fd_config(hidden_size=4, num_layers=2)
    model = mm1.MiniMaxM1Model(fd_config=fd)
    model.load_state_dict({"w": np.zeros([1], dtype=np.float32)})
    assert model.embed_tokens.load_state_dict_called
    assert model.norm.load_state_dict_called
    for layer in model.layers:
        assert layer.input_layernorm.load_state_dict_called


def test_causallm_forward_and_compute_logits(mm1):
    """CausalLM forward + compute_logits: exercises the top-level model."""
    fd = _make_fd_config(hidden_size=4, num_layers=1)
    model = mm1.MiniMaxM1ForCausalLM(fd)

    ids = paddle.to_tensor([0, 1], dtype="int64")
    meta = SimpleNamespace()
    hidden = model(inputs={"ids_remove_padding": ids}, forward_meta=meta)
    assert hidden.shape[-1] == 4 and hidden.shape[0] == 2

    logits = model.compute_logits(hidden.astype("float16"), meta)
    assert logits.dtype == paddle.float32


def test_causallm_name(mm1):
    """CausalLM.name() returns expected value."""
    assert mm1.MiniMaxM1ForCausalLM.name() == "MiniMaxM1ForCausalLM"


# ===================================================================
# 5. set_state_dict — HF->FD weight remapping
# ===================================================================


def test_set_state_dict_expert_remap(mm1):
    """set_state_dict remaps MoE expert weights: w1->gate_proj, w2->down_proj, w3->up_proj."""
    fd = _make_fd_config(hidden_size=4, num_layers=1)
    model = mm1.MiniMaxM1ForCausalLM(fd)

    captured = {}
    model.model.load_state_dict = lambda sd: captured.update(sd)
    model.lm_head.load_state_dict = lambda sd: None

    sd = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": np.ones([2, 4], dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": np.ones([4, 2], dtype=np.float32),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": np.ones([2, 4], dtype=np.float32),
    }
    model.set_state_dict(sd)

    assert "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight" in captured
    assert "model.layers.0.block_sparse_moe.experts.0.down_proj.weight" in captured
    assert "model.layers.0.block_sparse_moe.experts.0.up_proj.weight" in captured


def test_set_state_dict_qkv_merge(mm1):
    """set_state_dict merges q/k/v into qkv_proj for full attention layers."""
    fd = _make_fd_config(hidden_size=4, num_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    model = mm1.MiniMaxM1ForCausalLM(fd)

    captured = {}
    model.model.load_state_dict = lambda sd: captured.update(sd)
    model.lm_head.load_state_dict = lambda sd: None

    # Layer 1 is full attention (attn_type_list=[0,1])
    q_w = np.arange(16, dtype=np.float32).reshape(4, 4)  # [num_heads * head_dim, hidden]
    k_w = np.arange(8, dtype=np.float32).reshape(2, 4)  # [num_kv_heads * head_dim, hidden]
    v_w = np.arange(8, dtype=np.float32).reshape(2, 4)
    sd = {
        "model.layers.1.self_attn.q_proj.weight": q_w,
        "model.layers.1.self_attn.k_proj.weight": k_w,
        "model.layers.1.self_attn.v_proj.weight": v_w,
    }
    model.set_state_dict(sd)

    merged_key = "model.layers.1.self_attn.qkv_proj.weight"
    assert merged_key in captured
    expected = np.concatenate([q_w, k_w, v_w], axis=0)
    np.testing.assert_array_equal(captured[merged_key], expected)


def test_set_state_dict_passthrough(mm1):
    """Non-expert, non-qkv weights pass through unchanged."""
    fd = _make_fd_config(hidden_size=4, num_layers=1)
    model = mm1.MiniMaxM1ForCausalLM(fd)

    captured = {}
    model.model.load_state_dict = lambda sd: captured.update(sd)
    model.lm_head.load_state_dict = lambda sd: None

    sd = {"model.norm.weight": np.ones([4], dtype=np.float32)}
    model.set_state_dict(sd)
    assert "model.norm.weight" in captured


def test_set_state_dict_qkv_paddle_tensors(mm1):
    """QKV merge works with Paddle tensors (not just numpy)."""
    fd = _make_fd_config(hidden_size=4, num_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    model = mm1.MiniMaxM1ForCausalLM(fd)

    captured = {}
    model.model.load_state_dict = lambda sd: captured.update(sd)
    model.lm_head.load_state_dict = lambda sd: None

    q_w = paddle.arange(16, dtype="float32").reshape([4, 4])
    k_w = paddle.arange(8, dtype="float32").reshape([2, 4])
    v_w = paddle.arange(8, dtype="float32").reshape([2, 4])
    sd = {
        "model.layers.1.self_attn.q_proj.weight": q_w,
        "model.layers.1.self_attn.k_proj.weight": k_w,
        "model.layers.1.self_attn.v_proj.weight": v_w,
    }
    model.set_state_dict(sd)

    merged = captured["model.layers.1.self_attn.qkv_proj.weight"]
    assert isinstance(merged, paddle.Tensor)
    assert merged.shape == [8, 4]


# ===================================================================
# 6. MoE weight key map construction
# ===================================================================


def test_moe_default_weight_keys(mm1):
    """Unquantized MoE: weight_key_map has plain .weight keys."""
    fd = _make_fd_config(num_local_experts=4)
    moe = mm1.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = moe.experts.weight_key_map
    assert "gate_weight_key" in wkm
    assert wkm["up_gate_proj_expert_weight_key"].endswith(".up_gate_proj.weight")
    assert "weight_scale" not in str(wkm)


def test_moe_w4a8_weight_keys(mm1):
    """w4a8 quant: weight_key_map has .quant_weight + scales."""
    fd = _make_fd_config(num_local_experts=4)
    fd.quant_config = SimpleNamespace(moe_quant_type="w4a8")
    fd.model_config.is_quantized = True
    moe = mm1.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = moe.experts.weight_key_map
    assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
    assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
    assert "activation_scale" in wkm["up_gate_proj_expert_in_scale_key"]


def test_moe_w4afp8_dynamic_weight_keys(mm1):
    """Dynamic w4afp8: quant_weight + weight_scale but no activation_scale."""
    fd = _make_fd_config(num_local_experts=4)
    fd.quant_config = SimpleNamespace(moe_quant_type="w4afp8", moe_dynamic_quant=True)
    fd.model_config.is_quantized = True
    moe = mm1.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    wkm = moe.experts.weight_key_map
    assert "quant_weight" in wkm["up_gate_proj_expert_weight_key"]
    assert "weight_scale" in wkm["up_gate_proj_expert_weight_scale_key"]
    assert "in_scale_key" not in str(wkm)


def test_moe_tp_all_reduce(mm1):
    """MoE with tp_size > 1 sets the attribute."""
    fd = _make_fd_config(num_local_experts=4)
    fd.parallel_config.tensor_parallel_size = 2
    moe = mm1.MiniMaxM1MoE(fd, layer_id=0, prefix="model.layers.0.block_sparse_moe")
    assert moe.tp_size == 2


# ===================================================================
# 7. Linear attention construction and forward
# ===================================================================


def test_linear_attention_slope_rate_shape(mm1):
    fd = _make_fd_config(num_layers=2, num_attention_heads=4, head_dim=2)
    layer = mm1.MiniMaxM1LinearAttention(fd, layer_id=0, linear_layer_id=0, prefix="model.layers.0.self_attn")
    assert layer.slope_rate.shape == [4, 1, 1]
    assert (layer.slope_rate.flatten().numpy() > 0).all()


def test_linear_attention_kv_cache_shape(mm1):
    fd = _make_fd_config(num_attention_heads=4, head_dim=2)
    layer = mm1.MiniMaxM1LinearAttention(fd, layer_id=0, linear_layer_id=0, prefix="model.layers.0.self_attn")
    assert layer.kv_cache_shape == (4, 2, 2)


def test_linear_attention_forward(mm1):
    fd = _make_fd_config(hidden_size=4, num_attention_heads=4, head_dim=1)
    layer = mm1.MiniMaxM1LinearAttention(fd, layer_id=0, linear_layer_id=0, prefix="model.layers.0.self_attn")
    meta = SimpleNamespace()
    h = paddle.randn([1, 4])
    out = layer(forward_meta=meta, hidden_states=h)
    # LinearAttention adds seq=1 dim internally via 4D reshape
    assert out.shape[-1] == 4 and out.shape[0] == 1


def test_linear_attention_load_state_dict(mm1):
    fd = _make_fd_config(num_attention_heads=4, head_dim=2)
    layer = mm1.MiniMaxM1LinearAttention(fd, layer_id=0, linear_layer_id=0, prefix="model.layers.0.self_attn")
    sd = {"w": np.zeros([1], dtype=np.float32)}
    layer.load_state_dict(sd)
    assert layer.qkv_proj.load_state_dict_called
    assert layer.output_gate.load_state_dict_called
    assert layer.out_proj.load_state_dict_called
    assert layer.norm.load_state_dict_called


# ===================================================================
# 8. Full attention
# ===================================================================


def test_full_attention_forward(mm1):
    fd = _make_fd_config(hidden_size=4, num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    layer = mm1.MiniMaxM1Attention(fd, layer_id=1, prefix="model.layers.1.self_attn")
    meta = SimpleNamespace()
    h = paddle.randn([2, 4])
    out = layer(forward_meta=meta, hidden_states=h)
    assert out.shape[-1] == 4 and out.shape[0] == 2


def test_full_attention_load_state_dict(mm1):
    fd = _make_fd_config(num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    layer = mm1.MiniMaxM1Attention(fd, layer_id=1, prefix="model.layers.1.self_attn")
    layer.load_state_dict({"w": np.zeros([1], dtype=np.float32)})
    assert layer.qkv_proj.load_state_dict_called
    assert layer.o_proj.load_state_dict_called
    assert layer.attn.load_state_dict_called


# ===================================================================
# 9. MLP
# ===================================================================


def test_mlp_forward(mm1):
    fd = _make_fd_config(num_local_experts=1)
    mlp = mm1.MiniMaxM1MLP(fd, intermediate_size=8, prefix="model.layers.0.mlp")
    h = paddle.randn([2, 4])
    out = mlp.forward(h)
    assert out.shape == [2, 4]


def test_mlp_load_state_dict(mm1):
    fd = _make_fd_config()
    mlp = mm1.MiniMaxM1MLP(fd, intermediate_size=8, prefix="model.layers.0.mlp")
    mlp.load_state_dict({"w": np.zeros([1], dtype=np.float32)})
    assert mlp.gate_up_proj.load_state_dict_called
    assert mlp.down_proj.load_state_dict_called


# ===================================================================
# 10. Lightning Attention — Pure-Python reference algorithm
# ===================================================================


def _lightning_attention_numpy_ref(q, k, v, slope, kv_history=None):
    """
    Pure NumPy reference implementation of linear attention with exponential decay.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    output = np.zeros((b, h, n, e), dtype=np.float64)

    if kv_history is None:
        kv_state = np.zeros((b, h, d, e), dtype=np.float64)
    else:
        kv_state = kv_history.copy()

    for t in range(n):
        decay = np.exp(-slope)[np.newaxis, :, np.newaxis, np.newaxis]
        kv_state = kv_state * decay
        kt = k[:, :, t, :]
        vt = v[:, :, t, :]
        kv_state += kt[:, :, :, np.newaxis] * vt[:, :, np.newaxis, :]
        qt = q[:, :, t, :]
        output[:, :, t, :] = np.einsum("bhd,bhde->bhe", qt, kv_state)

    return output, kv_state


class TestLightningAttentionPurePython:
    """Validate Lightning Attention algorithm correctness via NumPy reference."""

    def test_single_token_output_shape(self):
        b, h, n, d = 1, 4, 1, 16
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.abs(np.random.randn(h)) * 0.1
        output, kv = _lightning_attention_numpy_ref(q, k, v, slope)
        assert output.shape == (b, h, n, d)
        assert kv.shape == (b, h, d, d)

    def test_multi_token_causal(self):
        """With slope approaching 0, approaches causal linear attention."""
        b, h, n, d = 1, 2, 4, 8
        np.random.seed(42)
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.full(h, 1e-8)
        output, _ = _lightning_attention_numpy_ref(q, k, v, slope)

        for t in range(n):
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
        _, kv_after_1 = _lightning_attention_numpy_ref(q1, k1, v1, slope)
        out2, _ = _lightning_attention_numpy_ref(q2, k2, v2, slope, kv_history=kv_after_1)
        q_full = np.concatenate([q1, q2], axis=2)
        k_full = np.concatenate([k1, k2], axis=2)
        v_full = np.concatenate([v1, v2], axis=2)
        out_full, _ = _lightning_attention_numpy_ref(q_full, k_full, v_full, slope)
        np.testing.assert_allclose(out2[:, :, 0, :], out_full[:, :, n, :], rtol=1e-5, atol=1e-7)

    def test_multi_head_independent(self):
        """Heads are computed independently - zeroing one head Q zeros its output."""
        b, h, n, d = 1, 8, 4, 16
        np.random.seed(7)
        q = np.random.randn(b, h, n, d)
        k = np.random.randn(b, h, n, d)
        v = np.random.randn(b, h, n, d)
        slope = np.abs(np.random.randn(h)) * 0.1
        q_masked = q.copy()
        q_masked[:, 3, :, :] = 0.0
        output, _ = _lightning_attention_numpy_ref(q_masked, k, v, slope)
        np.testing.assert_allclose(output[:, 3, :, :], 0.0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
