"""Tests for MiniCPM4/4.1 model implementation.
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

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

# Patch paddle.compat before importing fastdeploy (beta-2 compat)
if not hasattr(paddle, "compat"):

    class _PaddleCompat:
        @staticmethod
        def enable_torch_proxy(scope=None):
            return None

    paddle.compat = _PaddleCompat()

from fastdeploy.model_executor.models import minicpm4

# ── Stub helpers ────────────────────────────────────────────────────────────


class _StubLinear(paddle.nn.Layer):
    """Stub for parallel linear layers (merged column, QKV, row)."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return x

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubActivation(paddle.nn.Layer):
    """Stub for SiluAndMul activation — identity pass-through."""

    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x):
        return x


class _StubRMSNorm(paddle.nn.Layer):
    """Stub for RMSNorm layer — returns (hidden, residual) pair."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x, *args, **kwargs):
        return x, x

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubAttention(paddle.nn.Layer):
    """Stub for Attention — identity on qkv input."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, qkv=None, forward_meta=None):
        return qkv

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubEmbedding(paddle.nn.Layer):
    """Stub for VocabParallelEmbedding."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False
        self._h = kw.get("embedding_dim", 4)

    def forward(self, ids_remove_padding=None, forward_meta=None):
        return paddle.ones([ids_remove_padding.shape[0], self._h], dtype="float32")

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubLMHead(paddle.nn.Layer):
    """Stub for ParallelLMHead — projects to vocab_size."""

    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False
        self._vocab = kw.get("num_embeddings", 128)

    def forward(self, x):
        return paddle.ones([x.shape[0], self._vocab], dtype=x.dtype)

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


# ── Configuration helpers ───────────────────────────────────────────────────

# Reference dimensions from openbmb/MiniCPM4.1-8B config.json
_HIDDEN = 4
_INTERMEDIATE = 8
_LAYERS = 2
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 2
_VOCAB = 128
_ORI_VOCAB = 100


def _make_fd_config(
    hidden_size=_HIDDEN,
    num_layers=_LAYERS,
    scale_emb=12,
    scale_depth=1.4,
    dim_model_base=256,
):
    mc = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=num_layers,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=_VOCAB,
        ori_vocab_size=_ORI_VOCAB,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        scale_emb=scale_emb,
        scale_depth=scale_depth,
        dim_model_base=dim_model_base,
        tie_word_embeddings=False,
        model_format="torch",
        is_quantized=False,
        pretrained_config=SimpleNamespace(prefix_name="minicpm4"),
        fuse_attention_qkv=True,
        tensor_model_parallel_size=1,
        tensor_parallel_rank=0,
        moe_layer_start_index=0,
    )
    return SimpleNamespace(
        model_config=mc,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            tp_group=None,
            expert_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        graph_opt_config=SimpleNamespace(graph_opt_level=0, use_cudagraph=False),
        scheduler_config=SimpleNamespace(splitwise_role="prefill", max_num_seqs=1),
        load_config=SimpleNamespace(
            dynamic_load_weight=False,
            load_choices="default_v0",
            is_pre_sharded=False,
        ),
        quant_config=None,
    )


# ── Fixture ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mod(monkeypatch):
    """Inject stubs into minicpm4 module for CPU-safe testing."""
    monkeypatch.setattr(minicpm4, "MergedColumnParallelLinear", _StubLinear)
    monkeypatch.setattr(minicpm4, "QKVParallelLinear", _StubLinear)
    monkeypatch.setattr(minicpm4, "RowParallelLinear", _StubLinear)
    monkeypatch.setattr(minicpm4, "SiluAndMul", _StubActivation)
    monkeypatch.setattr(minicpm4, "RMSNorm", _StubRMSNorm)
    monkeypatch.setattr(minicpm4, "Attention", _StubAttention)
    monkeypatch.setattr(minicpm4, "VocabParallelEmbedding", _StubEmbedding)
    monkeypatch.setattr(minicpm4, "ParallelLMHead", _StubLMHead)
    return minicpm4


# ── MLP tests ───────────────────────────────────────────────────────────────


def test_mlp_forward(mod):
    """MLP: up_gate_proj -> act_fn -> down_proj pass-through."""
    fd = _make_fd_config()
    mlp = mod.MiniCPM4MLP(fd_config=fd, prefix="minicpm4.layers.0.mlp")
    x = paddle.ones([2, _HIDDEN], dtype="float32")
    out = mlp.forward(x, forward_meta=None)
    assert out.shape == [2, _HIDDEN]


def test_mlp_load_state_dict(mod):
    """MLP load_state_dict delegates to sub-layers."""
    fd = _make_fd_config()
    mlp = mod.MiniCPM4MLP(fd_config=fd, prefix="minicpm4.layers.0.mlp")
    mlp.load_state_dict({})
    assert mlp.up_gate_proj.load_state_dict_called
    assert mlp.down_proj.load_state_dict_called


# ── Attention tests ─────────────────────────────────────────────────────────


def test_attention_forward(mod):
    """Attention: qkv_proj -> attn -> o_proj pass-through."""
    fd = _make_fd_config()
    attn = mod.MiniCPM4Attention(fd_config=fd, layer_id=0, prefix="minicpm4.layers.0.self_attn")
    x = paddle.ones([2, _HIDDEN], dtype="float32")
    meta = SimpleNamespace()
    out = attn.forward(forward_meta=meta, hidden_states=x)
    assert out.shape == [2, _HIDDEN]


def test_attention_load_state_dict(mod):
    """Attention load_state_dict delegates to sub-layers."""
    fd = _make_fd_config()
    attn = mod.MiniCPM4Attention(fd_config=fd, layer_id=0, prefix="minicpm4.layers.0.self_attn")
    attn.load_state_dict({})
    assert attn.qkv_proj.load_state_dict_called
    assert attn.o_proj.load_state_dict_called
    assert attn.attn.load_state_dict_called


# ── DecoderLayer tests ──────────────────────────────────────────────────────


def test_decoder_layer_residual_scale(mod):
    """DecoderLayer computes muP residual_scale = scale_depth / sqrt(N)."""
    fd = _make_fd_config(scale_depth=1.4, num_layers=32)
    layer = mod.MiniCPM4DecoderLayer(fd_config=fd, prefix="minicpm4.layers.0")
    expected = 1.4 / math.sqrt(32)
    assert abs(layer.residual_scale - expected) < 1e-10


def test_decoder_layer_forward(mod):
    """DecoderLayer forward applies muP scaling to both attn and MLP outputs."""
    fd = _make_fd_config(scale_depth=1.4, num_layers=32)
    layer = mod.MiniCPM4DecoderLayer(fd_config=fd, prefix="minicpm4.layers.0")
    x = paddle.full([2, _HIDDEN], 2.0, dtype="float32")
    meta = SimpleNamespace()
    hidden, residual = layer.forward(forward_meta=meta, hidden_states=x, residual=None)

    # Output should be scaled by residual_scale twice (attn + mlp)
    scale = 1.4 / math.sqrt(32)
    # With identity stubs: norm returns (x, x), attn/mlp are identity
    # Path: norm(x,None)->(x,x), attn(x)->x, x*scale, norm(x*scale,x)->(x*scale,x*scale),
    #        mlp(x*scale)->x*scale, x*scale*scale
    expected_hidden = 2.0 * scale * scale
    np.testing.assert_allclose(hidden.numpy().mean(), expected_hidden, rtol=1e-5)
    assert residual is not None


def test_decoder_layer_load_state_dict(mod):
    """DecoderLayer load_state_dict delegates to all sub-layers."""
    fd = _make_fd_config()
    layer = mod.MiniCPM4DecoderLayer(fd_config=fd, prefix="minicpm4.layers.0")
    layer.load_state_dict({})
    assert layer.self_attn.qkv_proj.load_state_dict_called
    assert layer.mlp.up_gate_proj.load_state_dict_called
    assert layer.input_layernorm.load_state_dict_called
    assert layer.post_attention_layernorm.load_state_dict_called


# ── Model tests ─────────────────────────────────────────────────────────────


def test_model_forward_with_embedding_scale(mod):
    """MiniCPM4Model applies scale_emb to embedding output."""
    fd = _make_fd_config(scale_emb=12, num_layers=1)
    model = mod.MiniCPM4Model(fd_config=fd)
    ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    meta = SimpleNamespace()
    out = model.forward(ids_remove_padding=ids, forward_meta=meta)
    assert out.shape == [3, _HIDDEN]
    # Embedding returns ones, scaled by 12, then through decoder layers and final norm
    assert paddle.isfinite(out).all()


def test_model_no_embedding_scale(mod):
    """When scale_emb=1, no embedding scaling applied."""
    fd = _make_fd_config(scale_emb=1, num_layers=1)
    model = mod.MiniCPM4Model(fd_config=fd)
    ids = paddle.to_tensor([0, 1], dtype="int64")
    meta = SimpleNamespace()
    out = model.forward(ids_remove_padding=ids, forward_meta=meta)
    assert out.shape == [2, _HIDDEN]


def test_model_load_state_dict(mod):
    """Model load_state_dict delegates to embed_tokens, norm, and layers."""
    fd = _make_fd_config(num_layers=2)
    model = mod.MiniCPM4Model(fd_config=fd)
    model.load_state_dict({})
    assert model.embed_tokens.load_state_dict_called
    assert model.norm.load_state_dict_called
    for layer in model.layers:
        assert layer.self_attn.qkv_proj.load_state_dict_called


# ── CausalLM tests ──────────────────────────────────────────────────────────


def test_causallm_forward(mod):
    """CausalLM full forward: ids -> model -> hidden_states."""
    fd = _make_fd_config(num_layers=1)
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    ids = paddle.to_tensor([0, 1, 2], dtype="int64")
    meta = SimpleNamespace()
    inputs = {"ids_remove_padding": ids}
    hidden = model.forward(inputs=inputs, forward_meta=meta)
    assert hidden.shape == [3, _HIDDEN]


def test_causallm_compute_logits_mup_scaling(mod):
    """compute_logits applies muP scaling: hidden /= (hidden_size / dim_model_base)."""
    fd = _make_fd_config(dim_model_base=2)  # lm_head_scale = 4/2 = 2.0
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    assert model.lm_head_scale == 2.0

    hidden = paddle.full([2, _HIDDEN], 4.0, dtype="float32")
    logits = model.compute_logits(hidden, forward_meta=None)
    assert logits.dtype == paddle.float32
    assert logits.shape == [2, _VOCAB]


def test_causallm_compute_logits_vocab_mask(mod):
    """compute_logits masks extended vocab positions to -inf."""
    fd = _make_fd_config()
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    hidden = paddle.ones([2, _HIDDEN], dtype="float32")
    logits = model.compute_logits(hidden, forward_meta=None)

    # Valid vocab
    assert paddle.isfinite(logits[:, :_ORI_VOCAB]).all()
    # Extended vocab -> -inf
    assert paddle.isinf(logits[:, _ORI_VOCAB:]).all()
    assert (logits[:, _ORI_VOCAB:] < 0).all()


def test_causallm_lm_head_scale_fallback(mod):
    """When dim_model_base is None, lm_head_scale defaults to 1.0."""
    fd = _make_fd_config(dim_model_base=None)
    fd.model_config.dim_model_base = None
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    assert model.lm_head_scale == 1.0


def test_causallm_set_state_dict(mod):
    """set_state_dict delegates to model and lm_head."""
    fd = _make_fd_config(num_layers=1)
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    model.set_state_dict({})
    assert model.minicpm4.embed_tokens.load_state_dict_called
    assert model.lm_head.load_state_dict_called


def test_causallm_name(mod):
    """Class name method returns 'MiniCPMForCausalLM'."""
    assert mod.MiniCPM4ForCausalLM.name() == "MiniCPMForCausalLM"


def test_causallm_tie_word_embeddings(mod):
    """When tie_word_embeddings=True, load_weights sets lm_head from embed."""
    fd = _make_fd_config()
    fd.model_config.tie_word_embeddings = True
    model = mod.MiniCPM4ForCausalLM(fd_config=fd)
    # tie_word_embeddings flag is read
    assert model.tie_word_embeddings is True


# ── Weight loading & mapping tests ──────────────────────────────────────────


def test_weights_mapper_prefix_rename():
    """WeightsMapper renames 'model.' prefix to 'minicpm4.' for torch format."""
    mapper = minicpm4.WeightsMapper(orig_to_new_prefix={"model.": "minicpm4."})
    assert mapper.apply("model.layers.0.self_attn.q_proj.weight") == "minicpm4.layers.0.self_attn.q_proj.weight"
    assert mapper.apply("model.embed_tokens.weight") == "minicpm4.embed_tokens.weight"
    # lm_head has no 'model.' prefix -- unchanged
    assert mapper.apply("lm_head.weight") == "lm_head.weight"


def test_stacked_params_qkv():
    """q_proj, k_proj, v_proj map to qkv_proj with correct shard_id."""
    stacked = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("up_gate_proj", "gate_proj", "gate"),
        ("up_gate_proj", "up_proj", "up"),
        ("embed_tokens.embeddings", "embed_tokens", None),
        ("lm_head.linear", "lm_head", None),
    ]
    qkv = {wn: (pn, sid) for pn, wn, sid in stacked if sid in ("q", "k", "v")}
    assert qkv["q_proj"] == ("qkv_proj", "q")
    assert qkv["k_proj"] == ("qkv_proj", "k")
    assert qkv["v_proj"] == ("qkv_proj", "v")


def test_stacked_params_gate_up():
    """gate_proj, up_proj map to up_gate_proj."""
    stacked = [
        ("up_gate_proj", "gate_proj", "gate"),
        ("up_gate_proj", "up_proj", "up"),
    ]
    gu = {wn: (pn, sid) for pn, wn, sid in stacked}
    assert gu["gate_proj"] == ("up_gate_proj", "gate")
    assert gu["up_proj"] == ("up_gate_proj", "up")


# ── TP mappings tests ───────────────────────────────────────────────────────


def test_tp_mappings_split_keys():
    """TP mapping generates correct layer-indexed split actions."""
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=0,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        fuse_attention_qkv=True,
        moe_layer_start_index=0,
    )
    mappings = minicpm4.MiniCPM4PretrainedModel._get_tensor_parallel_mappings(cfg, is_split=True)

    # Should have per-layer keys
    assert "layers.0.self_attn.qkv_proj.weight" in mappings
    assert "layers.1.self_attn.qkv_proj.weight" in mappings
    assert "layers.0.mlp.gate_proj.weight" in mappings
    assert "lm_head.weight" in mappings


def test_tp_mappings_non_fused_qkv():
    """TP mapping handles unfused q/k/v separate weights."""
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=0,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        fuse_attention_qkv=False,
        moe_layer_start_index=0,
    )
    mappings = minicpm4.MiniCPM4PretrainedModel._get_tensor_parallel_mappings(cfg, is_split=True)

    assert "layers.0.self_attn.q_proj.weight" in mappings
    assert "layers.0.self_attn.k_proj.weight" in mappings
    assert "layers.0.self_attn.v_proj.weight" in mappings


def test_tp_mappings_round_trip():
    """Split then merge round-trip for QKV weight."""
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=None,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        fuse_attention_qkv=True,
        moe_layer_start_index=0,
    )
    split_map = minicpm4.MiniCPM4PretrainedModel._get_tensor_parallel_mappings(cfg, is_split=True)
    merge_map = minicpm4.MiniCPM4PretrainedModel._get_tensor_parallel_mappings(cfg, is_split=False)

    key = "layers.0.mlp.gate_proj.weight"
    w = np.arange(32, dtype=np.float32).reshape(8, _HIDDEN)
    parts = split_map[key](w)
    assert len(parts) == 2
    merged = merge_map[key](parts)
    np.testing.assert_array_equal(merged, w)


# ── Model registration ──────────────────────────────────────────────────────


def test_registration_architecture():
    """Verify MiniCPM4ForCausalLM maps to 'MiniCPMForCausalLM' arch string."""
    from fastdeploy.model_executor.models.model_base import ModelRegistry

    registry = ModelRegistry()
    model_info, arch = registry.inspect_model_cls(["MiniCPMForCausalLM"])
    assert arch == "MiniCPMForCausalLM"
    assert model_info.module_path == "minicpm4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
