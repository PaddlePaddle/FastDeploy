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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.models import ernie4_5_mtp

# ── Stubs ───────────────────────────────────────────────────────────────────


class _StubRMSNorm(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return (x,)

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubEHProjection(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return x[:, : x.shape[-1] // 2]

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubDecoderLayer(paddle.nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, _meta, hidden_states, _residual):
        return hidden_states + 1, None

    def load_state_dict(self, _sd):
        self.load_state_dict_called = True


class _StubEmbedTokens:
    def __init__(self, h):
        self.hidden_size = h

    def __call__(self, *, ids_remove_padding):
        return paddle.zeros([ids_remove_padding.shape[0], self.hidden_size], "float32")


class _StubFinalNorm(paddle.nn.Layer):
    def __init__(self, fd_config, is_last_norm=True):
        super().__init__()
        self.allgather_called = False
        self.is_last_norm = is_last_norm
        self.fd_config = fd_config

    def forward(self, h, residual=None, forward_meta=None):
        return (h,)

    def allgather(self, h, _total):
        self.allgather_called = True
        return h + 1


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_fd_config(hidden_size=4, num_layers=2, use_sp_moe=True):
    mc = SimpleNamespace(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        rms_norm_eps=1e-5,
        pretrained_config=SimpleNamespace(prefix_name="ernie"),
        moe_layer_start_index=0,
        ori_vocab_size=3,
    )
    fd = SimpleNamespace(
        model_config=mc,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=use_sp_moe),
        graph_opt_config=SimpleNamespace(graph_opt_level=0, use_cudagraph=False),
    )
    sharing = SimpleNamespace()
    sharing.ernie = SimpleNamespace(embed_tokens=_StubEmbedTokens(hidden_size), norm=_StubFinalNorm(fd))
    sharing.lm_head = lambda x: x
    fd.speculative_config = SimpleNamespace(sharing_model=sharing)
    return fd


@pytest.fixture()
def mtp(monkeypatch):
    monkeypatch.setattr(ernie4_5_mtp, "RMSNorm", _StubRMSNorm)
    monkeypatch.setattr(ernie4_5_mtp, "ParallelEHProjection", _StubEHProjection)
    monkeypatch.setattr(ernie4_5_mtp, "Ernie4_5_DecoderLayer", _StubDecoderLayer)
    return ernie4_5_mtp


# ── Tests ───────────────────────────────────────────────────────────────────


def test_tp_mappings():
    """GQA split + merge round-trip for tensor parallel mappings."""
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=None,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        num_hidden_layers=2,
        moe_layer_start_index=1,
    )
    split_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(cfg, is_split=True)
    fn = split_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]
    w = np.arange(48, dtype=np.float32).reshape(3, 16)
    parts = fn(w)
    assert len(parts) == 2 and all(p.shape == (3, 8) for p in parts)

    merge_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(cfg, is_split=False)
    merged = merge_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"](parts)
    assert np.array_equal(merged, w)


def test_model_forward(mtp):
    """MTPModel init, forward with allgather, and load_state_dict."""
    fd = _make_fd_config(hidden_size=4, num_layers=2)
    model = mtp.Ernie4_5_MTPModel(fd_config=fd)

    ids = paddle.to_tensor([1, 2], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    meta = SimpleNamespace(ids_remove_padding=ids)
    out = model(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=meta)
    assert out.shape == (2, 4)
    assert fd.speculative_config.sharing_model.ernie.norm.allgather_called

    model.load_state_dict({"w": np.zeros([1], dtype=np.float32)})
    assert model.enorm.load_state_dict_called
    assert all(l.load_state_dict_called for l in model.mtp_block)


def test_causallm(mtp):
    """CausalLM forward, compute_logits, set_state_dict."""
    fd = _make_fd_config(hidden_size=4, num_layers=1, use_sp_moe=False)
    model = mtp.Ernie4_5_MTPForCausalLM(fd)

    ids = paddle.to_tensor([0, 1], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    meta = SimpleNamespace(ids_remove_padding=ids)
    hidden = model(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=meta)
    logits = model.compute_logits(hidden.astype("float16"), meta)
    assert logits.dtype == paddle.float32
    assert paddle.isinf(logits[:, fd.model_config.ori_vocab_size :]).all().item()

    model.set_state_dict({"w": np.zeros([1], dtype=np.float32)})


def test_load_weights(monkeypatch):
    """Load weights with remap pipeline."""
    moe_mod = types.ModuleType("fastdeploy.model_executor.models.ernie4_5_moe")

    class _Moe:
        calls = []

        @staticmethod
        def load_weights(self, weights):
            _Moe.calls.append(list(weights))

    moe_mod.Ernie4_5_MoeForCausalLM = _Moe
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.models.ernie4_5_moe", moe_mod)

    utils_mod = types.ModuleType("fastdeploy.model_executor.utils")

    def _remap(weights_iter, mapping):
        _remap.mapping = mapping
        return list(weights_iter)

    utils_mod.remap_weight_keys = _remap
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.utils", utils_mod)

    model = ernie4_5_mtp.Ernie4_5_MTPForCausalLM.__new__(ernie4_5_mtp.Ernie4_5_MTPForCausalLM)
    model.load_weights(iter([("key", np.zeros([1], dtype=np.float32))]))
    assert _Moe.calls
    assert "mtp_linear_proj.0" in _remap.mapping


def test_empty_input_forward():
    """Empty batch path for MoE layers."""

    class _StubMLP:
        def __init__(self):
            self.calls = []

        def fused_moe(self, hidden_states=None, forward_meta=None):
            self.calls.append(hidden_states.shape)

    model = ernie4_5_mtp.Ernie4_5_MTPForCausalLM.__new__(ernie4_5_mtp.Ernie4_5_MTPForCausalLM)
    model.fd_config = SimpleNamespace(
        model_config=SimpleNamespace(moe_layer_start_index=1, num_hidden_layers=3, hidden_size=4)
    )
    layers = [SimpleNamespace(mlp=_StubMLP()) for _ in range(3)]
    model.ernie = SimpleNamespace(layers=layers)
    model.empty_input_forward(SimpleNamespace())
    assert layers[0].mlp.calls == []
    assert len(layers[1].mlp.calls) == 1


def test_tp_mappings_non_gqa_and_rank_slice():
    """Cover non-GQA mapping path and rank-selected split branch."""
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=8,
        num_hidden_layers=1,
        moe_layer_start_index=0,
    )
    split_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(cfg, is_split=True)
    key = "ernie.mtp_block.0.self_attn.qkv_proj.weight"
    w = np.arange(48, dtype=np.float32).reshape(3, 16)
    out = split_map[key](w)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 8)


def test_model_forward_without_allgather(mtp):
    """Forward path when sequence parallel allgather is disabled."""
    fd = _make_fd_config(hidden_size=4, num_layers=1, use_sp_moe=False)
    model = mtp.Ernie4_5_MTPModel(fd_config=fd)
    ids = paddle.to_tensor([0, 1], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    meta = SimpleNamespace(ids_remove_padding=ids)
    out = model(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=meta)
    assert out.shape == (2, 4)
    assert not fd.speculative_config.sharing_model.ernie.norm.allgather_called


def test_causallm_name_forward_and_empty_input_range(mtp):
    """Cover name(), forward(), and empty_input_forward no-op range branch."""
    fd = _make_fd_config(hidden_size=4, num_layers=1, use_sp_moe=False)
    model = mtp.Ernie4_5_MTPForCausalLM(fd)
    assert model.name() == "Ernie4_5_MTPForCausalLM"

    ids = paddle.to_tensor([0, 1], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    meta = SimpleNamespace(ids_remove_padding=ids)
    out = model.forward(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=meta)
    assert out.shape == (2, 4)

    # empty_input_forward: start==end should skip fused_moe calls.
    class _StubMLP:
        def __init__(self):
            self.calls = 0

        def fused_moe(self, hidden_states=None, forward_meta=None):
            self.calls += 1

    model.fd_config.model_config.moe_layer_start_index = 1
    model.fd_config.model_config.num_hidden_layers = 1
    model.ernie.layers = [SimpleNamespace(mlp=_StubMLP())]
    model.empty_input_forward(SimpleNamespace())
    assert model.ernie.layers[0].mlp.calls == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
