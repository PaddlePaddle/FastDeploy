"""
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

import sys
import types
from types import SimpleNamespace

import numpy as np
import paddle
import pytest
from paddleformers.transformers import PretrainedModel
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.conversion_utils import split_or_merge_func
from paddleformers.utils.log import logger as pf_logger
from fastdeploy.model_executor.models import ernie4_5_mtp

_PADDLEFORMERS_IMPORTS = (PretrainedModel, PretrainedConfig, split_or_merge_func, pf_logger)


class _StubRMSNorm(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return (x,)

    def load_state_dict(self, _state_dict):
        self.load_state_dict_called = True


class _StubEHProjection(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, x):
        return x[:, : x.shape[-1] // 2]

    def load_state_dict(self, _state_dict):
        self.load_state_dict_called = True


class _StubDecoderLayer(paddle.nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_state_dict_called = False

    def forward(self, _forward_meta, hidden_states, _residual):
        return hidden_states + 1, None

    def load_state_dict(self, _state_dict):
        self.load_state_dict_called = True


class _StubEmbedTokens:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.calls = []

    def __call__(self, *, ids_remove_padding):
        self.calls.append(ids_remove_padding)
        batch = int(ids_remove_padding.shape[0])
        return paddle.zeros([batch, self.hidden_size], dtype="float32")


class _StubFinalNorm(paddle.nn.Layer):
    def __init__(self, fd_config, is_last_norm=True):
        super().__init__()
        self.fd_config = fd_config
        self.is_last_norm = is_last_norm
        self.allgather_called = False

    def forward(self, hidden_states, residual=None, forward_meta=None):
        return (hidden_states,)

    def allgather(self, hidden_states, _total):
        self.allgather_called = True
        return hidden_states + 1




def _make_fd_config(hidden_size=4, num_layers=2, use_sequence_parallel_moe=True):
    model_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        rms_norm_eps=1e-5,
        pretrained_config=SimpleNamespace(prefix_name="ernie"),
        moe_layer_start_index=0,
        ori_vocab_size=3,
    )
    parallel_config = SimpleNamespace(use_sequence_parallel_moe=use_sequence_parallel_moe)
    graph_opt_config = SimpleNamespace(graph_opt_level=0, use_cudagraph=False)
    fd_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        graph_opt_config=graph_opt_config,
    )

    sharing_model = SimpleNamespace()
    sharing_model.ernie = SimpleNamespace()
    sharing_model.ernie.embed_tokens = _StubEmbedTokens(hidden_size)
    sharing_model.ernie.norm = _StubFinalNorm(fd_config)
    sharing_model.lm_head = lambda x: x

    fd_config.speculative_config = SimpleNamespace(sharing_model=sharing_model)
    return fd_config


@pytest.fixture()
def ernie_mtp(monkeypatch):
    monkeypatch.setattr(ernie4_5_mtp, "RMSNorm", _StubRMSNorm)
    monkeypatch.setattr(ernie4_5_mtp, "ParallelEHProjection", _StubEHProjection)
    monkeypatch.setattr(ernie4_5_mtp, "Ernie4_5_DecoderLayer", _StubDecoderLayer)
    return ernie4_5_mtp


def test_arch_and_init_weight():
    assert ernie4_5_mtp.Ernie4_5_MTPPretrainedModel.arch_name() == "Ernie4_5_MTPForCausalLM"
    model = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel()
    assert model._init_weight(layer=None) is None


def test_tp_mappings_gqa_split_merge_numpy():
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=None,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        num_hidden_layers=2,
        moe_layer_start_index=1,
    )

    split_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)
    split_fn = split_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]
    weight = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    parts = split_fn(weight)

    assert len(parts) == 2
    assert all(part.shape == (3, 8) for part in parts)

    merge_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=False)
    merge_fn = merge_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]
    merged = merge_fn(parts)

    assert np.array_equal(merged, weight)


def test_tp_mappings_gqa_split_with_rank_1d():
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=0,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        num_hidden_layers=1,
        moe_layer_start_index=0,
    )

    split_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)
    split_fn = split_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]
    weight = np.arange(16, dtype=np.float32)
    part = split_fn(weight)

    assert part.shape == (8,)
    expected = np.array([0, 1, 2, 3, 8, 9, 12, 13], dtype=np.float32)
    assert np.array_equal(part, expected)


def test_tp_mappings_non_gqa_split():
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=None,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=8,
        num_hidden_layers=1,
        moe_layer_start_index=0,
    )

    split_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)
    split_fn = split_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]
    weight = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    parts = split_fn(weight)

    assert len(parts) == 2
    assert all(part.shape == (3, 4) for part in parts)


def test_tp_mappings_merge_paddle_gpu_copy(monkeypatch):
    config = SimpleNamespace(
        tensor_model_parallel_size=2,
        tensor_parallel_rank=None,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        num_hidden_layers=1,
        moe_layer_start_index=0,
    )

    merge_map = ernie4_5_mtp.Ernie4_5_MTPPretrainedModel._get_tensor_parallel_mappings(config, is_split=False)
    merge_fn = merge_map["ernie.mtp_block.0.self_attn.qkv_proj.weight"]

    class _FakePlace:
        def is_gpu_place(self):
            return True

    class _FakeTensor:
        def __init__(self):
            self.place = _FakePlace()
            self.copy_args = None

        def _copy_to(self, place, blocking):
            self.copy_args = (place, blocking)
            return self

    fake_tensor = _FakeTensor()

    def _fake_concat(_items, axis=-1):
        return fake_tensor

    class _PinnedPlace:
        pass

    monkeypatch.setattr(ernie4_5_mtp.paddle, "concat", _fake_concat)
    monkeypatch.setattr(ernie4_5_mtp.paddle, "CUDAPinnedPlace", _PinnedPlace)

    weight_list = [
        paddle.to_tensor(np.ones((1, 8), dtype=np.float32)),
        paddle.to_tensor(np.zeros((1, 8), dtype=np.float32)),
    ]
    result = merge_fn(weight_list)

    assert result is fake_tensor
    assert isinstance(fake_tensor.copy_args[0], _PinnedPlace)
    assert fake_tensor.copy_args[1] is False


def test_mtp_model_forward_and_allgather(ernie_mtp):
    fd_config = _make_fd_config(hidden_size=4, num_layers=2, use_sequence_parallel_moe=True)
    model = ernie_mtp.Ernie4_5_MTPModel(fd_config=fd_config)

    ids = paddle.to_tensor([1, 2], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    forward_meta = SimpleNamespace(ids_remove_padding=ids)

    output = model(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=forward_meta)

    assert output.shape == (2, 4)
    assert paddle.all(output == 3).item()
    assert fd_config.speculative_config.sharing_model.ernie.norm.allgather_called is True


def test_mtp_model_load_state_dict_calls_components(ernie_mtp):
    fd_config = _make_fd_config(hidden_size=4, num_layers=2, use_sequence_parallel_moe=False)
    model = ernie_mtp.Ernie4_5_MTPModel(fd_config=fd_config)

    model.load_state_dict({"weight": np.zeros([1], dtype=np.float32)})

    assert model.enorm.load_state_dict_called is True
    assert model.hnorm.load_state_dict_called is True
    assert model.eh_proj.load_state_dict_called is True
    assert all(layer.load_state_dict_called for layer in model.mtp_block)


def test_causallm_compute_logits_and_forward(ernie_mtp):
    fd_config = _make_fd_config(hidden_size=4, num_layers=1, use_sequence_parallel_moe=False)
    model = ernie_mtp.Ernie4_5_MTPForCausalLM(fd_config)

    ids = paddle.to_tensor([0, 1], dtype="int64")
    prev = paddle.ones([2, 4], dtype="float32")
    forward_meta = SimpleNamespace(ids_remove_padding=ids)

    hidden_states = model(ids_remove_padding=ids, previous_hidden_states=prev, forward_meta=forward_meta)
    logits = model.compute_logits(hidden_states.astype("float16"), forward_meta)

    assert logits.dtype == paddle.float32
    assert logits.shape[1] == 4
    assert paddle.isinf(logits[:, fd_config.model_config.ori_vocab_size :]).all().item()


def test_causallm_set_state_dict_calls_ernie():
    class _StubErnie:
        def __init__(self):
            self.called = False

        def load_state_dict(self, _state_dict):
            self.called = True

    model = ernie4_5_mtp.Ernie4_5_MTPForCausalLM.__new__(ernie4_5_mtp.Ernie4_5_MTPForCausalLM)
    model.ernie = _StubErnie()

    model.set_state_dict({"weight": np.zeros([1], dtype=np.float32)})

    assert model.ernie.called is True


def test_causallm_empty_input_forward_calls_fused_moe():
    class _StubFusedMLP:
        def __init__(self):
            self.calls = []

        def fused_moe(self, hidden_states, forward_meta):
            self.calls.append((hidden_states.shape, forward_meta))

    class _StubLayer:
        def __init__(self):
            self.mlp = _StubFusedMLP()

    model = ernie4_5_mtp.Ernie4_5_MTPForCausalLM.__new__(ernie4_5_mtp.Ernie4_5_MTPForCausalLM)
    model.fd_config = SimpleNamespace(
        model_config=SimpleNamespace(moe_layer_start_index=1, num_hidden_layers=3, hidden_size=4)
    )
    model.ernie = SimpleNamespace(layers=[_StubLayer(), _StubLayer(), _StubLayer()])

    forward_meta = SimpleNamespace()
    model.empty_input_forward(forward_meta)

    assert model.ernie.layers[0].mlp.calls == []
    assert len(model.ernie.layers[1].mlp.calls) == 1
    assert len(model.ernie.layers[2].mlp.calls) == 1


def test_causallm_load_weights_uses_remap(monkeypatch):
    moe_module = types.ModuleType("fastdeploy.model_executor.models.ernie4_5_moe")

    class _StubMoe:
        calls = []

        @staticmethod
        def load_weights(self, weights):
            _StubMoe.calls.append((self, list(weights)))

    moe_module.Ernie4_5_MoeForCausalLM = _StubMoe
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.models.ernie4_5_moe", moe_module)

    utils_module = types.ModuleType("fastdeploy.model_executor.utils")

    def _remap_weight_keys(weights_iterator, mapping):
        _remap_weight_keys.mapping = mapping
        return list(weights_iterator)

    utils_module.remap_weight_keys = _remap_weight_keys
    monkeypatch.setitem(sys.modules, "fastdeploy.model_executor.utils", utils_module)

    model = ernie4_5_mtp.Ernie4_5_MTPForCausalLM.__new__(ernie4_5_mtp.Ernie4_5_MTPForCausalLM)
    weights = [("key", np.zeros([1], dtype=np.float32))]

    model.load_weights(iter(weights))

    assert _StubMoe.calls
    assert _StubMoe.calls[0][1] == weights
    assert "mtp_linear_proj.0" in _remap_weight_keys.mapping


def test_causallm_name():
    assert ernie4_5_mtp.Ernie4_5_MTPForCausalLM.name() == "Ernie4_5_MTPForCausalLM"
