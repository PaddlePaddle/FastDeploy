#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import types

import paddle
import pytest

from fastdeploy.config import LoadConfig
from fastdeploy.model_executor.model_loader import dummy_loader as dummy_loader_module
from fastdeploy.model_executor.model_loader.dummy_loader import DummyModelLoader


class _FakeParam:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.value = None

    def set_value(self, tensor):
        self.value = tensor


class _FakeModel:
    def __init__(self, named_params):
        self._named_params = named_params

    def named_parameters(self):
        return list(self._named_params)


class _DummyModel(paddle.nn.Layer):
    def __init__(self, _):
        super().__init__()
        self.linear = paddle.nn.Linear(2, 2)


def _make_loader():
    return DummyModelLoader(LoadConfig(args={}))


def _make_fd_config(
    *,
    architectures,
    convert_type="none",
    dynamic_load_weight=False,
    model_type="mtp",
):
    model_config = types.SimpleNamespace(
        architectures=architectures,
        convert_type=convert_type,
        enable_cache=False,
    )
    load_config = types.SimpleNamespace(dynamic_load_weight=dynamic_load_weight)
    speculative_config = types.SimpleNamespace(model_type=model_type)
    return types.SimpleNamespace(
        model_config=model_config,
        load_config=load_config,
        speculative_config=speculative_config,
        quant_config=None,
    )


def test_dummy_loader_initialization_basic():
    loader = _make_loader()
    params = [
        ("none.param", None),
        ("linear.weight", _FakeParam([2, 3], paddle.float32)),
        ("counter", _FakeParam([4], paddle.int64)),
        ("empty", _FakeParam([0], paddle.float32)),
    ]

    model = _FakeModel(params)
    loader._initialize_dummy_weights(model)

    params_map = dict(model.named_parameters())
    float_param = params_map["linear.weight"]
    int_param = params_map["counter"]
    empty_param = params_map["empty"]

    assert float_param.value is not None
    assert list(float_param.value.shape) == [2, 3]
    assert float_param.value.dtype == paddle.float32
    assert int_param.value is not None
    assert bool(paddle.all(int_param.value == 0))
    assert empty_param.value is None


def test_dummy_loader_initialization_nonzero_for_floats():
    loader = _make_loader()
    model = _FakeModel([("linear.weight", _FakeParam([4, 4], paddle.float32))])
    loader._initialize_dummy_weights(model, low=-0.5, high=0.5)

    weight = model.named_parameters()[0][1].value
    assert weight is not None
    assert bool(paddle.any(weight != 0))


def test_dummy_loader_download_model_noop():
    loader = _make_loader()
    loader.download_model(model_config=None)


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("mtp", "Ernie5MTPForCausalLMRL"),
        ("not_mtp", "Ernie5MoeForCausalLMRL"),
    ],
)
def test_dummy_loader_load_model_dynamic_arch(monkeypatch, model_type, expected_arch):
    seen = {"arch": None}

    def _get_class(arch):
        seen["arch"] = arch
        return _DummyModel

    fd_config = _make_fd_config(
        architectures=["Ernie5ForCausalLM"],
        dynamic_load_weight=True,
        model_type=model_type,
    )

    monkeypatch.setitem(sys.modules, "fastdeploy.rl", types.ModuleType("fastdeploy.rl"))
    monkeypatch.setattr(dummy_loader_module.ModelRegistry, "get_class", _get_class)
    monkeypatch.setattr(dummy_loader_module, "process_final_after_loading", lambda *_: None)

    loader = _make_loader()
    loader.load_model(fd_config=fd_config)
    assert seen["arch"] == expected_arch


def test_dummy_loader_load_model_convert_paths(monkeypatch):
    monkeypatch.setattr(dummy_loader_module.ModelRegistry, "get_class", lambda _: _DummyModel)
    monkeypatch.setattr(dummy_loader_module, "process_final_after_loading", lambda *_: None)
    monkeypatch.setattr(dummy_loader_module, "as_embedding_model", lambda model_cls: model_cls)

    loader = _make_loader()

    fd_config_embed = _make_fd_config(architectures=["FakeArch"], convert_type="embed")
    model = loader.load_model(fd_config=fd_config_embed)
    assert isinstance(model, _DummyModel)

    fd_config_invalid = _make_fd_config(architectures=["FakeArch"], convert_type="invalid")
    with pytest.raises(AssertionError):
        loader.load_model(fd_config=fd_config_invalid)
