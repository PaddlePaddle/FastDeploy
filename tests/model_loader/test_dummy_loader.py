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

import paddle

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


def _make_loader():
    return DummyModelLoader(LoadConfig(args={}))


def test_dummy_loader_initialization_basic():
    loader = _make_loader()
    params = [
        ("linear.weight", _FakeParam([2, 3], paddle.float32)),
        ("counter", _FakeParam([4], paddle.int64)),
        ("empty", _FakeParam([0], paddle.float32)),
    ]

    model_a = _FakeModel(params)
    loader._initialize_dummy_weights(model_a)

    weight_a = model_a.named_parameters()[0][1].value
    assert weight_a is not None
    assert list(weight_a.shape) == [2, 3]
    assert weight_a.dtype == paddle.float32

    int_param = model_a.named_parameters()[1][1]
    assert int_param.value is not None
    assert bool(paddle.all(int_param.value == 0))

    empty_param = model_a.named_parameters()[2][1]
    assert empty_param.value is None


def test_dummy_loader_initialization_nonzero_for_floats():
    loader = _make_loader()
    model = _FakeModel([("linear.weight", _FakeParam([4, 4], paddle.float32))])
    loader._initialize_dummy_weights(model, low=-0.5, high=0.5)

    weight = model.named_parameters()[0][1].value
    assert weight is not None
    assert bool(paddle.any(weight != 0))


def test_dummy_loader_load_model_basic(monkeypatch):
    class _FakeModelConfig:
        architectures = ["FakeArch"]
        convert_type = "none"
        enable_cache = False

    class _FakeLoadConfig:
        dynamic_load_weight = False

    class _FakeSpeculativeConfig:
        model_type = "mtp"

    class _FakeFDConfig:
        model_config = _FakeModelConfig()
        load_config = _FakeLoadConfig()
        speculative_config = _FakeSpeculativeConfig()
        quant_config = None

    class _DummyModel(paddle.nn.Layer):
        def __init__(self, _):
            super().__init__()
            self.linear = paddle.nn.Linear(2, 2)

    called = {"value": False}

    def _process_final_after_loading(*args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(dummy_loader_module.ModelRegistry, "get_class", lambda _: _DummyModel)
    monkeypatch.setattr(dummy_loader_module, "process_final_after_loading", _process_final_after_loading)

    loader = _make_loader()
    model = loader.load_model(fd_config=_FakeFDConfig())

    assert isinstance(model, _DummyModel)
    assert called["value"] is True
