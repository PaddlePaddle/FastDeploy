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

from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import paddle
import pytest

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda *args, **kwargs: None

from fastdeploy.model_executor.layers.attention import flash_attn_backend
from fastdeploy.model_executor.layers.batch_invariant_ops import batch_invariant_ops
from fastdeploy.model_executor.layers.moe import ep as ep_module
from fastdeploy.model_executor.layers.quantization import fp8_utils


def _install_package(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_init_flash_attn_version_enables_cutlass_compat(monkeypatch: pytest.MonkeyPatch):
    fake_flashmask_attention = object()
    flash_mask_pkg = _install_package(monkeypatch, "flash_mask")
    cute_pkg = _install_package(monkeypatch, "flash_mask.cute")
    interface_module = _install_module(
        monkeypatch,
        "flash_mask.cute.interface",
        flashmask_attention=fake_flashmask_attention,
    )
    flash_mask_pkg.cute = cute_pkg
    cute_pkg.interface = interface_module

    enable_compat = mock.Mock()
    monkeypatch.setattr(paddle, "enable_compat", enable_compat)
    monkeypatch.setattr(flash_attn_backend.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(flash_attn_backend, "get_sm_version", lambda: 100)
    monkeypatch.setattr(flash_attn_backend, "FLASH_ATTN_VERSION", None)
    monkeypatch.setattr(flash_attn_backend, "flashmask_attention_v4", None)

    flash_attn_backend.init_flash_attn_version()

    enable_compat.assert_called_once_with(scope={"cutlass"})
    assert flash_attn_backend.FLASH_ATTN_VERSION == 4
    assert flash_attn_backend.flashmask_attention_v4 is fake_flashmask_attention


def test_load_deep_gemm_enables_deep_gemm_compat(monkeypatch: pytest.MonkeyPatch):
    paddlefleet_pkg = _install_package(monkeypatch, "paddlefleet")
    ops_pkg = _install_package(monkeypatch, "paddlefleet.ops")
    deep_gemm_module = _install_module(monkeypatch, "paddlefleet.ops.deep_gemm")
    paddlefleet_pkg.ops = ops_pkg
    ops_pkg.deep_gemm = deep_gemm_module

    enable_compat = mock.Mock()
    monkeypatch.setattr(paddle, "enable_compat", enable_compat)
    monkeypatch.setattr(fp8_utils.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(fp8_utils, "get_sm_version", lambda: 100)

    loaded = fp8_utils.load_deep_gemm()

    enable_compat.assert_called_once_with(scope={"deep_gemm"})
    assert loaded is deep_gemm_module


def test_load_deep_ep_enables_deep_ep_compat(monkeypatch: pytest.MonkeyPatch):
    paddlefleet_pkg = _install_package(monkeypatch, "paddlefleet")
    ops_pkg = _install_package(monkeypatch, "paddlefleet.ops")
    deep_ep_module = _install_module(monkeypatch, "paddlefleet.ops.deep_ep")
    paddlefleet_pkg.ops = ops_pkg
    ops_pkg.deep_ep = deep_ep_module

    enable_compat = mock.Mock()
    monkeypatch.setattr(paddle, "enable_compat", enable_compat)
    monkeypatch.setattr(ep_module.envs, "FD_USE_PFCC_DEEP_EP", True)

    loaded = ep_module.load_deep_ep()

    enable_compat.assert_called_once_with(scope={"deep_ep"})
    assert loaded is deep_ep_module


def test_enable_batch_invariant_mode_raises_when_enable_compat_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(batch_invariant_ops, "_batch_invariant_MODE", False)
    monkeypatch.setattr(batch_invariant_ops, "paddle", object())

    with pytest.raises(RuntimeError, match=r"paddle\.enable_compat is unavailable\."):
        batch_invariant_ops.enable_batch_invariant_mode()
