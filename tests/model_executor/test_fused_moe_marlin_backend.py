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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import paddle
import pytest


MODULE_NAME = "fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend"
MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fastdeploy"
    / "model_executor"
    / "layers"
    / "moe"
    / "fused_moe_marlin_backend.py"
)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_marlin_backend(monkeypatch):
    fastdeploy_mod = _package("fastdeploy")
    model_executor_mod = _package("fastdeploy.model_executor")
    layers_mod = _package("fastdeploy.model_executor.layers")
    ops_mod = _package("fastdeploy.model_executor.ops")
    gpu_mod = types.ModuleType("fastdeploy.model_executor.ops.gpu")
    moe_pkg_mod = _package("fastdeploy.model_executor.layers.moe")
    moe_mod = types.ModuleType("fastdeploy.model_executor.layers.moe.moe")
    quant_pkg_mod = _package("fastdeploy.model_executor.layers.quantization")
    quant_base_mod = types.ModuleType("fastdeploy.model_executor.layers.quantization.quant_base")

    class QuantMethodBase:
        pass

    quant_base_mod.QuantMethodBase = QuantMethodBase
    gpu_mod.MoeWna16MarlinGemmApi = Mock()
    gpu_mod.tritonmoe_preprocess_func = Mock()
    gpu_mod.moe_topk_select = Mock()
    gpu_mod.gptq_marlin_repack = Mock()
    moe_mod.get_moe_scores = Mock()

    fastdeploy_mod.model_executor = model_executor_mod
    model_executor_mod.layers = layers_mod
    model_executor_mod.ops = ops_mod
    layers_mod.moe = moe_pkg_mod
    layers_mod.quantization = quant_pkg_mod
    ops_mod.gpu = gpu_mod

    modules = {
        "fastdeploy": fastdeploy_mod,
        "fastdeploy.model_executor": model_executor_mod,
        "fastdeploy.model_executor.layers": layers_mod,
        "fastdeploy.model_executor.layers.moe": moe_pkg_mod,
        "fastdeploy.model_executor.layers.moe.moe": moe_mod,
        "fastdeploy.model_executor.layers.quantization": quant_pkg_mod,
        "fastdeploy.model_executor.layers.quantization.quant_base": quant_base_mod,
        "fastdeploy.model_executor.ops": ops_mod,
        "fastdeploy.model_executor.ops.gpu": gpu_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module, gpu_mod, moe_mod


class _DummyMoELayer(paddle.nn.Layer):
    def __init__(self, hidden_size=32, moe_intermediate_size=16, num_local_experts=2):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = 2
        self.topk_method = "topk"
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.renormalize = True
        self.gate_correction_bias = paddle.zeros([num_local_experts], dtype="float32")

    def extract_moe_ffn_weights(self, state_dict):
        return state_dict["up"], state_dict["down"], None, None


def test_scale_permutations_are_stable(monkeypatch):
    marlin, _, _ = _load_marlin_backend(monkeypatch)

    scale_perm, scale_perm_single = marlin.get_scale_perms()

    assert len(scale_perm) == 64
    assert len(scale_perm_single) == 32
    assert scale_perm[:10] == [0, 8, 16, 24, 32, 40, 48, 56, 1, 9]
    assert scale_perm[-8:] == [7, 15, 23, 31, 39, 47, 55, 63]
    assert scale_perm_single[:16] == [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27]


def test_marlin_permute_scales_grouped_and_single_channel(monkeypatch):
    marlin, _, _ = _load_marlin_backend(monkeypatch)
    scale_perm, scale_perm_single = marlin.get_scale_perms()

    grouped = paddle.arange(128, dtype="int64").reshape([2, 64])
    grouped_out = marlin.marlin_permute_scales(grouped, size_k=128, size_n=16, group_size=64)
    grouped_expected = grouped.reshape([-1, len(scale_perm)])[:, scale_perm].reshape([-1, 16])
    np.testing.assert_array_equal(grouped_out.numpy(), grouped_expected.numpy())

    per_channel = paddle.arange(64, dtype="int64").reshape([2, 32])
    per_channel_out = marlin.marlin_permute_scales(per_channel, size_k=32, size_n=32, group_size=-1)
    per_channel_expected = per_channel.reshape([-1, len(scale_perm_single)])[:, scale_perm_single].reshape([-1, 32])
    np.testing.assert_array_equal(per_channel_out.numpy(), per_channel_expected.numpy())


def test_marlin_moe_permute_scales_handles_each_expert(monkeypatch):
    marlin, _, _ = _load_marlin_backend(monkeypatch)
    _, scale_perm_single = marlin.get_scale_perms()

    scales = paddle.arange(128, dtype="float32").reshape([2, 2, 32])
    out = marlin.marlin_moe_permute_scales(scales, size_k=32, size_n=32, group_size=-1)

    expected = paddle.stack(
        [expert.reshape([-1, len(scale_perm_single)])[:, scale_perm_single].reshape([2, 32]) for expert in scales],
        axis=0,
    )
    assert list(out.shape) == [2, 2, 32]
    np.testing.assert_array_equal(out.numpy(), expected.numpy())


def test_gptq_marlin_moe_repack_invokes_kernel_per_expert(monkeypatch):
    marlin, gpu_mod, _ = _load_marlin_backend(monkeypatch)
    calls = []

    def fake_repack(weight, perm, size_k, size_n, num_bits):
        calls.append((weight.numpy().copy(), perm.numpy().copy(), size_k, size_n, num_bits))
        return paddle.full([size_k // 16, size_n * (num_bits // 2)], len(calls), dtype=weight.dtype)

    gpu_mod.gptq_marlin_repack = fake_repack
    q_weight = paddle.arange(32, dtype="int32").reshape([2, 2, 8])
    perm = paddle.arange(6, dtype="int32").reshape([2, 3])

    out = marlin.gptq_marlin_moe_repack(q_weight, perm, size_k=32, size_n=4, num_bits=4)

    assert len(calls) == 2
    assert list(out.shape) == [2, 2, 8]
    np.testing.assert_array_equal(out[0].numpy(), np.ones([2, 8], dtype=np.int32))
    np.testing.assert_array_equal(out[1].numpy(), np.full([2, 8], 2, dtype=np.int32))
    np.testing.assert_array_equal(calls[0][0], q_weight[0].numpy())
    np.testing.assert_array_equal(calls[1][1], perm[1].numpy())

    with pytest.raises(AssertionError):
        marlin.gptq_marlin_moe_repack(q_weight, perm, size_k=17, size_n=4, num_bits=4)


def test_create_weights_registers_expected_marlin_parameters(monkeypatch):
    marlin, _, _ = _load_marlin_backend(monkeypatch)
    layer = _DummyMoELayer(hidden_size=32, moe_intermediate_size=16, num_local_experts=2)
    method = marlin.MarlinWeightOnlyMoEMethod()

    method.create_weights(layer)

    assert list(layer.up_gate_proj_weight.shape) == [2, 2, 64]
    assert list(layer.down_proj_weight.shape) == [2, 1, 64]
    assert list(layer.up_gate_proj_weight_scale.shape) == [2, 1, 32]
    assert list(layer.down_proj_weight_scale.shape) == [2, 1, 32]
    assert layer.up_gate_proj_weight.dtype == paddle.int32
    assert layer.down_proj_weight.dtype == paddle.int32
    assert layer.up_gate_proj_weight_scale.dtype == paddle.float32
    assert layer.down_proj_weight_scale.dtype == paddle.float32


def test_process_loaded_weights_quantizes_and_sets_parameters(monkeypatch):
    marlin, gpu_mod, _ = _load_marlin_backend(monkeypatch)

    def fake_repack(weight, _perm, size_k, size_n, num_bits):
        del weight
        return paddle.full([size_k // 16, size_n * (num_bits // 2)], 3, dtype="int32")

    gpu_mod.gptq_marlin_repack = fake_repack
    layer = _DummyMoELayer(hidden_size=32, moe_intermediate_size=16, num_local_experts=2)
    method = marlin.MarlinWeightOnlyMoEMethod()
    method.create_weights(layer)

    up_weights = [
        paddle.arange(1, 32 * 32 + 1, dtype="float32").reshape([32, 32]) + expert_idx
        for expert_idx in range(layer.num_local_experts)
    ]
    down_weights = [
        paddle.arange(1, 16 * 32 + 1, dtype="float32").reshape([16, 32]) + expert_idx
        for expert_idx in range(layer.num_local_experts)
    ]

    method.process_loaded_weights(layer, {"up": up_weights, "down": down_weights})

    assert list(layer.up_gate_proj_weight.shape) == [2, 2, 64]
    assert list(layer.down_proj_weight.shape) == [2, 1, 64]
    assert paddle.all(layer.up_gate_proj_weight == 3).item()
    assert paddle.all(layer.down_proj_weight == 3).item()
    assert paddle.all(paddle.isfinite(layer.up_gate_proj_weight_scale)).item()
    assert paddle.all(paddle.isfinite(layer.down_proj_weight_scale)).item()

    with pytest.raises(AssertionError):
        method.process_loaded_weights(layer, {"up": [paddle.ones([4, 4])], "down": down_weights})


def test_apply_uses_marlin_gemm_and_hook_for_topk_path(monkeypatch):
    marlin, gpu_mod, _ = _load_marlin_backend(monkeypatch)
    layer = SimpleNamespace(
        top_k=2,
        moe_intermediate_size=8,
        hidden_size=16,
        num_experts=4,
        topk_method="topk",
        gate_correction_bias=paddle.zeros([4], dtype="float32"),
        up_gate_proj_weight=paddle.ones([4, 1, 32], dtype="int32"),
        up_gate_proj_weight_scale=paddle.ones([4, 1, 16], dtype="float32"),
        down_proj_weight=paddle.ones([4, 1, 32], dtype="int32"),
        down_proj_weight_scale=paddle.ones([4, 1, 16], dtype="float32"),
    )
    method = marlin.MarlinWeightOnlyMoEMethod()
    x = paddle.ones([3, layer.hidden_size], dtype="float32")
    topk_ids = paddle.to_tensor([[0, 1], [1, 2], [2, 3]], dtype="int32")
    topk_weights = paddle.ones([3, layer.top_k], dtype="float32")
    hook = Mock()

    gpu_mod.moe_topk_select.return_value = (topk_ids, topk_weights)
    marlin.tritonmoe_preprocess_func = Mock(
        return_value=(
            paddle.arange(6, dtype="int32"),
            paddle.arange(layer.num_experts, dtype="int32"),
            paddle.to_tensor([6], dtype="int32"),
        )
    )
    marlin.MoeWna16MarlinGemmApi = Mock(
        side_effect=[
            (paddle.ones([6, layer.moe_intermediate_size * 2], dtype="float32"),),
            (paddle.ones([6, layer.hidden_size], dtype="float32"),),
        ]
    )

    out = method.apply(layer, x, gate=lambda _x: paddle.ones([3, layer.num_experts]), topk_ids_hookfunc=hook)

    assert list(out.shape) == [3, layer.hidden_size]
    hook.assert_called_once()
    np.testing.assert_array_equal(hook.call_args.kwargs["topk_ids"].numpy(), topk_ids.numpy())
    assert marlin.MoeWna16MarlinGemmApi.call_count == 2
    first_call = marlin.MoeWna16MarlinGemmApi.call_args_list[0].kwargs
    second_call = marlin.MoeWna16MarlinGemmApi.call_args_list[1].kwargs
    assert first_call["top_k"] == layer.top_k
    assert first_call["mul_topk_weights"] is False
    assert first_call["size_m"] == x.shape[0]
    assert first_call["size_n"] == layer.moe_intermediate_size * 2
    assert first_call["size_k"] == layer.hidden_size
    assert second_call["top_k"] == 1
    assert second_call["mul_topk_weights"] is True
    assert second_call["size_m"] == x.shape[0] * layer.top_k
    assert second_call["size_n"] == layer.hidden_size
    assert second_call["size_k"] == layer.moe_intermediate_size


def test_apply_uses_noaux_tc_score_path(monkeypatch):
    marlin, gpu_mod, moe_mod = _load_marlin_backend(monkeypatch)
    layer = SimpleNamespace(
        top_k=2,
        moe_intermediate_size=8,
        hidden_size=16,
        num_experts=4,
        topk_method="noaux_tc",
        n_group=2,
        topk_group=1,
        routed_scaling_factor=0.5,
        renormalize=False,
        gate_correction_bias=paddle.arange(4, dtype="float32"),
        up_gate_proj_weight=paddle.ones([4, 1, 32], dtype="int32"),
        up_gate_proj_weight_scale=paddle.ones([4, 1, 16], dtype="float32"),
        down_proj_weight=paddle.ones([4, 1, 32], dtype="int32"),
        down_proj_weight_scale=paddle.ones([4, 1, 16], dtype="float32"),
    )
    method = marlin.MarlinWeightOnlyMoEMethod()
    x = paddle.ones([2, layer.hidden_size], dtype="float32")
    gate_out = paddle.arange(8, dtype="float32").reshape([2, layer.num_experts])
    topk_ids = paddle.to_tensor([[0, 2], [1, 3]], dtype="int32")
    topk_weights = paddle.to_tensor([[0.7, 0.3], [0.6, 0.4]], dtype="float32")
    hook = Mock()

    moe_mod.get_moe_scores.return_value = (None, topk_weights, topk_ids)
    marlin.tritonmoe_preprocess_func = Mock(
        return_value=(
            paddle.arange(4, dtype="int32"),
            paddle.arange(layer.num_experts, dtype="int32"),
            paddle.to_tensor([4], dtype="int32"),
        )
    )
    marlin.MoeWna16MarlinGemmApi = Mock(
        side_effect=[
            (paddle.ones([4, layer.moe_intermediate_size * 2], dtype="float32"),),
            (paddle.ones([4, layer.hidden_size], dtype="float32"),),
        ]
    )

    out = method.apply(layer, x, gate=lambda _x: gate_out, topk_ids_hookfunc=hook)

    assert list(out.shape) == [2, layer.hidden_size]
    gpu_mod.moe_topk_select.assert_not_called()
    moe_mod.get_moe_scores.assert_called_once()
    score_args = moe_mod.get_moe_scores.call_args.args
    np.testing.assert_array_equal(score_args[0].numpy(), gate_out.numpy())
    assert score_args[1:] == (
        layer.n_group,
        layer.topk_group,
        layer.top_k,
        layer.routed_scaling_factor,
        layer.gate_correction_bias,
        layer.renormalize,
    )
    hook.assert_called_once()
    np.testing.assert_array_equal(hook.call_args.kwargs["topk_ids"].numpy(), topk_ids.numpy())
    marlin.tritonmoe_preprocess_func.assert_called_once()
    preprocess_args = marlin.tritonmoe_preprocess_func.call_args.args
    np.testing.assert_array_equal(preprocess_args[0].numpy(), topk_ids.numpy())
    # apply() picks the first m in [8, 16, 32, 48, 64] where tokens * top_k / experts / m < 0.9.
    assert preprocess_args[1:] == (layer.num_experts, 8)
