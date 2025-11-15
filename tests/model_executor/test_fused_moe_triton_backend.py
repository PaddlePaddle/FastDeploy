"""Unit tests for the Triton fused MoE backends.

These tests install lightweight GPU/operator stubs so the real
``fastdeploy.model_executor.layers.moe.fused_moe_triton_backend`` module can be
imported and exercised without CUDA kernels.  The suites cover the weight-only,
wfp8afp8, tensor-wise fp8, and block-wise fp8 quantization helpers to ensure the
most important control-flow branches are validated while keeping the numerics
deterministic and CPU friendly.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import paddle
import pytest

if not hasattr(paddle, "float8_e4m3fn"):
    paddle.float8_e4m3fn = paddle.float16


class _RecordingKernel:
    """Tiny stub that mimics the Triton kernel launch API."""

    def __init__(self):
        self.calls: list[dict] = []

    def __getitem__(self, grid):  # noqa: D401 - behavior mirrors kernel launch
        def _runner(*args, **kwargs):
            output = args[2]
            if isinstance(output, paddle.Tensor):
                # Encode the grid in the tensor so downstream reductions produce
                # deterministic values without relying on the actual kernel.
                output.set_value(paddle.full_like(output, float(len(self.calls) + 1)))
            self.calls.append({"grid": grid, "kwargs": kwargs})

        return _runner


def _fake_topk_select(gate_out, *_args, **_kwargs):
    token_num = gate_out.shape[0]
    top_k = 2
    topk_ids = paddle.tile(paddle.arange(top_k, dtype="int32"), [token_num, 1])
    topk_weights = paddle.full([token_num, top_k], 0.5, dtype=gate_out.dtype)
    return topk_ids, topk_weights


def _fake_preprocess(topk_ids, num_local_experts, _block_size):
    flattened = paddle.reshape(topk_ids, [-1])
    sorted_token_ids = paddle.sort(flattened)
    expert_ids = paddle.arange(flattened.shape[0], dtype="int32") % num_local_experts
    num_tokens_post_padded = paddle.to_tensor(flattened.shape[0], dtype="int32")
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _fake_per_token_quant(x, *_args, **_kwargs):
    return x.astype("float32"), paddle.ones([x.shape[0], 1], dtype="float32")


def _fake_scaled_fp8_quant(x, **_kwargs):
    return x.astype("float32"), paddle.ones([x.shape[0], 1], dtype="float32")


def _fake_per_block_cast_to_fp8(tensor, block_size=None):
    block_m, block_n = block_size or (2, 2)
    rows = (int(tensor.shape[0]) + block_m - 1) // block_m
    cols = (int(tensor.shape[1]) + block_n - 1) // block_n
    scales = paddle.ones([rows, cols], dtype="float32")
    return tensor.astype(paddle.float8_e4m3fn), scales


def _fake_get_moe_scores(gate_out, *_args, **_kwargs):
    topk_ids, topk_weights = _fake_topk_select(gate_out, None, None, None, None)
    return gate_out, topk_weights, topk_ids


@dataclass
class _FakeQuantConfig:
    name_value: str = "wint8"
    is_checkpoint_bf16: bool = False
    weight_block_size: tuple[int, int] = (2, 2)

    def name(self):  # noqa: D401 - mimic FastDeploy quant config API
        return self.name_value


class _Gate(paddle.nn.Layer):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):  # noqa: D401 - deterministic gating scores
        token_num = x.shape[0]
        base = paddle.arange(self.num_experts, dtype="float32")
        return paddle.tile(base.unsqueeze(0), [token_num, 1])


class _DummyMoELayer(paddle.nn.Layer):
    def __init__(self, quant_config: _FakeQuantConfig, *, load_choice: str = "default_v1"):
        super().__init__()
        self.num_local_experts = 2
        self.num_experts = self.num_local_experts
        self.moe_intermediate_size = 3
        self.hidden_size = 4
        self.top_k = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = 0.0
        self.topk_method = "noaux_tc"
        self.reduce_results = True
        self.tp_size = 2
        self.weight_dtype = "float32"
        self.fd_config = types.SimpleNamespace(load_config=types.SimpleNamespace(load_choices=load_choice))
        self.quant_method = types.SimpleNamespace(quant_config=quant_config)
        self.weight_key_map = {
            "up_gate_proj_expert_weight_scale_key": "up_scale_{}",
            "down_proj_expert_weight_scale_key": "down_scale_{}",
            "up_gate_proj_expert_in_scale_key": "up_in_scale_{}",
            "down_proj_expert_in_scale_key": "down_in_scale_{}",
        }
        self._up_weights = self._build_weight_list([self.hidden_size, self.moe_intermediate_size * 2])
        self._down_weights = self._build_weight_list([self.moe_intermediate_size, self.hidden_size])

    def _build_weight_list(self, shape):
        total = 1
        for dim in shape:
            total *= dim
        base = paddle.arange(total * self.num_local_experts, dtype="float32")
        return [base[i * total : (i + 1) * total].reshape(shape) for i in range(self.num_local_experts)]

    def set_expert_weights(self, up_list, down_list):
        self._up_weights = up_list
        self._down_weights = down_list

    def extract_moe_ffn_weights(self, _state_dict):
        return list(self._up_weights), list(self._down_weights), None, None


@pytest.fixture(scope="module")
def fused_backend_module():
    kernel = _RecordingKernel()
    kernels_mod = types.ModuleType("fastdeploy.model_executor.layers.moe.triton_moe_kernels")
    kernels_mod.fused_moe_kernel_paddle = kernel
    sys.modules["fastdeploy.model_executor.layers.moe.triton_moe_kernels"] = kernels_mod

    gpu_mod = types.ModuleType("fastdeploy.model_executor.ops.gpu")
    gpu_mod.tritonmoe_preprocess_func = _fake_preprocess
    gpu_mod.moe_topk_select = _fake_topk_select
    gpu_mod.per_token_quant = _fake_per_token_quant
    gpu_mod.moe_fused_hadamard_quant_fp8 = lambda tensor, **_kwargs: tensor

    def _dynamic_per_token_quant(output, input_tensor, scale_tensor, _scale_ub):
        output.set_value(input_tensor.astype(paddle.float16))
        scale_tensor.set_value(paddle.ones_like(scale_tensor))

    gpu_mod.dynamic_per_token_scaled_fp8_quant = _dynamic_per_token_quant
    sys.modules["fastdeploy.model_executor.ops.gpu"] = gpu_mod
    ops_pkg = types.ModuleType("fastdeploy.model_executor.ops")
    ops_pkg.gpu = gpu_mod
    sys.modules["fastdeploy.model_executor.ops"] = ops_pkg
    fastdeploy_executor = importlib.import_module("fastdeploy.model_executor")
    setattr(fastdeploy_executor, "ops", ops_pkg)

    if not hasattr(paddle, "incubate"):
        paddle.incubate = types.SimpleNamespace()
    if not hasattr(paddle.incubate, "nn"):
        paddle.incubate.nn = types.SimpleNamespace()
    if not hasattr(paddle.incubate.nn, "functional"):
        paddle.incubate.nn.functional = types.SimpleNamespace()
    paddle.incubate.nn.functional.swiglu = lambda tensor: tensor

    module = importlib.import_module("fastdeploy.model_executor.layers.moe.fused_moe_triton_backend")

    quant_ops = importlib.import_module("fastdeploy.model_executor.layers.quantization.ops")
    quant_ops.scaled_fp8_quant = _fake_scaled_fp8_quant

    module.get_moe_scores = _fake_get_moe_scores
    module.tensor_model_parallel_all_reduce = lambda tensor: tensor
    module.set_weight_attrs = lambda *_args, **_kwargs: None
    module.process_weight_transpose = lambda *_args, **_kwargs: None
    module.free_tensor = lambda *_args, **_kwargs: None
    module.weight_fully_copied = lambda *_args, **_kwargs: True

    class _Tracker:
        def __init__(self, shape, output_dim=True):
            self.shape = shape
            self.output_dim = output_dim

    module.TensorTracker = _Tracker

    return module, kernel


def _assert_tensor_contents(tensor, expected_shape):
    assert list(tensor.shape) == expected_shape
    assert paddle.is_tensor(tensor)


def test_weight_only_method_end_to_end(fused_backend_module):
    module, kernel = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wint8", is_checkpoint_bf16=False)
    layer = _DummyMoELayer(quant_config)
    method = module.TritonWeightOnlyMoEMethod(quant_config)

    method.create_weights(layer, model_format="torch")
    method.process_loaded_weights(layer, {})
    method.process_weights_after_loading(layer)

    gate = _Gate(layer.num_local_experts)
    x = paddle.ones([3, layer.hidden_size], dtype="float32")
    out = method.apply(layer, x, gate)

    _assert_tensor_contents(out, [3, layer.hidden_size])
    assert kernel.calls, "kernel launches were recorded"


def test_weight_only_method_handles_bf16_conversion(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wint8", is_checkpoint_bf16=True)
    layer = _DummyMoELayer(quant_config)
    method = module.TritonWeightOnlyMoEMethod(quant_config)

    tracker_values = []

    class _Tracker:
        def __init__(self, shape, output_dim):
            tracker_values.append((shape, output_dim))

    module.TensorTracker = _Tracker

    method.create_weights(layer, model_format="torch")
    method.process_weights_after_loading(layer)

    assert tracker_values, "TensorTracker constructed during setup"


def test_weight_only_transposes_quantized_weights(fused_backend_module):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wint8", is_checkpoint_bf16=False)
    layer = _DummyMoELayer(quant_config)
    method = module.TritonWeightOnlyMoEMethod(quant_config)

    method.create_weights(layer, model_format="custom")
    method.process_weights_after_loading(layer)

    assert getattr(layer, method.added_weight_attrs[0]).dtype == paddle.int8


def test_weight_only_checkpoint_loader_non_torch(fused_backend_module):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wint8", is_checkpoint_bf16=True)
    layer = _DummyMoELayer(quant_config)
    method = module.TritonWeightOnlyMoEMethod(quant_config)

    method.create_weights(layer, model_format="custom")
    method.process_weights_after_loading(layer)

    assert hasattr(layer, "up_gate_proj_weight")


def _build_scale_state_dict(layer):
    state_dict = {}
    for idx in range(layer.num_local_experts):
        for key in [
            "up_gate_proj_expert_weight_scale_key",
            "down_proj_expert_weight_scale_key",
            "up_gate_proj_expert_in_scale_key",
            "down_proj_expert_in_scale_key",
        ]:
            state_dict[layer.weight_key_map[key].format(idx)] = paddle.ones([1], dtype="float32")
    return state_dict


def test_wfp8afp8_prefill_and_decode(fused_backend_module):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wfp8afp8")
    layer = _DummyMoELayer(quant_config)
    method = module.Wfp8Afp8MoEMethod(quant_config)

    method.create_weights(layer, model_format="torch")

    gate = _Gate(layer.num_local_experts)
    x = paddle.ones([2, layer.hidden_size], dtype="float32")
    out = method.apply(layer, x, gate)
    _assert_tensor_contents(out, [2, layer.hidden_size])


def test_wfp8afp8_checkpoint_loader(fused_backend_module):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="wfp8afp8", is_checkpoint_bf16=True)
    layer = _DummyMoELayer(quant_config)
    method = module.Wfp8Afp8MoEMethod(quant_config)

    method.create_weights(layer, model_format="custom")

    assert layer.up_gate_proj_weight.shape[1] == layer.hidden_size


def test_tensorwise_fp8_quant_paths(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="tensor_fp8", weight_block_size=(2, 2))
    layer = _DummyMoELayer(quant_config)
    method = module.TensorWiseFP8MoEMethod(quant_config)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        _fake_per_block_cast_to_fp8,
    )

    method.create_weights(layer, model_format="torch")
    method.process_loaded_weights(layer, {})

    gate = _Gate(layer.num_local_experts)
    x = paddle.ones([2, layer.hidden_size], dtype="float32")
    out = method.apply(layer, x, gate)
    _assert_tensor_contents(out, [2, layer.hidden_size])


def test_tensorwise_process_prequanted_weights(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="tensor_fp8")
    layer = _DummyMoELayer(quant_config)
    method = module.TensorWiseFP8MoEMethod(quant_config)

    method.create_weights(layer, model_format="torch")
    monkeypatch.setattr(layer, "extract_moe_ffn_weights", lambda _state: (layer._up_weights, layer._down_weights))
    state_dict = _build_scale_state_dict(layer)
    method.process_prequanted_weights(layer, state_dict)

    assert state_dict == {}


def test_blockwise_fp8_runtime(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="block_fp8", weight_block_size=(2, 2))
    layer = _DummyMoELayer(quant_config)
    method = module.BlockWiseFP8MoEMethod(quant_config)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        _fake_per_block_cast_to_fp8,
    )

    method.create_weights(layer, model_format="torch")
    method.process_weights_after_loading(layer)

    gate = _Gate(layer.num_local_experts)
    x = paddle.ones([2, layer.hidden_size], dtype="float32")
    out = method.apply(layer, x, gate)
    _assert_tensor_contents(out, [2, layer.hidden_size])


def test_blockwise_non_torch_checkpoint_loader(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="block_fp8", is_checkpoint_bf16=True, weight_block_size=(2, 2))
    layer = _DummyMoELayer(quant_config)
    method = module.BlockWiseFP8MoEMethod(quant_config)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        _fake_per_block_cast_to_fp8,
    )

    method.create_weights(layer, model_format="custom")
    method.process_weights_after_loading(layer)

    assert getattr(layer, method.added_scale_attrs[0]).shape[0] == layer.num_local_experts


def test_blockwise_v0_loader_path(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="block_fp8", weight_block_size=(2, 2))
    layer = _DummyMoELayer(quant_config, load_choice="legacy_v0")
    method = module.BlockWiseFP8MoEMethod(quant_config)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        _fake_per_block_cast_to_fp8,
    )

    method.create_weights(layer, model_format="torch")

    assert method.weight_dtype == paddle.float8_e4m3fn


def test_blockwise_process_loaded_weights(fused_backend_module, monkeypatch):
    module, _ = fused_backend_module
    quant_config = _FakeQuantConfig(name_value="block_fp8", weight_block_size=(2, 2))
    layer = _DummyMoELayer(quant_config)
    method = module.BlockWiseFP8MoEMethod(quant_config)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        _fake_per_block_cast_to_fp8,
    )

    method.create_weights(layer, model_format="torch")
    method.process_loaded_weights(layer, {})

    assert getattr(layer, method.added_scale_attrs[0]).dtype == paddle.float32
